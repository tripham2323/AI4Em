import math

import pytest

from src.calibration.manager import CalibrationManager, CalibrationState
from src.calibration.profile import estimate_profile, fit_qc_policy
from src.contracts import CalibrationProfile, FeatureSample


def sample(timestamp_ms: int, *, valid: bool = True) -> FeatureSample:
    value = 0.3 if valid else math.nan
    return FeatureSample(
        timestamp_ms=timestamp_ms,
        frame_index=timestamp_ms // 100,
        source_id="camera:0",
        ear_left=value,
        ear_right=value,
        ear_mean=value,
        mar=0.1 if valid else math.nan,
        pitch=0.0 if valid else math.nan,
        yaw=0.0 if valid else math.nan,
        roll=0.0 if valid else math.nan,
        face_detected=valid,
        left_eye_valid=valid,
        right_eye_valid=valid,
        mouth_valid=valid,
        pose_valid=valid,
        reprojection_error_norm=0.0 if valid else math.nan,
    )


def profile(*, mode="P1", schema="facial_features_v1", asset="asset", size=(640, 480)):
    return CalibrationProfile(
        mode=mode,
        valid=True,
        ear_left_baseline=0.31,
        ear_right_baseline=0.27,
        mar_baseline=0.08,
        pitch_baseline=1.0,
        yaw_baseline=-2.0,
        roll_baseline=0.5,
        schema_version=schema,
        asset_sha256=asset,
        image_size=size,
        quality_stats={"mad_ratio_left": 0.04},
    )


def estimator(samples, **context):
    assert samples
    return profile(
        mode=context["mode"],
        schema="facial_features_v1",
        asset=context["asset_sha256"],
        size=context["image_size"],
    )


def manager(**overrides):
    values = {
        "mode": "P1",
        "schema_version": "facial_features_v1",
        "asset_sha256": "asset",
        "image_size": (640, 480),
        "profile_estimator": estimator,
        "calibration_seconds": 30,
        "min_valid_seconds": 20,
        "timeout_seconds": 60,
        "max_sample_age_ms": 100,
    }
    values.update(overrides)
    return CalibrationManager(**values)


def test_completes_only_after_wall_clock_and_valid_duration():
    calibration = manager()
    assert calibration.start(0).state is CalibrationState.COLLECTING
    result = None
    for timestamp in range(100, 30_001, 100):
        result = calibration.update(sample(timestamp))
    assert result is not None
    assert result.state is CalibrationState.COMPLETE
    assert result.elapsed_ms == 30_000
    assert result.valid_duration_ms == 29_900
    assert result.profile is not None
    assert result.profile.ear_left_baseline != result.profile.ear_right_baseline


def test_invalid_samples_do_not_complete_and_timeout_fails_without_fallback():
    calibration = manager()
    calibration.start(0)
    for timestamp in range(100, 60_001, 100):
        result = calibration.update(sample(timestamp, valid=False))
    assert result.state is CalibrationState.FAILED
    assert result.profile is None
    assert "timed out" in result.reason


def test_retry_resets_samples_timing_and_failure():
    calibration = manager(calibration_seconds=1, min_valid_seconds=0.5, timeout_seconds=2)
    calibration.start(0)
    calibration.update(sample(100, valid=False))
    calibration.invalidate("camera changed")
    restarted = calibration.start(10_000)
    assert restarted.state is CalibrationState.COLLECTING
    assert restarted.valid_samples == 0
    assert restarted.valid_duration_ms == 0
    for timestamp in range(10_100, 11_001, 100):
        result = calibration.update(sample(timestamp))
    assert result.state is CalibrationState.COMPLETE


@pytest.mark.parametrize("field,value", [
    ("mode", "P0"),
    ("schema_version", "other"),
    ("asset_sha256", "other"),
    ("image_size", (320, 240)),
])
def test_incompatible_estimated_profile_fails(field, value):
    def incompatible(samples, **context):
        values = {"mode": "P1", "schema": "facial_features_v1", "asset": "asset", "size": (640, 480)}
        mapping = {"schema_version": "schema", "asset_sha256": "asset", "image_size": "size"}
        values[mapping.get(field, field)] = value
        return profile(**values)

    calibration = manager(
        profile_estimator=incompatible,
        calibration_seconds=1,
        min_valid_seconds=0.5,
        timeout_seconds=2,
    )
    calibration.start(0)
    for timestamp in range(100, 1_001, 100):
        result = calibration.update(sample(timestamp))
    assert result.state is CalibrationState.FAILED
    assert result.profile is None


def test_p0_uses_only_an_explicit_compatible_population_profile():
    calibration = CalibrationManager(
        mode="P0",
        schema_version="facial_features_v1",
        asset_sha256="asset",
        image_size=(640, 480),
        population_profile=profile(mode="P0"),
    )
    result = calibration.start(123)
    assert result.state is CalibrationState.COMPLETE
    assert result.profile is not None
    assert result.profile.mode == "P0"


def test_timestamps_must_increase():
    calibration = manager()
    calibration.start(100)
    with pytest.raises(ValueError, match="strictly increase"):
        calibration.update(sample(100))


def test_timeout_is_absolute_even_when_late_sample_would_reach_valid_minimum():
    calibration = manager(calibration_seconds=1, min_valid_seconds=0.5, timeout_seconds=2)
    calibration.start(0)
    for timestamp in range(100, 701, 100):
        calibration.update(sample(timestamp))
    result = calibration.update(sample(2501))
    assert result.state is CalibrationState.FAILED
    assert result.profile is None
    assert "timed out" in result.reason


def test_calibration_cannot_join_sources_or_reuse_frame_indices():
    calibration = manager()
    calibration.start(0)
    calibration.update(sample(100))
    changed_source = sample(200)
    changed_source = FeatureSample(
        changed_source.timestamp_ms, changed_source.frame_index, "camera:1",
        changed_source.ear_left, changed_source.ear_right, changed_source.ear_mean,
        changed_source.mar, changed_source.pitch, changed_source.yaw, changed_source.roll,
        changed_source.face_detected, changed_source.left_eye_valid, changed_source.right_eye_valid,
        changed_source.mouth_valid, changed_source.pose_valid, changed_source.reprojection_error_norm,
    )
    with pytest.raises(ValueError, match="different sources"):
        calibration.update(changed_source)

    calibration.start(1000)
    first = sample(1100)
    calibration.update(first)
    repeated_frame = FeatureSample(
        1200, first.frame_index, first.source_id,
        first.ear_left, first.ear_right, first.ear_mean, first.mar,
        first.pitch, first.yaw, first.roll, first.face_detected,
        first.left_eye_valid, first.right_eye_valid, first.mouth_valid,
        first.pose_valid, first.reprojection_error_norm,
    )
    with pytest.raises(ValueError, match="frame indices"):
        calibration.update(repeated_frame)


def test_manager_accepts_phase9_estimator_and_frozen_qc_policy():
    training_rows = [sample(timestamp) for timestamp in range(0, 30_001, 100)]
    calibration = CalibrationManager(
        mode="P1",
        schema_version="facial_features_v1",
        asset_sha256="asset",
        image_size=(640, 480),
        profile_estimator=estimate_profile,
        profile_estimator_kwargs={"qc_policy": fit_qc_policy(training_rows)},
    )
    calibration.start(0)
    for timestamp in range(100, 30_001, 100):
        result = calibration.update(sample(timestamp))
    assert result.state is CalibrationState.COMPLETE
    assert result.profile is not None and result.profile.valid
