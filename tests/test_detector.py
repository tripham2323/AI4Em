from types import SimpleNamespace

import numpy as np
import pytest

from src.calibration.manager import CalibrationManager
from src.contracts import (
    CalibrationProfile,
    FeatureSample,
    FramePacket,
    SystemStatus,
    TemporalSample,
)
from src.realtime.buffer import PredictionBuffer
from src.realtime.detector import DrowsinessDetector
from src.realtime.smoother import PredictionSmoother

FEATURES = tuple(f"f{i}" for i in range(16))


def population_profile():
    return CalibrationProfile("P0", True, .3, .28, .1, 0., 0., 0.,
                              "facial_features_v1", "asset", (2, 2), {})


class Pipeline:
    def __init__(self):
        self.valid = True
        self.processed = 0
        self.resets = 0
        self.closed = 0

    def process(self, packet):
        self.processed += 1
        value = .3 if self.valid else np.nan
        return FeatureSample(packet.timestamp_ms, packet.frame_index, packet.source_id,
            value, value, value, .1 if self.valid else np.nan,
            0. if self.valid else np.nan, 0. if self.valid else np.nan, 0. if self.valid else np.nan,
            self.valid, self.valid, self.valid, self.valid, self.valid, 0.)

    def reset(self): self.resets += 1
    def close(self): self.closed += 1


class Temporal:
    def __init__(self): self.resets = 0
    def update(self, sample, profile):
        values = np.full(16, .5, dtype=np.float32)
        validity = np.full(4, sample.face_detected, dtype=np.bool_)
        return [TemporalSample(sample.timestamp_ms, values, validity, 0, {})]
    def reset(self): self.resets += 1


class Model:
    model_id = "test-model"
    feature_names = FEATURES
    schema_version = "facial_features_v1"
    calibration_mode = "P0"
    def __init__(self): self.calls = 0
    def predict_proba(self, x):
        self.calls += 1
        assert x.shape == (1, 100, 16)
        return np.array([[.2, .3, .5]], dtype=np.float32)


def make_detector(now=None):
    pipeline = Pipeline()
    temporal = Temporal()
    model = Model()
    calibration = CalibrationManager(mode="P0", schema_version="facial_features_v1",
        asset_sha256="asset", image_size=(2, 2), population_profile=population_profile())
    feature_buffer = PredictionBuffer(feature_names=FEATURES, schema_version="facial_features_v1",
        sequence_steps=100, max_missing_ratio=.2, max_gap_ms=1000,
        transformer=lambda values, validity: values)
    clock_fn = now or (lambda: clock.current)
    detector = DrowsinessDetector(pipeline=pipeline, calibration=calibration,
        temporal=temporal, buffer=feature_buffer, model=model, clock_ms=clock_fn)
    return detector, pipeline, temporal, model


clock = SimpleNamespace(current=0)


def packet(timestamp):
    return FramePacket(np.zeros((2, 2, 3), dtype=np.uint8), timestamp, timestamp // 100, "camera:0")


def test_warmup_then_predicts_every_second_with_fixed_class_order():
    detector, _, _, model = make_detector()
    result = None
    for timestamp in range(0, 10_000, 100):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
    assert result is not None
    assert result.system_status is SystemStatus.READY
    assert result.raw_prediction is not None
    assert result.raw_prediction.class_id.value == 2
    assert model.calls == 1
    for timestamp in range(10_000, 10_900, 100):
        clock.current = timestamp
        detector.process(packet(timestamp))
    assert model.calls == 1
    clock.current = 10_900
    detector.process(packet(10_900))
    assert model.calls == 2


def test_stale_frame_is_unavailable_without_running_pipeline():
    detector, pipeline, _, _ = make_detector(now=lambda: 1000)
    result = detector.process(packet(0))
    assert result.system_status is SystemStatus.UNRELIABLE
    assert result.quality["reason"] == "stale frame"
    assert pipeline.processed == 0
    assert detector.last_feature_sample is None


def test_last_feature_sample_tracks_processed_packet_and_clears_on_reset_close():
    detector, _, _, _ = make_detector()
    clock.current = 0
    detector.process(packet(0))
    assert detector.last_feature_sample is not None
    assert detector.last_feature_sample.timestamp_ms == 0
    assert detector.last_temporal_sample is not None
    assert detector.last_temporal_sample.timestamp_ms == 0
    detector.reset_session()
    assert detector.last_feature_sample is None
    assert detector.last_temporal_sample is None
    detector.close()
    assert detector.last_feature_sample is None
    assert detector.last_temporal_sample is None


def test_replay_can_supply_source_time_without_using_wall_clock():
    detector, pipeline, _, _ = make_detector(now=lambda: 99_999)
    result = detector.process(packet(0), current_time_ms=0)
    assert result.system_status is SystemStatus.WARMING_UP
    assert pipeline.processed == 1


def test_results_expose_calibration_progress_and_message_for_ui():
    detector, _, _, _ = make_detector()
    clock.current = 0
    result = detector.process(packet(0))
    assert result.quality["calibration_progress"] == 1.0
    assert result.quality["calibration_msg"] == "population profile ready"


def test_no_face_is_immediate_and_long_gap_resets_history():
    detector, pipeline, temporal, _ = make_detector()
    clock.current = 0
    detector.process(packet(0))
    pipeline.valid = False
    for timestamp in (100, 1201):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
    assert result.system_status is SystemStatus.NO_FACE
    assert temporal.resets >= 1
    assert len(detector.buffer) <= 1


def test_reset_and_close_release_owned_dependencies():
    detector, pipeline, temporal, _ = make_detector()
    detector.reset_session()
    assert pipeline.resets == 1
    assert temporal.resets == 1
    detector.close()
    detector.close()
    assert pipeline.closed == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_names", tuple(reversed(FEATURES)), "feature order"),
        ("schema_version", "other-schema", "schema differ"),
        ("calibration_mode", "P1", "calibration mode differ"),
    ],
)
def test_model_contract_mismatch_is_rejected_before_processing(field, value, message):
    detector, pipeline, temporal, model = make_detector()
    setattr(model, field, value)
    with pytest.raises(ValueError, match=message):
        DrowsinessDetector(pipeline=pipeline, calibration=detector.calibration,
            temporal=temporal, buffer=detector.buffer, model=model)


def test_pipeline_contract_rejects_non_feature_sample():
    detector, pipeline, _, _ = make_detector()
    pipeline.process = lambda packet: object()
    clock.current = 0
    with pytest.raises(TypeError, match="FeatureSample"):
        detector.process(packet(0))
    assert detector.last_feature_sample is None


def test_p1_first_frame_starts_collection_without_duplicate_timestamp_update():
    pipeline = Pipeline()
    temporal = Temporal()
    model = Model()
    model.calibration_mode = "P1"
    calibration = CalibrationManager(mode="P1", schema_version="facial_features_v1",
        asset_sha256="asset", image_size=(2, 2), profile_estimator=lambda samples, **context: None)
    feature_buffer = PredictionBuffer(feature_names=FEATURES, schema_version="facial_features_v1",
        sequence_steps=100, max_missing_ratio=.2, max_gap_ms=1000,
        transformer=lambda values, validity: values)
    detector = DrowsinessDetector(pipeline=pipeline, calibration=calibration,
        temporal=temporal, buffer=feature_buffer, model=model, clock_ms=lambda: 0)
    result = detector.process(packet(0))
    assert result.system_status is SystemStatus.CALIBRATING
    assert calibration.snapshot.valid_samples == 0


def test_p1_completion_resets_history_before_warmup_and_prediction():
    pipeline = Pipeline()
    temporal = Temporal()
    model = Model()
    model.calibration_mode = "P1"
    model_calls = []
    def predict_two_steps(x):
        model_calls.append(x.shape)
        return np.array([[.2, .3, .5]], dtype=np.float32)
    model.predict_proba = predict_two_steps

    def estimate_profile(samples, **context):
        return CalibrationProfile("P1", True, .3, .28, .1, 0., 0., 0.,
            "facial_features_v1", context["asset_sha256"], context["image_size"], {})

    calibration = CalibrationManager(mode="P1", schema_version="facial_features_v1",
        asset_sha256="asset", image_size=(2, 2), profile_estimator=estimate_profile,
        calibration_seconds=1, min_valid_seconds=.5, timeout_seconds=2)
    feature_buffer = PredictionBuffer(feature_names=FEATURES, schema_version="facial_features_v1",
        sequence_steps=2, max_missing_ratio=.2, max_gap_ms=1000,
        transformer=lambda values, validity: values)
    detector = DrowsinessDetector(pipeline=pipeline, calibration=calibration,
        temporal=temporal, buffer=feature_buffer, model=model, prediction_interval_ms=100,
        clock_ms=lambda: clock.current)

    for timestamp in range(0, 1001, 100):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
    assert result.system_status is SystemStatus.WARMING_UP
    assert calibration.is_complete
    assert temporal.resets == 1
    assert len(feature_buffer) == 0

    for timestamp in (1100, 1200):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
    assert result.raw_prediction is not None
    assert model_calls == [(1, 2, 16)]


def test_invalid_model_probability_becomes_error_not_warning_or_exception():
    detector, _, _, model = make_detector()
    model.predict_proba = lambda x: np.array([[np.nan, 0.5, 0.5]], dtype=np.float32)
    for timestamp in range(0, 10_000, 100):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
    assert result.system_status is SystemStatus.ERROR
    assert result.raw_prediction is None
    assert "model inference failed" in result.quality["reason"]


@pytest.mark.parametrize("probabilities", [
    [[-.1, .6, .5]],
    [[.2, .2, .2]],
    [[.2, .8]],
])
def test_other_invalid_model_outputs_become_error(probabilities):
    detector, _, _, model = make_detector()
    model.predict_proba = lambda x: np.asarray(probabilities, dtype=np.float32)
    for timestamp in range(0, 10_000, 100):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
    assert result.system_status is SystemStatus.ERROR
    assert result.raw_prediction is None


def test_detector_exposes_raw_immediately_and_smooth_only_after_three_predictions():
    detector, _, _, _ = make_detector()
    detector.smoother = PredictionSmoother(samples=3, expiry_ms=2000)
    predictions = []
    for timestamp in range(0, 12_000, 100):
        clock.current = timestamp
        result = detector.process(packet(timestamp))
        if result.raw_prediction is not None:
            predictions.append(result)
    assert len(predictions) == 3
    assert predictions[0].smoothed_prediction is None
    assert predictions[1].smoothed_prediction is None
    assert predictions[2].smoothed_prediction is not None
