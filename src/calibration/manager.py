"""Session calibration lifecycle without model fine-tuning.

Profile estimation is deliberately injected.  Phase 9 owns the statistical
estimator; this module owns collection time, validity accounting, retries and
profile compatibility for realtime sessions.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Protocol

from src.contracts import CalibrationProfile, FeatureSample


class CalibrationState(StrEnum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    state: CalibrationState
    mode: str
    elapsed_ms: int
    valid_duration_ms: int
    valid_samples: int
    progress: float
    reason: str
    profile: CalibrationProfile | None


class ProfileEstimator(Protocol):
    def __call__(
        self,
        samples: Sequence[FeatureSample],
        *,
        mode: str,
        asset_sha256: str,
        image_size: tuple[int, int],
        **kwargs: Any,
    ) -> CalibrationProfile: ...


def _positive_milliseconds(value: Real, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    milliseconds = round(float(value) * 1000.0)
    if milliseconds <= 0:
        raise ValueError(f"{name} must be positive")
    return milliseconds


def _timestamp(value: int, name: str = "timestamp_ms") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _valid_image_size(value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, Integral) or item <= 0 for item in value)
    ):
        raise ValueError("image_size must be positive integer (W, H)")
    return int(value[0]), int(value[1])


def _sample_is_valid(sample: FeatureSample) -> bool:
    measurements = (
        sample.ear_left,
        sample.ear_right,
        sample.mar,
        sample.pitch,
        sample.yaw,
        sample.roll,
    )
    return bool(
        sample.face_detected
        and sample.left_eye_valid
        and sample.right_eye_valid
        and sample.mouth_valid
        and sample.pose_valid
        and all(math.isfinite(value) for value in measurements)
    )


class CalibrationManager:
    """Collect and freeze one compatible P0/P1 calibration profile per session."""

    def __init__(
        self,
        *,
        mode: str,
        schema_version: str,
        asset_sha256: str,
        image_size: tuple[int, int],
        profile_estimator: ProfileEstimator | None = None,
        profile_estimator_kwargs: dict[str, Any] | None = None,
        population_profile: CalibrationProfile | None = None,
        calibration_seconds: Real = 30,
        min_valid_seconds: Real = 20,
        timeout_seconds: Real = 60,
        max_sample_age_ms: int = 100,
    ) -> None:
        if mode not in {"P0", "P1"}:
            raise ValueError("mode must be P0 or P1")
        if not isinstance(schema_version, str) or not schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(asset_sha256, str) or not asset_sha256:
            raise ValueError("asset_sha256 must be a non-empty string")
        self._target_ms = _positive_milliseconds(calibration_seconds, "calibration_seconds")
        self._min_valid_ms = _positive_milliseconds(min_valid_seconds, "min_valid_seconds")
        self._timeout_ms = _positive_milliseconds(timeout_seconds, "timeout_seconds")
        if self._min_valid_ms > self._target_ms or self._target_ms > self._timeout_ms:
            raise ValueError("require min_valid_seconds <= calibration_seconds <= timeout_seconds")
        if (
            isinstance(max_sample_age_ms, bool)
            or not isinstance(max_sample_age_ms, Integral)
            or max_sample_age_ms <= 0
        ):
            raise ValueError("max_sample_age_ms must be a positive integer")
        if mode == "P1" and profile_estimator is None:
            raise ValueError("P1 requires a Phase 9 profile_estimator")
        if mode == "P0" and population_profile is None:
            raise ValueError("P0 requires a population_profile")

        self.mode = mode
        self.schema_version = schema_version
        self.asset_sha256 = asset_sha256
        self.image_size = _valid_image_size(image_size)
        self._profile_estimator = profile_estimator
        self._profile_estimator_kwargs = dict(profile_estimator_kwargs or {})
        self._population_profile = population_profile
        self._max_sample_age_ms = int(max_sample_age_ms)
        self.reset()

    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def profile(self) -> CalibrationProfile | None:
        return self._profile

    @property
    def is_complete(self) -> bool:
        return self._state is CalibrationState.COMPLETE and self._profile is not None

    @property
    def snapshot(self) -> CalibrationSnapshot:
        elapsed = 0 if self._start_ms is None or self._last_ms is None else self._last_ms - self._start_ms
        progress = min(1.0, elapsed / self._target_ms) if self._target_ms else 0.0
        return CalibrationSnapshot(
            state=self._state,
            mode=self.mode,
            elapsed_ms=elapsed,
            valid_duration_ms=self._valid_duration_ms,
            valid_samples=len(self._samples),
            progress=progress,
            reason=self._reason,
            profile=self._profile,
        )

    def start(self, timestamp_ms: int) -> CalibrationSnapshot:
        started = _timestamp(timestamp_ms)
        self.reset()
        self._start_ms = self._last_ms = started
        if self.mode == "P0":
            assert self._population_profile is not None
            self._profile = self._validate_profile(self._population_profile)
            self._state = CalibrationState.COMPLETE
            self._reason = "population profile ready"
        else:
            self._state = CalibrationState.COLLECTING
            self._reason = "collecting calibration samples"
        return self.snapshot

    def update(self, sample: FeatureSample) -> CalibrationSnapshot:
        if self.mode == "P0":
            if not self.is_complete:
                raise RuntimeError("start calibration before update")
            return self.snapshot
        if self._state is not CalibrationState.COLLECTING:
            raise RuntimeError("calibration is not collecting")
        timestamp_ms = _timestamp(sample.timestamp_ms)
        if not isinstance(sample.source_id, str) or not sample.source_id:
            raise ValueError("calibration source_id must be a non-empty string")
        frame_index = _timestamp(sample.frame_index, "frame_index")
        assert self._start_ms is not None and self._last_ms is not None
        if timestamp_ms <= self._last_ms:
            raise ValueError("calibration timestamps must strictly increase")
        if self._source_id is None:
            self._source_id = sample.source_id
        elif sample.source_id != self._source_id:
            raise ValueError("calibration cannot join different sources")
        if self._frame_index is not None and frame_index <= self._frame_index:
            raise ValueError("calibration frame indices must strictly increase")
        self._frame_index = frame_index

        previous_valid = self._last_sample_valid
        current_valid = _sample_is_valid(sample)
        delta_ms = timestamp_ms - self._last_ms
        if previous_valid and current_valid:
            self._valid_duration_ms += min(delta_ms, self._max_sample_age_ms)
        if current_valid:
            self._samples.append(sample)
        self._last_sample_valid = current_valid
        self._last_ms = timestamp_ms

        elapsed_ms = timestamp_ms - self._start_ms
        if elapsed_ms > self._timeout_ms:
            return self._fail("calibration timed out before profile completion")
        if elapsed_ms >= self._target_ms and self._valid_duration_ms >= self._min_valid_ms:
            return self.finish()
        if elapsed_ms >= self._timeout_ms:
            return self._fail("calibration timed out before enough valid data")
        return self.snapshot

    def finish(self) -> CalibrationSnapshot:
        if self._state is not CalibrationState.COLLECTING:
            raise RuntimeError("calibration is not collecting")
        assert self._start_ms is not None and self._last_ms is not None
        if self._last_ms - self._start_ms > self._timeout_ms:
            return self._fail("calibration timed out before profile completion")
        if self._last_ms - self._start_ms < self._target_ms:
            raise RuntimeError("calibration target duration has not elapsed")
        if self._valid_duration_ms < self._min_valid_ms:
            raise RuntimeError("calibration does not have enough valid duration")
        assert self._profile_estimator is not None
        try:
            profile = self._profile_estimator(
                tuple(self._samples),
                mode=self.mode,
                asset_sha256=self.asset_sha256,
                image_size=self.image_size,
                **self._profile_estimator_kwargs,
            )
            self._profile = self._validate_profile(profile)
        # Phase 9 is an injected boundary; any estimator failure must become a
        # retryable calibration failure instead of terminating the live loop.
        except Exception as exc:  # noqa: BLE001
            return self._fail(f"profile estimation failed: {exc}")
        self._state = CalibrationState.COMPLETE
        self._reason = "calibration complete"
        return self.snapshot

    def invalidate(self, reason: str) -> CalibrationSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        return self._fail(reason.strip())

    def reset(self) -> CalibrationSnapshot:
        self._state = CalibrationState.IDLE
        self._start_ms: int | None = None
        self._last_ms: int | None = None
        self._last_sample_valid = False
        self._valid_duration_ms = 0
        self._samples: list[FeatureSample] = []
        self._source_id: str | None = None
        self._frame_index: int | None = None
        self._profile: CalibrationProfile | None = None
        self._reason = "not started"
        return self.snapshot

    def _fail(self, reason: str) -> CalibrationSnapshot:
        self._state = CalibrationState.FAILED
        self._profile = None
        self._reason = reason
        return self.snapshot

    def _validate_profile(self, profile: CalibrationProfile) -> CalibrationProfile:
        if not isinstance(profile, CalibrationProfile):
            raise TypeError("profile estimator must return CalibrationProfile")
        if not profile.valid:
            raise ValueError("profile is invalid")
        if profile.mode != self.mode:
            raise ValueError("profile mode does not match calibration mode")
        if profile.schema_version != self.schema_version:
            raise ValueError("profile schema does not match runtime schema")
        if profile.asset_sha256 != self.asset_sha256:
            raise ValueError("profile asset does not match runtime asset")
        if profile.image_size != self.image_size:
            raise ValueError("profile image size does not match camera")
        baselines = (
            profile.ear_left_baseline,
            profile.ear_right_baseline,
            profile.mar_baseline,
            profile.pitch_baseline,
            profile.yaw_baseline,
            profile.roll_baseline,
        )
        if not all(math.isfinite(value) for value in baselines):
            raise ValueError("profile baselines must be finite")
        if profile.ear_left_baseline <= 0 or profile.ear_right_baseline <= 0:
            raise ValueError("eye baselines must be positive")
        return profile
