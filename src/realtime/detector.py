"""Realtime orchestration around injected Phase 9/10/13 dependencies."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from time import perf_counter_ns
from typing import Any, Protocol

import numpy as np

from src.calibration.manager import CalibrationState
from src.contracts import (
    DetectionResult,
    DriverState,
    FeatureSample,
    FramePacket,
    Prediction,
    SequenceWindow,
    SystemStatus,
    TemporalSample,
)
from src.realtime.buffer import PredictionBuffer


class RawFeaturePipeline(Protocol):
    def process(self, packet: FramePacket) -> FeatureSample: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class TemporalExtractor(Protocol):
    def update(self, sample: FeatureSample, profile: Any) -> Sequence[TemporalSample]: ...
    def reset(self) -> None: ...


class ModelRuntime(Protocol):
    model_id: str
    feature_names: Sequence[str]
    schema_version: str
    calibration_mode: str

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


def _clock_ms() -> int:
    return perf_counter_ns() // 1_000_000


class DrowsinessDetector:
    """Turn source packets into current, timestamped three-class detections."""

    def __init__(
        self,
        *,
        pipeline: RawFeaturePipeline,
        calibration: Any,
        temporal: TemporalExtractor,
        buffer: PredictionBuffer,
        model: ModelRuntime,
        prediction_interval_ms: int = 1000,
        stale_ms: int = 500,
        max_gap_ms: int = 1000,
        clock_ms=_clock_ms,
        smoother: Any | None = None,
    ) -> None:
        for value, name in ((prediction_interval_ms, "prediction_interval_ms"), (stale_ms, "stale_ms"), (max_gap_ms, "max_gap_ms")):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if tuple(model.feature_names) != buffer.feature_names:
            raise ValueError("model and buffer feature order differ")
        if model.schema_version != buffer.schema_version:
            raise ValueError("model and buffer schema differ")
        if model.calibration_mode != calibration.mode:
            raise ValueError("model and calibration mode differ")
        self.pipeline = pipeline
        self.calibration = calibration
        self.temporal = temporal
        self.buffer = buffer
        self.model = model
        self.smoother = smoother
        self.prediction_interval_ms = int(prediction_interval_ms)
        self.stale_ms = int(stale_ms)
        self.max_gap_ms = int(max_gap_ms)
        self._clock_ms = clock_ms
        self._closed = False
        self._last_prediction_ms: int | None = None
        self._unreliable_since_ms: int | None = None
        self._last_feature_sample: FeatureSample | None = None
        self._last_temporal_sample: TemporalSample | None = None

    @property
    def last_feature_sample(self) -> FeatureSample | None:
        """Raw feature output for parity/evidence from the most recent packet."""
        return self._last_feature_sample

    @property
    def last_temporal_sample(self) -> TemporalSample | None:
        """Latest causal grid sample for read-only UI diagnostics."""
        return self._last_temporal_sample

    def process(self, packet: FramePacket, *, current_time_ms: int | None = None) -> DetectionResult:
        if self._closed:
            raise RuntimeError("DrowsinessDetector is closed")
        self._last_feature_sample = None
        if current_time_ms is None:
            now_ms = int(self._clock_ms())
        elif isinstance(current_time_ms, bool) or not isinstance(current_time_ms, Integral) or current_time_ms < 0:
            raise ValueError("current_time_ms must be a nonnegative integer")
        else:
            now_ms = int(current_time_ms)
        age_ms = now_ms - packet.timestamp_ms
        if age_ms < 0:
            raise ValueError("capture timestamp uses a different or future clock")
        if age_ms > self.stale_ms:
            return self._result(SystemStatus.UNRELIABLE, reason="stale frame", frame_age_ms=age_ms)

        sample = self.pipeline.process(packet)
        if not isinstance(sample, FeatureSample):
            raise TypeError("pipeline must return FeatureSample")
        self._last_feature_sample = sample
        current_valid = bool(
            sample.face_detected
            and sample.left_eye_valid
            and sample.right_eye_valid
            and sample.mouth_valid
            and sample.pose_valid
        )
        self._track_current_quality(sample.timestamp_ms, current_valid)

        if self.calibration.state is CalibrationState.IDLE:
            started = self.calibration.start(sample.timestamp_ms)
            if started.state is CalibrationState.COLLECTING:
                return self._result(SystemStatus.CALIBRATING, sample=sample, reason=started.reason)
        if self.calibration.state is CalibrationState.COLLECTING:
            calibration_snapshot = self.calibration.update(sample)
            if calibration_snapshot.state is CalibrationState.COMPLETE:
                self.temporal.reset()
                self.buffer.reset()
                self._last_prediction_ms = None
                return self._result(SystemStatus.WARMING_UP, sample=sample, reason="calibration complete")
            if calibration_snapshot.state is CalibrationState.FAILED:
                return self._result(SystemStatus.UNRELIABLE, sample=sample, reason=calibration_snapshot.reason)
            return self._result(SystemStatus.CALIBRATING, sample=sample, reason=calibration_snapshot.reason)
        if self.calibration.state is CalibrationState.FAILED or self.calibration.profile is None:
            return self._result(SystemStatus.UNRELIABLE, sample=sample, reason=self.calibration.snapshot.reason)

        temporal_samples = self.temporal.update(sample, self.calibration.profile)
        for temporal_sample in temporal_samples:
            self.buffer.append(temporal_sample)
            self._last_temporal_sample = temporal_sample

        if not sample.face_detected:
            return self._result(SystemStatus.NO_FACE, sample=sample, reason="no face")
        if not current_valid:
            return self._result(SystemStatus.UNRELIABLE, sample=sample, reason="current feature sample invalid")

        window = self.buffer.get_window()
        if window is None:
            return self._result(SystemStatus.WARMING_UP, sample=sample, reason="sequence window not ready")
        if self._last_prediction_ms is not None and window.end_timestamp_ms - self._last_prediction_ms < self.prediction_interval_ms:
            return self._result(SystemStatus.READY, sample=sample, reason="awaiting prediction cadence")

        try:
            raw = self._predict(window)
        # Phase 13 is an injected runtime boundary; inference failures become
        # an explicit ERROR result so the capture/UI loop remains controllable.
        except Exception as exc:  # noqa: BLE001
            return self._result(SystemStatus.ERROR, sample=sample, reason=f"model inference failed: {exc}")
        self._last_prediction_ms = raw.timestamp_ms
        smoothed = self.smoother.update(raw) if self.smoother is not None else None
        return DetectionResult(
            raw_prediction=raw,
            smoothed_prediction=smoothed,
            system_status=SystemStatus.READY,
            quality={
                "current_valid": True,
                "window": window.quality,
                "frame_age_ms": age_ms,
                **self._calibration_quality(),
            },
            calibration_status=self.calibration.state.value,
        )

    def _track_current_quality(self, timestamp_ms: int, current_valid: bool) -> None:
        if current_valid:
            self._unreliable_since_ms = None
            return
        if self._unreliable_since_ms is None:
            self._unreliable_since_ms = timestamp_ms
        elif timestamp_ms - self._unreliable_since_ms > self.max_gap_ms:
            self.temporal.reset()
            self.buffer.reset()
            if self.smoother is not None:
                self.smoother.reset()
            self._last_prediction_ms = None

    def _predict(self, window: SequenceWindow) -> Prediction:
        probabilities = np.asarray(self.model.predict_proba(window.x[np.newaxis, ...]), dtype=np.float32)
        if probabilities.shape == (1, 3):
            probabilities = probabilities[0]
        if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
            raise ValueError("model probabilities must be finite shape [3]")
        if np.any(probabilities < 0) or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
            raise ValueError("model probabilities must be nonnegative and sum to one")
        class_id = DriverState(int(np.argmax(probabilities)))
        return Prediction(
            timestamp_ms=window.end_timestamp_ms,
            probabilities=probabilities.copy(),
            class_id=class_id,
            valid=True,
            reason="",
            model_id=self.model.model_id,
        )

    def _result(
        self,
        status: SystemStatus,
        *,
        sample: FeatureSample | None = None,
        reason: str,
        frame_age_ms: int | None = None,
    ) -> DetectionResult:
        quality = {"reason": reason, **self._calibration_quality()}
        if sample is not None:
            quality.update(
                face_detected=sample.face_detected,
                left_eye_valid=sample.left_eye_valid,
                right_eye_valid=sample.right_eye_valid,
                mouth_valid=sample.mouth_valid,
                pose_valid=sample.pose_valid,
            )
        if frame_age_ms is not None:
            quality["frame_age_ms"] = frame_age_ms
        return DetectionResult(None, None, status, quality, self.calibration.state.value)

    def _calibration_quality(self) -> dict[str, Any]:
        snapshot = self.calibration.snapshot
        return {
            "calibration_progress": 1.0 if snapshot.state is CalibrationState.COMPLETE else snapshot.progress,
            "calibration_msg": snapshot.reason,
        }

    def reset_session(self) -> None:
        if self._closed:
            raise RuntimeError("DrowsinessDetector is closed")
        self.pipeline.reset()
        self.calibration.reset()
        self.temporal.reset()
        self.buffer.reset()
        if self.smoother is not None:
            self.smoother.reset()
        self._last_prediction_ms = None
        self._unreliable_since_ms = None
        self._last_feature_sample = None
        self._last_temporal_sample = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._last_feature_sample = None
            self._last_temporal_sample = None
            self.pipeline.close()
