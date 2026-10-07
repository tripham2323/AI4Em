"""Chronological source-time replay and parity comparison."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.contracts import DetectionResult, FeatureSample, FramePacket, Prediction

FEATURE_VALUE_FIELDS = (
    "ear_left",
    "ear_right",
    "ear_mean",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "reprojection_error_norm",
)
FEATURE_VALIDITY_FIELDS = (
    "face_detected",
    "left_eye_valid",
    "right_eye_valid",
    "mouth_valid",
    "pose_valid",
)


class ReplayDetector(Protocol):
    @property
    def last_feature_sample(self) -> FeatureSample | None: ...
    def reset_session(self) -> None: ...
    def process(self, packet: FramePacket, *, current_time_ms: int | None = None) -> DetectionResult: ...


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    source_id: str
    frame_index: int
    timestamp_ms: int
    system_status: str
    calibration_status: str
    feature_values: tuple[float | None, ...] | None
    feature_validity: tuple[bool, ...] | None
    raw_timestamp_ms: int | None
    raw_model_id: str | None
    raw_probabilities: tuple[float, float, float] | None
    raw_class_id: int | None
    smoothed_timestamp_ms: int | None
    smoothed_model_id: str | None
    smoothed_probabilities: tuple[float, float, float] | None
    smoothed_class_id: int | None


def _feature_trace(
    sample: FeatureSample | None,
    *,
    packet: FramePacket,
) -> tuple[tuple[float | None, ...] | None, tuple[bool, ...] | None]:
    if sample is None:
        return None, None
    if not isinstance(sample, FeatureSample):
        raise TypeError("detector.last_feature_sample must be FeatureSample or None")
    if (
        sample.source_id != packet.source_id
        or sample.frame_index != packet.frame_index
        or sample.timestamp_ms != packet.timestamp_ms
    ):
        raise ValueError("replay feature trace does not match the current packet identity")
    values = tuple(
        float(value) if math.isfinite(value) else None
        for value in (getattr(sample, field) for field in FEATURE_VALUE_FIELDS)
    )
    validity = tuple(bool(getattr(sample, field)) for field in FEATURE_VALIDITY_FIELDS)
    return values, validity


def _probabilities(
    prediction: Prediction | None, *, packet_timestamp_ms: int
) -> tuple[float, float, float] | None:
    if prediction is None:
        return None
    if prediction.timestamp_ms < 0 or prediction.timestamp_ms > packet_timestamp_ms:
        raise ValueError("replay received prediction outside causal source time")
    values = np.asarray(prediction.probabilities)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("replay received malformed prediction probabilities")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-5):
        raise ValueError("replay prediction probabilities must sum to one")
    return tuple(float(value) for value in values)


def replay_session(
    packets: Iterable[FramePacket],
    detector: ReplayDetector,
    *,
    reset_session: bool = True,
) -> tuple[ReplayRecord, ...]:
    """Replay every packet against its source timestamp, never wall-clock speed."""
    if reset_session:
        detector.reset_session()
    records: list[ReplayRecord] = []
    source_id: str | None = None
    previous_timestamp: int | None = None
    previous_frame_index: int | None = None
    for packet in packets:
        if not isinstance(packet, FramePacket):
            raise TypeError("packets must contain FramePacket values")
        if source_id is None:
            source_id = packet.source_id
        elif packet.source_id != source_id:
            raise ValueError("one replay_session cannot join different sources")
        if previous_timestamp is not None and packet.timestamp_ms <= previous_timestamp:
            raise ValueError("replay packet timestamps must strictly increase")
        if previous_frame_index is not None and packet.frame_index <= previous_frame_index:
            raise ValueError("replay frame indices must strictly increase")
        detection = detector.process(packet, current_time_ms=packet.timestamp_ms)
        feature_values, feature_validity = _feature_trace(
            detector.last_feature_sample,
            packet=packet,
        )
        raw_class = None if detection.raw_prediction is None or detection.raw_prediction.class_id is None else int(detection.raw_prediction.class_id)
        smooth_class = None if detection.smoothed_prediction is None or detection.smoothed_prediction.class_id is None else int(detection.smoothed_prediction.class_id)
        records.append(ReplayRecord(
            source_id=packet.source_id,
            frame_index=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            system_status=detection.system_status.value,
            calibration_status=detection.calibration_status,
            feature_values=feature_values,
            feature_validity=feature_validity,
            raw_timestamp_ms=None if detection.raw_prediction is None else detection.raw_prediction.timestamp_ms,
            raw_model_id=None if detection.raw_prediction is None else detection.raw_prediction.model_id,
            raw_probabilities=_probabilities(detection.raw_prediction, packet_timestamp_ms=packet.timestamp_ms),
            raw_class_id=raw_class,
            smoothed_timestamp_ms=None if detection.smoothed_prediction is None else detection.smoothed_prediction.timestamp_ms,
            smoothed_model_id=None if detection.smoothed_prediction is None else detection.smoothed_prediction.model_id,
            smoothed_probabilities=_probabilities(detection.smoothed_prediction, packet_timestamp_ms=packet.timestamp_ms),
            smoothed_class_id=smooth_class,
        ))
        previous_timestamp = packet.timestamp_ms
        previous_frame_index = packet.frame_index
    return tuple(records)


def compare_replays(
    expected: Sequence[ReplayRecord],
    actual: Sequence[ReplayRecord],
    *,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Compare all scheduled timestamps and report, rather than drop, mismatches."""
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite nonnegative number")
    mismatches: list[dict[str, object]] = []
    if len(expected) != len(actual):
        mismatches.append({"field": "record_count", "expected": len(expected), "actual": len(actual)})
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected):
            mismatches.append({"index": index, "field": "unexpected_record", "actual": actual[index].timestamp_ms})
            continue
        if index >= len(actual):
            mismatches.append({"index": index, "field": "missing_record", "expected": expected[index].timestamp_ms})
            continue
        left, right = expected[index], actual[index]
        for field in (
            "source_id", "frame_index", "timestamp_ms", "system_status", "calibration_status",
            "feature_validity", "raw_timestamp_ms", "raw_model_id", "raw_class_id",
            "smoothed_timestamp_ms", "smoothed_model_id", "smoothed_class_id",
        ):
            if getattr(left, field) != getattr(right, field):
                mismatches.append({
                    "index": index,
                    "timestamp_ms": left.timestamp_ms,
                    "field": field,
                    "expected": getattr(left, field),
                    "actual": getattr(right, field),
                })
        for field in ("feature_values", "raw_probabilities", "smoothed_probabilities"):
            expected_values = getattr(left, field)
            actual_values = getattr(right, field)
            presence_differs = (expected_values is None) != (actual_values is None)
            values_differ = (
                expected_values is not None
                and actual_values is not None
                and (
                    len(expected_values) != len(actual_values)
                    or any(
                        (expected_value is None) != (actual_value is None)
                        or (
                            expected_value is not None
                            and actual_value is not None
                            and not math.isclose(
                                expected_value,
                                actual_value,
                                rel_tol=0,
                                abs_tol=atol,
                            )
                        )
                        for expected_value, actual_value in zip(expected_values, actual_values, strict=False)
                    )
                )
            )
            if presence_differs or values_differ:
                mismatches.append({
                    "index": index,
                    "timestamp_ms": left.timestamp_ms,
                    "field": field,
                    "expected": expected_values,
                    "actual": actual_values,
                })
    return {
        "match": not mismatches,
        "expected_records": len(expected),
        "actual_records": len(actual),
        "atol": float(atol),
        "mismatches": mismatches,
    }
