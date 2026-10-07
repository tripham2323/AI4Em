"""Validated fixed-length feature window buffer."""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from typing import Protocol

import numpy as np

from src.contracts import SequenceWindow, TemporalSample


class FeatureTransformer(Protocol):
    def __call__(self, values: np.ndarray, validity: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _BufferedSample:
    timestamp_ms: int
    segment_id: int
    row: np.ndarray
    missing_required: bool


class PredictionBuffer:
    """Keep the latest causal timesteps and emit only accepted model windows."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        schema_version: str,
        sequence_steps: int,
        max_missing_ratio: Real,
        max_gap_ms: int,
        transformer: FeatureTransformer,
        required_validity_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        names = tuple(feature_names)
        if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty unique strings")
        if not isinstance(schema_version, str) or not schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if isinstance(sequence_steps, bool) or not isinstance(sequence_steps, Integral) or sequence_steps <= 0:
            raise ValueError("sequence_steps must be a positive integer")
        if isinstance(max_missing_ratio, bool) or not isinstance(max_missing_ratio, Real) or not 0 <= max_missing_ratio <= 1:
            raise ValueError("max_missing_ratio must be between zero and one")
        if isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, Integral) or max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be a positive integer")
        indices = tuple(required_validity_indices)
        if not indices or any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in indices):
            raise ValueError("required_validity_indices must contain nonnegative integers")
        if not callable(transformer):
            raise TypeError("transformer must be callable")
        self.feature_names = names
        self.schema_version = schema_version
        self.sequence_steps = int(sequence_steps)
        self.max_missing_ratio = float(max_missing_ratio)
        self.max_gap_ms = int(max_gap_ms)
        self.required_validity_indices = tuple(int(index) for index in indices)
        self._transformer = transformer
        self._samples: deque[_BufferedSample] = deque(maxlen=self.sequence_steps)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def ready(self) -> bool:
        return self.get_window() is not None

    def append(self, sample: TemporalSample) -> None:
        if not isinstance(sample, TemporalSample):
            raise TypeError("sample must be TemporalSample")
        if isinstance(sample.timestamp_ms, bool) or not isinstance(sample.timestamp_ms, Integral) or sample.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        if isinstance(sample.segment_id, bool) or not isinstance(sample.segment_id, Integral) or sample.segment_id < 0:
            raise ValueError("segment_id must be a nonnegative integer")
        values = np.asarray(sample.values)
        validity = np.asarray(sample.validity)
        if values.shape != (len(self.feature_names),) or not np.issubdtype(values.dtype, np.floating):
            raise ValueError("temporal values do not match ordered feature_names")
        if validity.ndim != 1 or not np.issubdtype(validity.dtype, np.bool_):
            raise ValueError("validity must be a one-dimensional boolean array")
        if max(self.required_validity_indices) >= validity.size:
            raise ValueError("validity does not contain every required channel")

        timestamp_ms = int(sample.timestamp_ms)
        segment_id = int(sample.segment_id)
        if self._samples:
            previous = self._samples[-1]
            if timestamp_ms <= previous.timestamp_ms:
                raise ValueError("temporal timestamps must strictly increase")
            if segment_id != previous.segment_id or timestamp_ms - previous.timestamp_ms > self.max_gap_ms:
                self.reset()
        row = np.asarray(self._transformer(values.copy(), validity.copy()), dtype=np.float32)
        if row.shape != (len(self.feature_names),) or not np.isfinite(row).all():
            raise ValueError("transformer must return one finite float per feature")
        missing = not bool(np.all(validity[list(self.required_validity_indices)]))
        self._samples.append(_BufferedSample(timestamp_ms, segment_id, row.copy(), missing))

    def get_window(self) -> SequenceWindow | None:
        if len(self._samples) != self.sequence_steps:
            return None
        samples = tuple(self._samples)
        if samples[-1].missing_required:
            return None
        missing_count = sum(item.missing_required for item in samples)
        missing_ratio = missing_count / self.sequence_steps
        if missing_ratio > self.max_missing_ratio:
            return None
        timestamps = [item.timestamp_ms for item in samples]
        if any(right - left > self.max_gap_ms for left, right in pairwise(timestamps)):
            return None
        x = np.stack([item.row for item in samples]).astype(np.float32, copy=False)
        return SequenceWindow(
            x=x,
            start_timestamp_ms=timestamps[0],
            end_timestamp_ms=timestamps[-1],
            quality={
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "segment_id": samples[-1].segment_id,
            },
            feature_names=self.feature_names,
            schema_version=self.schema_version,
        )

    def reset(self) -> None:
        self._samples.clear()
