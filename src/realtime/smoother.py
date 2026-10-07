"""Timestamp-aware probability smoothing."""
from __future__ import annotations

import math
from collections import deque
from numbers import Integral

import numpy as np

from src.contracts import DriverState, Prediction


class PredictionSmoother:
    """Average the latest valid probability vectors without hiding gaps."""

    def __init__(self, *, samples: int = 3, expiry_ms: int = 2000) -> None:
        if isinstance(samples, bool) or not isinstance(samples, Integral) or samples <= 0:
            raise ValueError("samples must be a positive integer")
        if isinstance(expiry_ms, bool) or not isinstance(expiry_ms, Integral) or expiry_ms <= 0:
            raise ValueError("expiry_ms must be a positive integer")
        self.samples = int(samples)
        self.expiry_ms = int(expiry_ms)
        self._predictions: deque[Prediction] = deque(maxlen=self.samples)
        self._model_id: str | None = None

    def __len__(self) -> int:
        return len(self._predictions)

    def update(self, prediction: Prediction) -> Prediction | None:
        self._validate(prediction)
        if self._predictions:
            previous = self._predictions[-1]
            if prediction.timestamp_ms <= previous.timestamp_ms:
                raise ValueError("prediction timestamps must strictly increase")
            if prediction.timestamp_ms - previous.timestamp_ms >= self.expiry_ms:
                self.reset()
        if self._model_id is not None and prediction.model_id != self._model_id:
            raise ValueError("cannot smooth predictions from different models")
        self._model_id = prediction.model_id
        stored = Prediction(
            timestamp_ms=int(prediction.timestamp_ms),
            probabilities=np.asarray(prediction.probabilities, dtype=np.float32).copy(),
            class_id=prediction.class_id,
            valid=True,
            reason=prediction.reason,
            model_id=prediction.model_id,
        )
        self._predictions.append(stored)
        if len(self._predictions) < self.samples:
            return None

        probabilities = np.mean(
            np.stack([item.probabilities for item in self._predictions]),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        probabilities /= probabilities.sum(dtype=np.float32)
        return Prediction(
            timestamp_ms=prediction.timestamp_ms,
            probabilities=probabilities,
            class_id=DriverState(int(np.argmax(probabilities))),
            valid=True,
            reason=f"mean of {self.samples} valid predictions",
            model_id=prediction.model_id,
        )

    def reset(self) -> None:
        self._predictions.clear()
        self._model_id = None

    @staticmethod
    def _validate(prediction: Prediction) -> None:
        if not isinstance(prediction, Prediction):
            raise TypeError("prediction must be Prediction")
        if not prediction.valid:
            raise ValueError("only valid predictions can be smoothed")
        if isinstance(prediction.timestamp_ms, bool) or not isinstance(prediction.timestamp_ms, Integral) or prediction.timestamp_ms < 0:
            raise ValueError("prediction timestamp must be a nonnegative integer")
        probabilities = np.asarray(prediction.probabilities)
        if probabilities.shape != (3,) or not np.issubdtype(probabilities.dtype, np.floating):
            raise ValueError("probabilities must be a floating array of shape [3]")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("probabilities must be finite and nonnegative")
        if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
            raise ValueError("probabilities must sum to one")
        expected_class = DriverState(int(np.argmax(probabilities)))
        if prediction.class_id is not expected_class:
            raise ValueError("class_id must match probability argmax")
        if not isinstance(prediction.model_id, str) or not prediction.model_id:
            raise ValueError("model_id must be a non-empty string")
