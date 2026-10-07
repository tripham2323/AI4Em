"""Non-blocking warning state machine; audio is returned as a command."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from threading import Lock
from typing import Any

import numpy as np

from src.contracts import AlertDecision, DetectionResult, SystemStatus


_DEFAULTS: dict[str, Any] = {
    "prediction_interval_s": 1.0,
    "stale_ms": 500,
    "smoothing_expiry_s": 2.0,
    "low_enter": 0.60,
    "low_exit": 0.40,
    "low_enter_duration_s": 3.0,
    "low_exit_duration_s": 5.0,
    "drowsy_enter": 0.65,
    "drowsy_exit": 0.40,
    "drowsy_enter_duration_s": 2.0,
    "drowsy_exit_duration_s": 5.0,
    "strong_duration_s": 10.0,
    "audio_cooldown_s": 15.0,
    "muted": False,
}


class AlertManager:
    """Convert smoothed, current predictions into UI warning/audio decisions.

    ``timestamp_ms`` must use the same monotonic clock epoch as prediction
    timestamps. This class never plays audio or waits; consumers execute the
    returned ``audio_command`` on their owning UI thread.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(_DEFAULTS)
        if config is not None:
            if not isinstance(config, Mapping):
                raise TypeError("config must be a mapping")
            values.update({key: value for key, value in config.items() if key in values})
        self._validate(values)
        self._thresholds = values
        self._stale_ms = int(values["stale_ms"])
        # The smoother expires after two seconds without a valid prediction;
        # warning dwell cannot bridge a gap that invalidates that history.
        self._max_valid_gap_ms = round(values["smoothing_expiry_s"] * 1000)
        self._muted = values["muted"]
        self._mute_lock = Lock()
        self.reset()

    @staticmethod
    def _validate(values: Mapping[str, Any]) -> None:
        for key in (
            "prediction_interval_s", "smoothing_expiry_s", "low_enter_duration_s", "low_exit_duration_s",
            "drowsy_enter_duration_s", "drowsy_exit_duration_s", "strong_duration_s",
            "audio_cooldown_s",
        ):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive number")
        stale = values["stale_ms"]
        if isinstance(stale, bool) or not isinstance(stale, Integral) or stale <= 0:
            raise ValueError("stale_ms must be a positive integer")
        for key in ("low_enter", "low_exit", "drowsy_enter", "drowsy_exit"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite probability in [0, 1]")
        if values["low_exit"] >= values["low_enter"]:
            raise ValueError("low_exit must be below low_enter")
        if values["drowsy_exit"] >= values["drowsy_enter"]:
            raise ValueError("drowsy_exit must be below drowsy_enter")
        if not isinstance(values["muted"], bool):
            raise ValueError("muted must be a boolean")

    def set_muted(self, muted: bool) -> None:
        if not isinstance(muted, bool):
            raise TypeError("muted must be a bool")
        with self._mute_lock:
            self._muted = muted

    def reset(self) -> None:
        """Clear warning/timer/cooldown history for a new inference session."""
        self._level = 0
        self._strong = False
        self._last_timestamp_ms: int | None = None
        self._last_valid_timestamp_ms: int | None = None
        self._risk_enter_since: int | None = None
        self._risk_exit_since: int | None = None
        self._drowsy_enter_since: int | None = None
        self._drowsy_exit_since: int | None = None
        self._noface_since: int | None = None
        self._last_audio_ms: dict[str, int] = {}

    @staticmethod
    def _prediction(result: DetectionResult, timestamp_ms: int, stale_ms: int):
        if result.system_status != SystemStatus.READY:
            return None
        prediction = result.smoothed_prediction
        if prediction is None or not prediction.valid:
            return None
        prediction_time = prediction.timestamp_ms
        if (isinstance(prediction_time, bool) or not isinstance(prediction_time, Integral)
                or prediction_time < 0):
            raise ValueError("prediction timestamp must be a nonnegative integer")
        age_ms = timestamp_ms - int(prediction_time)
        if age_ms < 0:
            raise ValueError("prediction timestamp cannot be in the future")
        if age_ms > stale_ms:
            return None
        probabilities = np.asarray(prediction.probabilities)
        if probabilities.shape != (3,) or not np.issubdtype(probabilities.dtype, np.number):
            raise ValueError("prediction probabilities must have shape (3,)")
        probabilities = probabilities.astype(np.float64, copy=False)
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)):
            raise ValueError("prediction probabilities must be finite and sum to one")
        return probabilities

    def _clear_pending(self) -> None:
        self._risk_enter_since = None
        self._risk_exit_since = None
        self._drowsy_enter_since = None
        self._drowsy_exit_since = None

    def _message(self) -> str:
        if self._strong:
            return "Phát hiện dấu hiệu buồn ngủ. Hãy dừng xe và nghỉ ngơi."
        if self._level == 2:
            return "Phát hiện dấu hiệu buồn ngủ."
        if self._level == 1:
            return "Bạn đang có dấu hiệu giảm tỉnh táo."
        return ""

    def _audio(self, command: str, timestamp_ms: int, *, bypass_cooldown: bool = False) -> str | None:
        with self._mute_lock:
            if self._muted:
                return None
            cooldown_ms = round(self._thresholds["audio_cooldown_s"] * 1000)
            previous = self._last_audio_ms.get(command)
            if not bypass_cooldown and previous is not None and timestamp_ms - previous < cooldown_ms:
                return None
            self._last_audio_ms[command] = timestamp_ms
            return command

    def _decision(self, timestamp_ms: int, audio_command: str | None = None,
                  message: str | None = None) -> AlertDecision:
        return AlertDecision(
            level=self._level,
            strong=self._strong,
            audio_command=audio_command,
            message=self._message() if message is None else message,
            timestamp_ms=timestamp_ms,
        )

    def update(self, result: DetectionResult, timestamp_ms: int) -> AlertDecision:
        if not isinstance(result, DetectionResult):
            raise TypeError("result must be a DetectionResult")
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, Integral) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        timestamp_ms = int(timestamp_ms)
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must strictly increase")
        probabilities = self._prediction(result, timestamp_ms, self._stale_ms)
        self._last_timestamp_ms = timestamp_ms
        if probabilities is None:
            self._clear_pending()
            self._last_valid_timestamp_ms = None
            if result.system_status == SystemStatus.NO_FACE:
                if self._noface_since is None:
                    self._noface_since = timestamp_ms
                if timestamp_ms - self._noface_since >= 2_000:
                    return self._decision(timestamp_ms, message="Không quan sát được khuôn mặt")
            else:
                self._noface_since = None
            return self._decision(timestamp_ms)

        self._noface_since = None
        if (self._last_valid_timestamp_ms is not None
                and timestamp_ms - self._last_valid_timestamp_ms >= self._max_valid_gap_ms):
            self._clear_pending()
        self._last_valid_timestamp_ms = timestamp_ms

        p_risk = float(probabilities[1] + probabilities[2])
        p_drowsy = float(probabilities[2])
        audio_command = None
        t = self._thresholds

        # Track entry/strong dwell from the beginning of the qualifying
        # probability run, even while a Level 1 UI warning is active.
        if p_drowsy >= t["drowsy_enter"]:
            if self._drowsy_enter_since is None:
                self._drowsy_enter_since = timestamp_ms
        else:
            self._drowsy_enter_since = None

        if self._level == 2:
            if p_drowsy < t["drowsy_exit"]:
                if self._drowsy_exit_since is None:
                    self._drowsy_exit_since = timestamp_ms
                elif timestamp_ms - self._drowsy_exit_since >= round(t["drowsy_exit_duration_s"] * 1000):
                    self._level = 1 if p_risk >= t["low_exit"] else 0
                    self._strong = False
                    self._drowsy_exit_since = None
            else:
                self._drowsy_exit_since = None
        elif self._level == 1:
            if p_risk < t["low_exit"]:
                if self._risk_exit_since is None:
                    self._risk_exit_since = timestamp_ms
                elif timestamp_ms - self._risk_exit_since >= round(t["low_exit_duration_s"] * 1000):
                    self._level = 0
                    self._risk_exit_since = None
            else:
                self._risk_exit_since = None

            if self._drowsy_enter_since is not None and (
                timestamp_ms - self._drowsy_enter_since >= round(t["drowsy_enter_duration_s"] * 1000)
            ):
                self._level = 2
                self._risk_exit_since = None
                audio_command = self._audio("drowsy", timestamp_ms)
        else:
            self._risk_exit_since = None
            if p_risk >= t["low_enter"]:
                if self._risk_enter_since is None:
                    self._risk_enter_since = timestamp_ms
            else:
                self._risk_enter_since = None

            if self._drowsy_enter_since is not None and (
                timestamp_ms - self._drowsy_enter_since >= round(t["drowsy_enter_duration_s"] * 1000)
            ):
                self._level = 2
                self._risk_enter_since = None
                audio_command = self._audio("drowsy", timestamp_ms)
            elif (self._risk_enter_since is not None and
                  timestamp_ms - self._risk_enter_since >= round(t["low_enter_duration_s"] * 1000)):
                self._level = 1
                self._risk_enter_since = None

        if self._level == 2:
            if (not self._strong and self._drowsy_enter_since is not None
                    and timestamp_ms - self._drowsy_enter_since >= round(t["strong_duration_s"] * 1000)):
                self._strong = True
                audio_command = self._audio("strong", timestamp_ms, bypass_cooldown=True)
            elif self._strong:
                if audio_command is None:
                    audio_command = self._audio("strong", timestamp_ms)
            elif audio_command is None:
                audio_command = self._audio("drowsy", timestamp_ms)

        return self._decision(timestamp_ms, audio_command)
