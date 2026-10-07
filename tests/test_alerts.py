from __future__ import annotations

import numpy as np
import pytest

from src.alerts.manager import AlertManager
from src.contracts import (
    DetectionResult,
    DriverState,
    Prediction,
    SystemStatus,
)


def detection(timestamp_ms: int, probabilities=None, *, status=SystemStatus.READY):
    prediction = None
    if probabilities is not None:
        values = np.asarray(probabilities, dtype=np.float32)
        prediction = Prediction(
            timestamp_ms=timestamp_ms,
            probabilities=values,
            class_id=DriverState(int(values.argmax())),
            valid=True,
            reason="",
            model_id="test",
        )
    return DetectionResult(
        raw_prediction=prediction,
        smoothed_prediction=prediction,
        system_status=status,
        quality={},
        calibration_status="complete",
    )


def update(manager, timestamp_ms, probabilities=None, *, status=SystemStatus.READY):
    # Model the documented one-second prediction cadence while the assertions
    # focus on a boundary between observations. Explicit unknowns are never
    # filled in by this helper.
    last = manager._last_timestamp_ms
    if last is not None and (probabilities is not None or status != SystemStatus.READY):
        while timestamp_ms - last > 1_500:
            last += 1_000
            manager.update(detection(last, probabilities, status=status), last)
    return manager.update(detection(timestamp_ms, probabilities, status=status), timestamp_ms)


def test_low_warning_requires_three_seconds_and_has_exact_boundary():
    manager = AlertManager()
    risk = (0.30, 0.40, 0.30)

    assert update(manager, 0, risk).level == 0
    assert update(manager, 2_999, risk).level == 0
    decision = update(manager, 3_000, risk)
    assert decision.level == 1
    assert decision.audio_command is None
    assert "giảm tỉnh táo" in decision.message
    assert update(manager, 3_001, risk).level == 1


def test_drowsy_preempts_low_and_enters_after_two_seconds():
    manager = AlertManager()
    drowsy = (0.10, 0.20, 0.70)

    assert update(manager, 0, drowsy).level == 0
    assert update(manager, 1_999, drowsy).level == 0
    decision = update(manager, 2_000, drowsy)
    assert decision.level == 2
    assert decision.audio_command == "drowsy"
    assert update(manager, 2_001, drowsy).level == 2


def test_one_high_prediction_does_not_alarm_and_oscillation_restarts_dwell():
    manager = AlertManager()
    high = (0.10, 0.50, 0.40)
    below = (0.41, 0.30, 0.29)

    assert update(manager, 0, high).level == 0
    assert update(manager, 2_000, below).level == 0
    assert update(manager, 3_000, high).level == 0
    assert update(manager, 5_999, high).level == 0
    assert update(manager, 6_000, high).level == 1


def test_unknown_breaks_pending_entry_and_emits_no_ai_audio():
    manager = AlertManager()
    risk = (0.20, 0.40, 0.40)
    drowsy = (0.10, 0.20, 0.70)

    update(manager, 0, risk)
    unknown = update(manager, 1_000, None, status=SystemStatus.NO_FACE)
    assert unknown.level == 0
    assert unknown.audio_command is None
    assert update(manager, 2_999, risk).level == 0
    assert update(manager, 4_998, risk).level == 0
    assert update(manager, 4_999, risk).level == 0
    assert update(manager, 5_000, risk).level == 1

    manager.reset()
    update(manager, 0, drowsy)
    assert update(manager, 2_000, None, status=SystemStatus.UNRELIABLE).audio_command is None
    assert update(manager, 2_001, drowsy).level == 0


def test_stale_prediction_is_unknown_and_does_not_extend_a_pending_timer():
    manager = AlertManager()
    risk = (0.20, 0.40, 0.40)

    update(manager, 0, risk)
    stale = detection(0, risk)
    result = manager.update(stale, 600)
    assert result.level == 0
    assert result.audio_command is None
    assert update(manager, 3_000, risk).level == 0


def test_two_second_gap_cannot_be_counted_as_continuous_valid_dwell():
    manager = AlertManager()
    risk = (0.20, 0.40, 0.40)

    manager.update(detection(0, risk), 0)
    manager.update(detection(2_000, risk), 2_000)
    assert update(manager, 4_999, risk).level == 0
    assert update(manager, 5_000, risk).level == 1


def test_strong_escalates_at_ten_seconds_even_inside_regular_audio_cooldown():
    manager = AlertManager()
    drowsy = (0.10, 0.20, 0.70)

    update(manager, 0, drowsy)
    entered = update(manager, 2_000, drowsy)
    assert entered.level == 2
    assert entered.audio_command == "drowsy"
    before = update(manager, 9_999, drowsy)
    assert before.level == 2 and not before.strong
    strong = update(manager, 10_000, drowsy)
    assert strong.level == 2 and strong.strong
    assert strong.audio_command == "strong"
    after = update(manager, 10_001, drowsy)
    assert after.level == 2 and after.strong
    assert after.audio_command is None


def test_drowsy_exits_only_after_five_seconds_and_can_fall_back_to_level_one():
    manager = AlertManager()
    drowsy = (0.10, 0.20, 0.70)
    risk_without_drowsy = (0.50, 0.30, 0.20)

    update(manager, 0, drowsy)
    update(manager, 2_000, drowsy)
    assert update(manager, 3_000, risk_without_drowsy).level == 2
    assert update(manager, 7_999, risk_without_drowsy).level == 2
    result = update(manager, 8_000, risk_without_drowsy)
    assert result.level == 1
    assert not result.strong
    assert update(manager, 8_001, risk_without_drowsy).level == 1


def test_level_one_exits_after_five_seconds_below_risk_exit():
    manager = AlertManager()
    risk = (0.20, 0.40, 0.40)
    low = (0.70, 0.20, 0.10)

    update(manager, 0, risk)
    update(manager, 3_000, risk)
    assert update(manager, 4_000, low).level == 1
    assert update(manager, 8_999, low).level == 1
    assert update(manager, 9_000, low).level == 0
    assert update(manager, 9_001, low).level == 0


def test_audio_same_level_repeats_no_more_than_every_fifteen_seconds():
    manager = AlertManager({"strong_duration_s": 60})
    drowsy = (0.10, 0.20, 0.70)
    update(manager, 0, drowsy)
    assert update(manager, 2_000, drowsy).audio_command == "drowsy"
    assert update(manager, 16_999, drowsy).audio_command is None
    assert update(manager, 17_000, drowsy).audio_command == "drowsy"
    assert update(manager, 17_001, drowsy).audio_command is None


def test_muting_suppresses_only_audio_and_preserves_warning_state():
    manager = AlertManager()
    risk = (0.20, 0.40, 0.40)
    drowsy = (0.10, 0.20, 0.70)

    update(manager, 0, risk)
    low = update(manager, 3_000, risk)
    assert low.level == 1
    candidate = detection(4_000, drowsy)
    before_smoothed = candidate.smoothed_prediction.probabilities.copy()
    manager.set_muted(True)
    manager.update(candidate, 4_000)
    muted = update(manager, 6_000, drowsy)
    assert muted.level == 2
    assert muted.audio_command is None
    np.testing.assert_array_equal(candidate.smoothed_prediction.probabilities, before_smoothed)
    manager.set_muted(False)
    strong = update(manager, 14_000, drowsy)
    assert strong.level == 2 and strong.strong
    assert strong.audio_command == "strong"


def test_no_face_message_is_technical_after_two_seconds_without_clearing_warning():
    manager = AlertManager()
    drowsy = (0.10, 0.20, 0.70)

    update(manager, 0, drowsy)
    update(manager, 2_000, drowsy)
    before = update(manager, 4_999, None, status=SystemStatus.NO_FACE)
    assert "khuôn mặt" not in before.message.lower()
    result = update(manager, 5_000, None, status=SystemStatus.NO_FACE)
    assert result.level == 2
    assert result.audio_command is None
    assert result.message == "Không quan sát được khuôn mặt"
    assert update(manager, 5_001, None, status=SystemStatus.NO_FACE).message == result.message


def test_strong_repeat_is_at_least_fifteen_seconds_after_escalation():
    manager = AlertManager()
    drowsy = (0.10, 0.20, 0.70)
    update(manager, 0, drowsy)
    update(manager, 10_000, drowsy)
    assert update(manager, 24_999, drowsy).audio_command is None
    assert update(manager, 25_000, drowsy).audio_command == "strong"
    assert update(manager, 25_001, drowsy).audio_command is None


def test_invalid_probabilities_are_rejected_and_reset_clears_episode_state():
    manager = AlertManager()
    malformed = detection(0, (0.2, 0.2, 0.2))
    with pytest.raises(ValueError):
        manager.update(malformed, 0)

    risk = (0.20, 0.40, 0.40)
    update(manager, 0, risk)
    update(manager, 3_000, risk)
    manager.reset()
    assert update(manager, 4_000, risk).level == 0
