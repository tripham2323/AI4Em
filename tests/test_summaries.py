import numpy as np
import pytest

from src.contracts import TemporalSample


def samples():
    rows = []
    for t in range(100, 10100, 100):
        values = np.ones(16, dtype=np.float32)
        values[0] = t / 1000.0
        receipt = {'start_ms': t - 100, 'end_ms': t, 'eye_valid': True,
                   'mouth_valid': True, 'pose_valid': True, 'closed_proxy': t <= 2000,
                   'eye_closed': t <= 2000, 'mouth_open': False}
        event = [{'kind': 'closure', 'start_ms': 0, 'end_ms': 2000,
                  'duration_s': 2.0, 'censored': False, 'candidate': False}] if t == 2000 else []
        rows.append(TemporalSample(t, values, np.ones(16, dtype=np.bool_), 0,
                                   {'held_intervals': [receipt], 'events': event}))
    return rows


def test_summary_uses_seconds_for_slope_and_actual_valid_duration():
    from src.datasets.summaries import summarize_window
    result = summarize_window(samples(), start_ms=0, end_ms=10000)
    assert result['ear_left_norm_slope'] == pytest.approx(1.0)
    assert result['eye_valid_duration_s'] == pytest.approx(10.0)
    assert result['closure_duration_s'] == pytest.approx(2.0)
    assert result['blink_count'] == 0
    assert result['eye_valid_ratio'] == pytest.approx(1.0)


def test_summary_excludes_invalid_measurements_and_clips_intervals():
    from src.datasets.summaries import summarize_window
    rows = samples()
    rows[0].values[0] = 1000.0
    rows[0].validity[0] = False
    result = summarize_window(rows, start_ms=500, end_ms=10000)
    assert result['ear_left_norm_slope'] == pytest.approx(1.0)
    assert result['eye_valid_duration_s'] == pytest.approx(9.5)
    assert result['closure_duration_s'] == pytest.approx(1.5)


def test_unknown_intervals_do_not_count_as_open_or_closed():
    from src.datasets.summaries import summarize_window
    rows = samples()
    for row in rows[:20]:
        row.event_summaries['held_intervals'][0]['eye_valid'] = False
    result = summarize_window(rows, start_ms=0, end_ms=10000)
    assert result['eye_valid_duration_s'] == pytest.approx(8.0)
    assert result['closure_duration_s'] == 0.0
    assert result['eye_valid_ratio'] == pytest.approx(0.8)


def test_missing_duration_is_union_of_invalid_channels():
    from src.datasets.summaries import summarize_window
    rows = samples()
    for row in rows[:20]:
        row.event_summaries['held_intervals'][0]['eye_valid'] = False
    for row in rows[-20:]:
        row.event_summaries['held_intervals'][0]['mouth_valid'] = False
    result = summarize_window(rows, start_ms=0, end_ms=10000)
    assert result['missing_duration_ratio'] == pytest.approx(0.4)


def test_nonintersecting_completed_closure_does_not_dilute_mean():
    from src.datasets.summaries import summarize_window
    rows = samples()
    rows[0].event_summaries['events'] = [
        {'kind': 'closure', 'start_ms': -500, 'end_ms': 0,
         'duration_s': 0.5, 'censored': False, 'candidate': True}]
    result = summarize_window(rows, start_ms=0, end_ms=10000)
    assert result['closure_mean_duration_s'] == pytest.approx(2.0)
