"""Valid-only window summaries from the shared causal temporal receipts."""
from __future__ import annotations

import math
import numpy as np

from src.contracts import TemporalSample

SIGNALS = ((0, 'ear_left_norm'), (1, 'ear_right_norm'), (2, 'mar_delta'),
           (3, 'pitch_delta'), (4, 'yaw_delta'), (5, 'roll_delta'),
           (9, 'pitch_velocity_dps'))


def summarize_window(samples: list[TemporalSample], *, start_ms: int, end_ms: int) -> dict[str, float]:
    if end_ms <= start_ms:
        raise ValueError('Summary interval must have positive duration')
    rows = [s for s in samples if start_ms <= s.timestamp_ms <= end_ms]
    result = {}
    for column, name in SIGNALS:
        usable = [s for s in rows if s.validity[column] and math.isfinite(float(s.values[column]))]
        y = np.asarray([s.values[column] for s in usable], dtype=np.float64)
        x = np.asarray([(s.timestamp_ms - start_ms) / 1000 for s in usable], dtype=np.float64)
        for stat in ('mean', 'std', 'min', 'max', 'slope'):
            result[f'{name}_{stat}'] = math.nan
        if len(y):
            result.update({f'{name}_mean': float(y.mean()), f'{name}_std': float(y.std()),
                           f'{name}_min': float(y.min()), f'{name}_max': float(y.max())})
        if len(y) > 1:
            centered = x - x.mean()
            denominator = float(centered @ centered)
            if denominator:
                result[f'{name}_slope'] = float(centered @ (y - y.mean()) / denominator)
    span = (end_ms - start_ms) / 1000
    durations = {'eye': 0.0, 'mouth': 0.0, 'pose': 0.0}
    joint_valid_duration = 0.0
    closed_duration = 0.0
    mouth_open_duration = 0.0
    seen_intervals = set()
    events = {}
    for sample in rows:
        for interval in sample.event_summaries.get('held_intervals', []):
            key = (interval['start_ms'], interval['end_ms'])
            if key in seen_intervals:
                continue
            seen_intervals.add(key)
            dt = max(0, min(end_ms, interval['end_ms']) - max(start_ms, interval['start_ms'])) / 1000
            if all(interval.get(f'{channel}_valid', False) for channel in durations):
                joint_valid_duration += dt
            for channel in durations:
                if interval.get(f'{channel}_valid', False):
                    durations[channel] += dt
            if interval.get('eye_valid', False) and interval.get('eye_closed', False):
                closed_duration += dt
            if interval.get('mouth_valid', False) and interval.get('mouth_open', False):
                mouth_open_duration += dt
        for event in sample.event_summaries.get('events', []):
            key = (event['kind'], event['start_ms'], event['end_ms'])
            events[key] = event
    for channel, duration in durations.items():
        result[f'{channel}_valid_duration_s'] = duration
        result[f'{channel}_valid_ratio'] = duration / span
    result['missing_duration_ratio'] = 1.0 - joint_valid_duration / span
    result['closure_duration_s'] = closed_duration
    result['mouth_open_duration_s'] = mouth_open_duration
    for kind, prefix in (('closure', 'blink'), ('yawn', 'yawn')):
        completed = [e for e in events.values() if e['kind'] == kind and e.get('candidate')
                     and not e.get('censored') and start_ms < e['end_ms'] <= end_ms]
        event_durations = [float(e['duration_s']) for e in completed]
        result[f'{prefix}_count'] = float(len(completed))
        denominator = durations['eye' if kind == 'closure' else 'mouth']
        result[f'{prefix}_rate_per_min'] = len(completed) * 60 / denominator if denominator else math.nan
        result[f'{prefix}_mean_duration_s'] = float(np.mean(event_durations)) if event_durations else math.nan
        result[f'{prefix}_max_duration_s'] = max(event_durations) if event_durations else math.nan
    closures = [(min(end_ms, e['end_ms']) - max(start_ms, e['start_ms'])) / 1000
                for e in events.values() if e['kind'] == 'closure'
                and min(end_ms, e['end_ms']) > max(start_ms, e['start_ms'])]
    ongoing = float(rows[-1].event_summaries.get('closure_elapsed_s', 0)) if rows else 0.0
    if ongoing:
        closures.append(min(span, ongoing))
    result['closure_max_duration_s'] = max(closures, default=0.0)
    result['closure_mean_duration_s'] = float(np.mean(closures)) if closures else 0.0
    for name, column in (('perclos_60', 6), ('perclos_ready', 14), ('calibration_valid', 15)):
        result[name] = float(rows[-1].values[column]) if rows and rows[-1].validity[column] else math.nan
    return result
