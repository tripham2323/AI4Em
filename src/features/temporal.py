"""Native event detection plus causal 10 Hz features; no raw extraction changes.

Receipts are immutable JSON-compatible deltas, not copies of the full history:
``held_intervals`` contains real held-time slices since the previous grid output,
with start_ms/end_ms and eye_valid/mouth_valid/pose_valid/closed_proxy flags,
plus the native hysteretic eye_closed/mouth_open states.
``events`` contains newly completed/censored native events, with kind, start_ms,
end_ms, duration_s, censored, candidate (boolean) and classification. Closure
candidate=True means a completed blink; yawn candidate=True means >=2 seconds.
RF intersects held intervals with its [start, end) window and counts uncensored
completions with start < end_ms <= end.
Open event starts and elapsed times are separate scalars. No event transition is
replayed on grid samples. The engine's rolling history is bounded to 60 seconds.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from typing import Any

import numpy as np

from src.calibration.profile import transform_sample
from src.contracts import CalibrationProfile, FeatureSample, TemporalSample

FEATURE_NAMES = (
    'ear_left_norm', 'ear_right_norm', 'mar_delta', 'pitch_delta', 'yaw_delta',
    'roll_delta', 'perclos_60', 'closure_elapsed_s', 'yawn_elapsed_s',
    'pitch_velocity_dps', 'left_eye_valid', 'right_eye_valid', 'mouth_valid',
    'pose_valid', 'perclos_ready', 'calibration_valid',
)

_DEFAULTS = {
    'schema_version': 'facial_temporal_v1', 'sequence_fps': 10,
    'max_sample_age_ms': 100, 'max_gap_s': 1., 'statistics_window_s': 60.,
    'blink_enter': .65, 'blink_exit': .75, 'blink_min_s': .10, 'blink_max_s': .80,
    'perclos_threshold': .20, 'perclos_min_history_s': 30., 'perclos_min_coverage': .80,
    'yawn_enter': .35, 'yawn_exit': .25, 'yawn_min_s': 2.,
    'rule_ear_threshold': .20, 'rule_ear_closure_s': 1., 'rule_closure_s': 1.5,
    'rule_drowsy_perclos': .25, 'rule_low_vigilance_perclos': .15,
}


def resolve_temporal_config(config: dict) -> dict:
    """Resolve/freeze phase-local values, rejecting unsupported timing semantics."""
    resolved = {key: config.get(key, default) for key, default in _DEFAULTS.items()}
    if resolved['schema_version'] != 'facial_temporal_v1':
        raise ValueError('unsupported temporal schema_version')
    for key in _DEFAULTS:
        if key == 'schema_version':
            continue
        number = resolved[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'{key} must be a finite number')
        resolved[key] = float(number)
    if resolved['sequence_fps'] != 10 or resolved['max_sample_age_ms'] != 100:
        raise ValueError('temporal v1 requires 10 Hz and a 100 ms hold cap')
    for key in ('max_gap_s', 'statistics_window_s', 'blink_min_s', 'blink_max_s',
                'perclos_min_history_s', 'yawn_min_s', 'rule_ear_closure_s', 'rule_closure_s'):
        if resolved[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= resolved['blink_enter'] < resolved['blink_exit']:
        raise ValueError('blink hysteresis requires enter < exit')
    if not 0 <= resolved['yawn_exit'] < resolved['yawn_enter']:
        raise ValueError('mouth hysteresis requires exit < enter')
    if resolved['blink_min_s'] > resolved['blink_max_s']:
        raise ValueError('blink duration minimum exceeds maximum')
    if resolved['perclos_min_history_s'] > resolved['statistics_window_s']:
        raise ValueError('PERCLOS minimum history exceeds statistics window')
    for key in ('perclos_threshold', 'perclos_min_coverage', 'rule_drowsy_perclos', 'rule_low_vigilance_perclos'):
        if not 0 <= resolved[key] <= 1:
            raise ValueError(f'{key} must be within [0,1]')
    if not 0 <= resolved['rule_low_vigilance_perclos'] <= resolved['rule_drowsy_perclos']:
        raise ValueError('rule Low Vigilance threshold exceeds Drowsy threshold')
    if resolved['rule_ear_threshold'] <= 0:
        raise ValueError('rule_ear_threshold must be positive')
    return resolved


def temporal_config_hash(config: dict) -> str:
    """Identity includes resolved defaults rather than only explicitly set keys."""
    encoded = json.dumps(resolve_temporal_config(config), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


class TemporalFeatureExtractor:
    """Events use native timestamps; output uses latest-not-future held samples."""

    def __init__(self, profile: CalibrationProfile, config: dict):
        self.profile = profile
        self.config = resolve_temporal_config(config)
        self.config_sha256 = temporal_config_hash(self.config)
        self._segment_id = -1
        self.reset()

    def reset(self) -> None:
        """Clear all causal history and start a new segment on the next sample."""
        self._segment_id += 1
        self._source: str | None = None
        self._latest: FeatureSample | None = None
        self._transformed: dict[str, float | bool] | None = None
        self._segment_start = 0
        self._next_tick = 0
        self._clock = 0
        self._pitch_velocity = math.nan
        self._starts: dict[str, int | None] = {'closure': None, 'yawn': None, 'absolute_closure': None}
        self._intervals: deque[dict[str, Any]] = deque()
        self._events: deque[dict[str, Any]] = deque()
        self._closure_events: deque[dict[str, Any]] = deque()
        self._closure_max: deque[dict[str, Any]] = deque()
        self._event_counts = {'closure': 0, 'blink': 0, 'yawn': 0,
                              'censored_closure': 0, 'censored_yawn': 0}
        self._event_duration_sums = {'closure': 0., 'blink': 0., 'yawn': 0.}
        self._pending_intervals: list[dict[str, Any]] = []
        self._pending_events: list[dict[str, Any]] = []
        self._durations = {'eye_valid': 0., 'mouth_valid': 0., 'pose_valid': 0., 'closed_proxy': 0.}

    def update(
        self,
        sample: FeatureSample,
        profile: CalibrationProfile | None = None,
    ) -> list[TemporalSample]:
        """Process one native sample, optionally binding a new session profile.

        Offline consumers bind the profile in the constructor. Realtime
        calibration may provide the completed profile on each call; changing it
        is accepted only while the extractor has no active history.
        """
        if profile is not None and profile != self.profile:
            if self._latest is not None:
                raise ValueError("cannot change calibration profile during an active temporal segment")
            self.profile = profile
        if self._latest is not None:
            if sample.source_id == self._source and sample.timestamp_ms <= self._latest.timestamp_ms:
                raise ValueError('native timestamp must strictly increase within a source')
            if sample.source_id != self._source or sample.timestamp_ms - self._latest.timestamp_ms > self.config['max_gap_s'] * 1000:
                self.reset()
        if self._latest is None:
            self._source = sample.source_id
            self._segment_start = self._next_tick = self._clock = sample.timestamp_ms
        result = []
        # The arriving sample must not influence any grid timestamp before it.
        while self._next_tick < sample.timestamp_ms:
            self._advance(self._next_tick)
            result.append(self._emit(self._next_tick))
            self._next_tick += 100
        self._advance(sample.timestamp_ms)
        previous = self._latest
        previous_transform = self._transformed
        transformed = dict(transform_sample(sample, self.profile))
        # Face absence is authoritative even for manually constructed samples.
        if not sample.face_detected or not self.profile.valid:
            for key in ('left_eye_valid', 'right_eye_valid', 'mouth_valid', 'pose_valid'):
                transformed[key] = False
        for names, flag in ((('ear_left_norm',), 'left_eye_valid'), (('ear_right_norm',), 'right_eye_valid'),
                            (('mar_delta',), 'mouth_valid'), (('pitch_delta', 'yaw_delta', 'roll_delta'), 'pose_valid')):
            if not transformed[flag] or not all(math.isfinite(float(transformed[name])) for name in names):
                transformed[flag] = False
                for name in names:
                    transformed[name] = math.nan
        self._pitch_velocity = math.nan
        if (previous is not None and previous_transform is not None and previous_transform['pose_valid']
                and transformed['pose_valid'] and sample.timestamp_ms - previous.timestamp_ms <= 100):
            delta = (float(transformed['pitch_delta']) - float(previous_transform['pitch_delta']) + 180.) % 360. - 180.
            self._pitch_velocity = delta * 1000 / (sample.timestamp_ms - previous.timestamp_ms)
        self._latest = sample
        self._transformed = transformed
        self._transition(sample.timestamp_ms)
        if self._next_tick == sample.timestamp_ms:
            result.append(self._emit(self._next_tick))
            self._next_tick += 100
        return result

    def _flags(self) -> dict[str, bool]:
        transformed = self._transformed
        if transformed is None:
            return dict(eye_valid=False, mouth_valid=False, pose_valid=False, closed_proxy=False,
                        eye_closed=False, mouth_open=False)
        eye = bool(transformed['left_eye_valid'] and transformed['right_eye_valid'])
        return {
            'eye_valid': eye, 'mouth_valid': bool(transformed['mouth_valid']),
            'pose_valid': bool(transformed['pose_valid'] and math.isfinite(self._pitch_velocity)),
            'closed_proxy': eye and float(transformed['ear_left_norm']) <= self.config['perclos_threshold']
                           and float(transformed['ear_right_norm']) <= self.config['perclos_threshold'],
            'eye_closed': eye and self._starts['closure'] is not None,
            'mouth_open': bool(transformed['mouth_valid'] and self._starts['yawn'] is not None),
        }

    @staticmethod
    def _append_merged(records, interval: dict[str, Any]) -> None:
        if records and records[-1]['end_ms'] == interval['start_ms'] and all(
                records[-1][key] == interval[key] for key in
                ('eye_valid', 'mouth_valid', 'pose_valid', 'closed_proxy', 'eye_closed', 'mouth_open')):
            records[-1]['end_ms'] = interval['end_ms']
        else:
            records.append(interval.copy())

    def _record_interval(self, begin: int, end: int, flags: dict[str, bool]) -> None:
        if end <= begin:
            return
        interval = {'start_ms': begin, 'end_ms': end, **flags}
        self._append_merged(self._intervals, interval)
        self._append_merged(self._pending_intervals, interval)
        for key in self._durations:
            if flags[key]:
                self._durations[key] += end - begin

    def _advance(self, target: int) -> None:
        if self._latest is not None and target > self._clock:
            expiry = self._latest.timestamp_ms + 100
            held_end = min(target, expiry)
            self._record_interval(self._clock, held_end, self._flags())
            if target > expiry:
                for kind in self._starts:
                    self._end_event(kind, expiry, censored=True)
                self._record_interval(max(self._clock, expiry), target,
                                      dict(eye_valid=False, mouth_valid=False, pose_valid=False, closed_proxy=False,
                                           eye_closed=False, mouth_open=False))
        self._clock = target
        cutoff = target - self.config['statistics_window_s'] * 1000
        while self._intervals and self._intervals[0]['start_ms'] < cutoff:
            interval = self._intervals[0]
            removed_end = min(interval['end_ms'], cutoff)
            removed = removed_end - interval['start_ms']
            for key in self._durations:
                if interval[key]:
                    self._durations[key] -= removed
            if interval['end_ms'] <= cutoff:
                self._intervals.popleft()
            else:
                interval['start_ms'] = cutoff
                break
        while self._events and self._events[0]['end_ms'] <= cutoff:
            event = self._events.popleft()
            self._adjust_event_stats(event, -1)
            if event['kind'] == 'closure':
                self._closure_events.popleft()
        while self._closure_max and self._closure_max[0]['end_ms'] <= cutoff:
            self._closure_max.popleft()

    def _transition(self, timestamp: int) -> None:
        transformed = self._transformed
        flags = self._flags()
        left, right = float(transformed['ear_left_norm']), float(transformed['ear_right_norm'])
        if not flags['eye_valid']:
            self._end_event('closure', timestamp, censored=True)
            self._end_event('absolute_closure', timestamp, censored=True)
        else:
            if self._starts['closure'] is None and left < self.config['blink_enter'] and right < self.config['blink_enter']:
                self._starts['closure'] = timestamp
            elif self._starts['closure'] is not None and left > self.config['blink_exit'] and right > self.config['blink_exit']:
                self._end_event('closure', timestamp, censored=False)
            if self._absolute_ear() < self.config['rule_ear_threshold']:
                if self._starts['absolute_closure'] is None:
                    self._starts['absolute_closure'] = timestamp
            else:
                self._end_event('absolute_closure', timestamp, censored=False)
        if not flags['mouth_valid']:
            self._end_event('yawn', timestamp, censored=True)
        else:
            mar = float(transformed['mar_delta'])
            if self._starts['yawn'] is None and mar > self.config['yawn_enter']:
                self._starts['yawn'] = timestamp
            elif self._starts['yawn'] is not None and mar < self.config['yawn_exit']:
                self._end_event('yawn', timestamp, censored=False)

    def _end_event(self, kind: str, timestamp: int, *, censored: bool) -> None:
        start = self._starts[kind]
        if start is None:
            return
        duration = (timestamp - start) / 1000.
        classification = None
        if not censored:
            if kind == 'closure':
                if self.config['blink_min_s'] <= duration <= self.config['blink_max_s']:
                    classification = 'blink'
                elif duration > self.config['blink_max_s']:
                    classification = 'prolonged'
            elif kind == 'yawn' and duration >= self.config['yawn_min_s']:
                classification = 'yawn'
        event = {'kind': kind, 'start_ms': start, 'end_ms': timestamp,
                 'duration_s': duration, 'censored': censored,
                 'candidate': classification in ('blink', 'yawn'), 'classification': classification}
        self._events.append(event)
        self._adjust_event_stats(event, 1)
        if kind == 'closure':
            self._closure_events.append(event)
            while self._closure_max and self._closure_max[-1]['duration_s'] <= duration:
                self._closure_max.pop()
            self._closure_max.append(event)
        self._pending_events.append(event.copy())
        self._starts[kind] = None

    def _adjust_event_stats(self, event: dict[str, Any], direction: int) -> None:
        """Amortized O(1) event statistics, updated only at completion/expiry."""
        kind = event['kind']
        counted = []
        if kind == 'closure':
            counted.append('closure')
            if event['candidate']:
                counted.append('blink')
            if event['censored']:
                self._event_counts['censored_closure'] += direction
        elif kind == 'yawn':
            if event['candidate']:
                counted.append('yawn')
            if event['censored']:
                self._event_counts['censored_yawn'] += direction
        for key in counted:
            self._event_counts[key] += direction
            self._event_duration_sums[key] += direction * event['duration_s']
            if self._event_counts[key] == 0:
                self._event_duration_sums[key] = 0.

    def _absolute_ear(self) -> float:
        if self._latest is None or not self._flags()['eye_valid']:
            return math.nan
        return (self._latest.ear_left + self._latest.ear_right) / 2.

    def _elapsed(self, kind: str, timestamp: int, valid: bool) -> float:
        if not valid:
            return math.nan
        start = self._starts[kind]
        return 0. if start is None else (timestamp - start) / 1000.

    def _emit(self, timestamp: int) -> TemporalSample:
        usable = self._latest is not None and timestamp - self._latest.timestamp_ms <= 100
        transformed = self._transformed if usable else None
        flags = self._flags() if usable else dict(eye_valid=False, mouth_valid=False, pose_valid=False, closed_proxy=False)
        history_ms = min(timestamp - self._segment_start, self.config['statistics_window_s'] * 1000)
        coverage = self._durations['eye_valid'] / history_ms if history_ms > 0 else 0.
        raw_perclos = self._durations['closed_proxy'] / self._durations['eye_valid'] if self._durations['eye_valid'] > 0 else math.nan
        ready = history_ms >= self.config['perclos_min_history_s'] * 1000 and coverage >= self.config['perclos_min_coverage']
        values = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
        if transformed is not None:
            for index, name in enumerate(FEATURE_NAMES[:6]):
                values[index] = transformed[name]
            if not flags['pose_valid']:
                values[3:6] = np.nan
        values[6] = raw_perclos if ready else math.nan
        values[7] = self._elapsed('closure', timestamp, flags['eye_valid'])
        values[8] = self._elapsed('yawn', timestamp, flags['mouth_valid'])
        values[9] = self._pitch_velocity if flags['pose_valid'] else math.nan
        values[10:] = (bool(transformed and transformed['left_eye_valid']), bool(transformed and transformed['right_eye_valid']),
                       flags['mouth_valid'], flags['pose_valid'], ready,
                       bool(self._transformed and self._transformed['calibration_valid']))
        validity = np.isfinite(values)
        counts = self._event_counts
        duration_sums = self._event_duration_sums
        # Closure intervals do not overlap: only the oldest can straddle cutoff.
        closure_sum = duration_sums['closure']
        completed_max = self._closure_max[0]['duration_s'] if self._closure_max else 0.
        if self._closure_events:
            first = self._closure_events[0]
            trimmed_s = max(0., timestamp - history_ms - first['start_ms']) / 1000.
            closure_sum -= trimmed_s
            if trimmed_s > 0 and self._closure_max[0] is first:
                next_max = self._closure_max[1]['duration_s'] if len(self._closure_max) > 1 else 0.
                completed_max = max(first['duration_s'] - trimmed_s, next_max)
        current_closure = self._elapsed('closure', timestamp, flags['eye_valid'])
        eye_s, mouth_s = self._durations['eye_valid'] / 1000., self._durations['mouth_valid'] / 1000.
        summary = {
            'face_detected': bool(usable and self._latest.face_detected),
            'profile_valid': bool(self.profile.valid),
            'absolute_ear_mean': self._absolute_ear() if usable else math.nan,
            'absolute_closure_elapsed_s': self._elapsed('absolute_closure', timestamp, flags['eye_valid']),
            'closure_elapsed_s': current_closure,
            'yawn_elapsed_s': self._elapsed('yawn', timestamp, flags['mouth_valid']),
            'closure_start_ms': self._starts['closure'], 'yawn_start_ms': self._starts['yawn'],
            'absolute_closure_start_ms': self._starts['absolute_closure'],
            'history_s': history_ms / 1000., 'coverage': coverage, 'perclos_raw': raw_perclos,
            'perclos_ready': bool(ready), 'eye_valid_duration_s': eye_s,
            'mouth_valid_duration_s': mouth_s, 'pose_valid_duration_s': self._durations['pose_valid'] / 1000.,
            'closed_proxy_duration_s': self._durations['closed_proxy'] / 1000.,
            'blink_count': counts['blink'], 'yawn_count': counts['yawn'],
            'blink_rate_per_min': counts['blink'] * 60 / eye_s if eye_s > 0 else math.nan,
            'yawn_rate_per_min': counts['yawn'] * 60 / mouth_s if mouth_s > 0 else math.nan,
            'blink_mean_duration_s': duration_sums['blink'] / counts['blink'] if counts['blink'] else math.nan,
            'yawn_mean_duration_s': duration_sums['yawn'] / counts['yawn'] if counts['yawn'] else math.nan,
            'closure_mean_duration_s': closure_sum / counts['closure'] if counts['closure'] else math.nan,
            'closure_max_duration_s': (max(completed_max, min(current_closure, history_ms / 1000))
                                       if math.isfinite(current_closure) else
                                       completed_max if counts['closure'] else math.nan),
            'censored_closure_count': counts['censored_closure'],
            'censored_mouth_count': counts['censored_yawn'],
            'held_intervals': self._pending_intervals, 'events': self._pending_events,
        }
        # Transfer lists, never reuse or mutate objects already published in receipts.
        self._pending_intervals = []
        self._pending_events = []
        values.setflags(write=False)
        validity.setflags(write=False)
        return TemporalSample(timestamp, values, validity, self._segment_id, summary)
