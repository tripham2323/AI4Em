"""Frozen three-class diagnostic rules, never calibrated class probabilities."""
from __future__ import annotations

import math

import numpy as np

from src.contracts import DriverState, Prediction, TemporalSample
from src.features.temporal import FEATURE_NAMES, resolve_temporal_config

_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


class RuleBasedClassifier:
    """EAR-only and temporal rules share emitted timestamps and quality masks.

    Valid predictions use one-hot [Alert, Low Vigilance, Drowsy] as an explicit
    diagnostic convention. Invalid predictions have class_id=None and three NaNs.
    EAR-only does not predict Low Vigilance and does not use normalized closure.
    """

    def __init__(self, config: dict, *, mode: str):
        if mode not in ('ear_only', 'temporal'):
            raise ValueError('rule mode must be ear_only or temporal')
        self.config = resolve_temporal_config(config)
        self.mode = mode
        self.model_id = f'rules:{mode}'

    def _invalid(self, sample: TemporalSample, reason: str) -> Prediction:
        return Prediction(sample.timestamp_ms, np.full(3, np.nan, dtype=np.float32),
                          None, False, reason, self.model_id)

    def _valid(self, sample: TemporalSample, state: DriverState, reason: str) -> Prediction:
        diagnostic = np.zeros(3, dtype=np.float32)
        diagnostic[int(state)] = 1.
        return Prediction(sample.timestamp_ms, diagnostic, state, True,
                          f'{reason}; diagnostic_one_hot_not_calibrated_probabilities', self.model_id)

    def predict(self, sample: TemporalSample) -> Prediction:
        if sample.values.shape != (16,) or sample.validity.shape != (16,):
            raise ValueError('rule input must have the ordered 16 temporal channels')
        summary = sample.event_summaries
        if not summary.get('face_detected', False):
            return self._invalid(sample, 'no_face_or_stale_sample')
        if not summary.get('profile_valid', False):
            return self._invalid(sample, 'invalid_calibration_profile')
        for name in ('left_eye_valid', 'right_eye_valid', 'mouth_valid', 'pose_valid'):
            index = _INDEX[name]
            if not sample.validity[index] or sample.values[index] != 1.:
                return self._invalid(sample, f'missing_current_{name}')
        for name in ('ear_left_norm', 'ear_right_norm', 'mar_delta', 'pitch_delta',
                     'yaw_delta', 'roll_delta', 'pitch_velocity_dps'):
            index = _INDEX[name]
            if not sample.validity[index] or not math.isfinite(float(sample.values[index])):
                return self._invalid(sample, f'invalid_current_{name}')
        if self.mode == 'ear_only':
            absolute_ear = summary.get('absolute_ear_mean', math.nan)
            elapsed = summary.get('absolute_closure_elapsed_s', math.nan)
            if not math.isfinite(absolute_ear) or not math.isfinite(elapsed):
                return self._invalid(sample, 'invalid_absolute_ear_or_duration')
            if absolute_ear < self.config['rule_ear_threshold'] and elapsed >= self.config['rule_ear_closure_s']:
                return self._valid(sample, DriverState.DROWSY, 'absolute_ear_prolonged_closure')
            return self._valid(sample, DriverState.ALERT, 'absolute_ear_not_prolonged')
        closure_index = _INDEX['closure_elapsed_s']
        closure = float(sample.values[closure_index])
        if sample.validity[closure_index] and math.isfinite(closure) and closure >= self.config['rule_closure_s']:
            return self._valid(sample, DriverState.DROWSY, 'normalized_prolonged_closure')
        perclos_index = _INDEX['perclos_60']
        ready_index = _INDEX['perclos_ready']
        perclos = float(sample.values[perclos_index])
        if (not sample.validity[ready_index] or sample.values[ready_index] != 1.
                or not sample.validity[perclos_index] or not math.isfinite(perclos)):
            return self._invalid(sample, 'insufficient_valid_perclos_history')
        if perclos >= self.config['rule_drowsy_perclos']:
            return self._valid(sample, DriverState.DROWSY, 'ready_perclos_drowsy')
        if perclos >= self.config['rule_low_vigilance_perclos']:
            return self._valid(sample, DriverState.LOW_VIGILANCE, 'ready_perclos_low_vigilance')
        return self._valid(sample, DriverState.ALERT, 'ready_perclos_alert')
