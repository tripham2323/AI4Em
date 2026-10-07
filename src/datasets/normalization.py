"""Valid-only train statistics and strict 16-channel tensor conversion."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

FEATURE_NAMES = (
    'ear_left_norm', 'ear_right_norm', 'mar_delta', 'pitch_delta',
    'yaw_delta', 'roll_delta', 'perclos_60', 'closure_elapsed_s',
    'yawn_elapsed_s', 'pitch_velocity_dps', 'left_eye_valid',
    'right_eye_valid', 'mouth_valid', 'pose_valid', 'perclos_ready',
    'calibration_valid',
)


def scaler_hash(scaler: dict) -> str:
    """Hash every persisted field except the hash itself, including provenance."""
    payload = {key: value for key, value in scaler.items() if key != 'hash'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def validate_scaler(scaler: dict, feature_names: tuple[str, ...] = FEATURE_NAMES) -> None:
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError('feature names must be nonempty unique strings')
    if tuple(scaler.get('feature_names', ())) != names:
        raise ValueError('scaler feature order mismatch')
    if any(name not in FEATURE_NAMES for name in names):
        raise ValueError('scaler contains unknown feature')
    if scaler.get('hash') != scaler_hash(scaler):
        raise ValueError('scaler hash mismatch')
    continuous = [name for name in names if name in FEATURE_NAMES[:10]]
    expected = len(continuous)
    for key in ('means', 'scales', 'counts'):
        array = np.asarray(scaler.get(key, []), dtype=float)
        if array.shape != (expected,) or not np.isfinite(array).all():
            raise ValueError(f'invalid scaler {key}')
        if key != 'means' and np.any(array <= 0):
            raise ValueError(f'invalid scaler {key}')
        if key == 'counts' and np.any(array != np.floor(array)):
            raise ValueError('scaler counts must be integers')

def validated_arrays(values, validity) -> tuple[np.ndarray, np.ndarray]:
    """Validate pre-imputation data without requiring invalid slots to be finite."""
    values = np.asarray(values, dtype=np.float64)
    raw_validity = np.asarray(validity)
    if values.ndim != 2 or values.shape[1] != 16 or raw_validity.shape != values.shape:
        raise ValueError('values and validity must have matching Nx16 shapes')
    if not np.isin(raw_validity, (False, True)).all():
        raise ValueError('validity must contain only booleans')
    validity = raw_validity.astype(bool, copy=False)
    if not np.isfinite(values[validity]).all():
        raise ValueError('nonfinite valid feature value')
    if not validity[:, 10:].all() or not np.isin(values[:, 10:], (0, 1)).all():
        raise ValueError('binary flags must be finite valid booleans')
    # A channel asserted invalid cannot simultaneously claim valid measurements.
    for flag, channels in ((10, (0, 7)), (11, (1, 7)), (12, (2, 8)), (13, (3, 4, 5, 9)),
                           (14, (6,))):
        if np.any(validity[:, channels] & (values[:, flag:flag+1] == 0)):
            raise ValueError('binary flag contradicts feature validity')
    return values, validity


def fit_scaler(train_unique_timesteps: pd.DataFrame, feature_names: tuple[str, ...]) -> dict:
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or any(name not in FEATURE_NAMES for name in names):
        raise ValueError('feature order mismatch')
    frame = train_unique_timesteps
    required = {'video_id', 'segment_id', 'timestamp_ms', 'values', 'validity'}
    if not required.issubset(frame.columns):
        raise ValueError(f'missing train timestep columns: {sorted(required-set(frame.columns))}')
    if 'split' in frame and not frame['split'].eq('train').all():
        raise ValueError('scaler requires train rows only')
    unique = []
    seen = {}
    for row in frame.itertuples(index=False):
        key = (row.video_id, row.segment_id, row.timestamp_ms)
        values = np.asarray(row.values, dtype=float)
        validity = np.asarray(row.validity)
        if key in seen:
            old_values, old_validity = seen[key]
            if not (np.array_equal(values, old_values, equal_nan=True) and np.array_equal(validity, old_validity)):
                raise ValueError(f'conflicting duplicate train timestep {key}')
        else:
            seen[key] = (values, validity)
            unique.append((values, validity))
    if not unique:
        raise ValueError('blocked scaler: zero valid train observations')
    values, validity = validated_arrays(np.stack([row[0] for row in unique]), np.stack([row[1] for row in unique]))
    means, scales, counts = [], [], []
    for name in names:
        channel = FEATURE_NAMES.index(name)
        if channel >= 10:
            continue
        valid = values[validity[:, channel], channel]
        if len(valid) == 0:
            raise ValueError(f'blocked scaler: zero valid train observations for {name}')
        mean, scale = float(np.mean(valid)), float(np.std(valid, ddof=0))
        if not np.isfinite(mean) or not np.isfinite(scale):
            raise ValueError(f'nonfinite scaler statistics for {name}')
        means.append(mean)
        scales.append(scale if scale > 0 else 1.)
        counts.append(len(valid))
    scaler = dict(feature_names=list(names), means=means, scales=scales, counts=counts)
    scaler['hash'] = scaler_hash(scaler)
    return scaler

def transform_values(values, validity, scaler: dict) -> np.ndarray:
    names = tuple(scaler.get('feature_names', FEATURE_NAMES))
    validate_scaler(scaler, names)
    values, validity = validated_arrays(values, validity)
    output = np.zeros((values.shape[0], len(names)), dtype=np.float32)
    means = iter(np.asarray(scaler['means'], dtype=np.float64))
    scales = iter(np.asarray(scaler['scales'], dtype=np.float64))
    for out_col, name in enumerate(names):
        channel = FEATURE_NAMES.index(name)
        if channel < 10:
            mean = float(next(means)); scale = float(next(scales))
            valid = validity[:, channel]
            transformed = np.zeros(len(values), dtype=np.float64)
            np.subtract(values[:, channel], mean, out=transformed, where=valid)
            np.divide(transformed, scale, out=transformed, where=valid)
            output[:, out_col] = transformed.astype(np.float32)
        else:
            output[:, out_col] = values[:, channel].astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError('normalization produced nonfinite float32 values')
    return output

