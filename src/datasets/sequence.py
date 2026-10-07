"""One shared accepted-window index for RF summaries and lazy LSTM sequences.

Endpoints are sampled ticks (a 100-tick window spans 9900ms); nominal duration
is 10000ms. Row endpoints are exclusive and address timestamp-sorted video data.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .normalization import FEATURE_NAMES, transform_values, validate_scaler, validated_arrays

INDEX_DTYPES = dict(video_id='object', subject_id='object', label_id='int64',
    split='object', segment_id='int64', start_row='int64', end_row='int64',
    start_timestamp_ms='int64', end_timestamp_ms='int64', profile_hash='object')
TEMPORAL_COLUMNS = ('video_id', 'subject_id', 'label_id', 'split', 'timestamp_ms',
                    'segment_id', 'values', 'validity', 'event_summaries')


def _integer_ticks(series, name):
    values = np.asarray(series, dtype=float)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError(f'{name} must contain finite integer values')
    return values.astype(np.int64)


def _video_frame(frame):
    missing = set(TEMPORAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'missing temporal columns: {sorted(missing)}')
    frame = frame.sort_values('timestamp_ms', kind='stable').reset_index(drop=True)
    times = _integer_ticks(frame.timestamp_ms, 'timestamp_ms')
    if len(times) and np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be unique and strictly increasing')
    _integer_ticks(frame.segment_id, 'segment_id')
    for name in ('video_id', 'subject_id', 'label_id', 'split', 'profile_hash'):
        if name in frame and frame[name].nunique(dropna=False) != 1 and len(frame):
            raise ValueError(f'inconsistent video {name}')
    if len(frame) and int(frame.label_id.iloc[0]) not in (0, 1, 2):
        raise ValueError('label_id must be video_weak class 0/1/2')
    return frame


def _role(video_id, frame, split):
    roles = split.get('roles', {})
    matches = [role for role, members in roles.items() if video_id in members.get('video_ids', [])]
    if roles and len(matches) != 1:
        raise ValueError(f'video {video_id} must belong to exactly one split role')
    role = matches[0] if matches else str(frame.split.iloc[0])
    if len(frame) and not frame.split.eq(role).all():
        raise ValueError(f'video {video_id} split role mismatch')
    return role


def build_window_index(temporal: pd.DataFrame, *, split: dict, config: dict) -> pd.DataFrame:
    for key, expected in (('sequence_fps', 10), ('sequence_window_s', 10), ('stride_s', 1)):
        if float(config.get(key, expected)) != expected:
            raise ValueError(f'{key} must be {expected} for the frozen window contract')
    if float(config.get('max_missing_ratio', .20)) != .20:
        raise ValueError('max_missing_ratio must be 0.20')
    records = []
    coverage = dict(scheduled=0, accepted=0, rejected=0, reasons={}, by_video={}, by_role={},
                    steps=100, stride_ticks=10, tick_ms=100, nominal_duration_ms=10000)
    mode = config.get('calibration_mode', config.get('mode', 'P1'))
    ranges = split.get('reserved_ranges', {}).get(mode, {})
    if not set(TEMPORAL_COLUMNS).issubset(temporal.columns):
        raise ValueError('missing temporal columns')
    for video_id, original in temporal.groupby('video_id', sort=True):
        frame = _video_frame(original)
        role = _role(video_id, frame, split)
        counters = dict(scheduled=0, accepted=0, rejected=0, reasons={})
        coverage['by_video'][str(video_id)] = counters
        role_counts = coverage['by_role'].setdefault(role, dict(scheduled=0, accepted=0, rejected=0, reasons={}))
        values = np.stack(frame['values'].to_numpy()).astype(float)
        validity = np.stack(frame['validity'].to_numpy())
        if values.shape != (len(frame), 16) or validity.shape != values.shape:
            raise ValueError('temporal values/validity must be Nx16')
        if not np.isin(values[:, 10:], (0, 1)).all():
            raise ValueError('binary flags must be finite 0/1')
        times = frame.timestamp_ms.to_numpy(dtype=np.int64)
        segments = frame.segment_id.to_numpy(dtype=np.int64)
        # Schedule within each segment; gaps remain rejected rather than stitched.
        for segment in pd.unique(segments):
            positions = np.flatnonzero(segments == segment)
            if len(positions) and np.any(np.diff(positions) != 1):
                raise ValueError('segment rows must be contiguous')
            for start in positions[::10]:
                end = int(start)+100
                if end > positions[-1]+1:
                    break
                reason = None
                window_times = times[start:end]
                if np.any(np.diff(window_times) != 100):
                    reason = 'timestamp_gap'
                reserved = ranges.get(str(video_id), ranges.get(video_id))
                if reserved is not None:
                    intervals = reserved if isinstance(reserved, list) else [reserved]
                    for interval in intervals:
                        if isinstance(interval, dict):
                            lo, hi = interval['start_ms'], interval['end_ms']
                        else:
                            lo, hi = interval
                        if window_times[0] < hi and window_times[-1] >= lo:
                            reason = reason or 'reserved_prefix'
                missing = np.any(values[start:end, 10:14] == 0, axis=1)
                if missing[-1]:
                    reason = reason or 'current_invalid'
                if np.count_nonzero(missing) > 20:
                    reason = reason or 'missing_ratio'
                for target in (coverage, counters, role_counts):
                    target['scheduled'] += 1
                    target['rejected' if reason else 'accepted'] += 1
                    if reason:
                        target['reasons'][reason] = target['reasons'].get(reason, 0)+1
                if reason:
                    continue
                row = frame.iloc[int(start)]
                records.append(dict(video_id=str(video_id), subject_id=str(row.subject_id),
                    label_id=int(row.label_id), split=role, segment_id=int(segment),
                    start_row=int(start), end_row=end, start_timestamp_ms=int(window_times[0]),
                    end_timestamp_ms=int(window_times[-1]),
                    profile_hash=str(row.get('profile_hash', ''))))
    # Preserve zero-row membership when source metadata is available.
    for source in split.get('sources', []):
        video_id = str(source['video_id'])
        coverage['by_video'].setdefault(video_id, dict(scheduled=0, accepted=0, rejected=0, reasons={}))
    for role in split.get('roles', {}):
        coverage['by_role'].setdefault(role, dict(scheduled=0, accepted=0, rejected=0, reasons={}))
    index = pd.DataFrame.from_records(records, columns=list(INDEX_DTYPES)).astype(INDEX_DTYPES)
    index.attrs['coverage'] = coverage
    return index


def _manifest_videos(manifest):
    if tuple(manifest.get('feature_names', ())) != FEATURE_NAMES:
        raise ValueError('manifest feature order mismatch')
    for key in ('schema_version', 'derived_hash', 'split_hash', 'snapshot_sha256'):
        if not manifest.get(key):
            raise ValueError(f'manifest missing {key}')
    if manifest['schema_version'] != 'facial_features_v1':
        raise ValueError('unsupported derived schema_version')
    videos = {}
    for entry in manifest.get('videos', []):
        video_id = str(entry['video_id'])
        if video_id in videos:
            raise ValueError('duplicate manifest video')
        path = Path(entry['path'])
        if not path.is_absolute() or not path.is_file() or path.suffix.lower() != '.parquet':
            raise ValueError(f'invalid derived cache path for {video_id}')
        videos[video_id] = entry
    return videos


def _load_video(entry):
    frame = _video_frame(pd.read_parquet(entry['path']))
    if len(frame) != int(entry['rows']):
        raise ValueError('cache row count differs from manifest')
    for key in ('video_id', 'subject_id', 'label_id', 'split', 'profile_hash'):
        if key in entry:
            if key not in frame or not frame[key].eq(entry[key]).all():
                raise ValueError(f'cache {key} differs from manifest')
    if len(frame):
        validated_arrays(np.stack(frame['values']), np.stack(frame['validity']))
    return frame


def _validate_slice(row, frame):
    start, end = int(row.start_row), int(row.end_row)
    if start < 0 or end > len(frame) or end-start != 100:
        raise ValueError('index must select exactly 100 cache rows')
    window = frame.iloc[start:end]
    times = window.timestamp_ms.to_numpy(dtype=np.int64)
    if (times[0] != int(row.start_timestamp_ms) or times[-1] != int(row.end_timestamp_ms)
            or np.any(np.diff(times) != 100)):
        raise ValueError('index timestamp/cache mismatch')
    for key in ('video_id', 'subject_id', 'label_id', 'split', 'segment_id', 'profile_hash'):
        if key in window and not window[key].eq(getattr(row, key)).all():
            raise ValueError(f'index {key}/cache mismatch')
    values = np.stack(window['values'])
    missing = np.any(values[:, 10:14] == 0, axis=1)
    if missing[-1] or np.count_nonzero(missing) > 20:
        raise ValueError('index references rejected window')
    return start, end


def unique_train_timesteps(index: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Union accepted training slices, loading derived Parquet only, never raw video."""
    videos = _manifest_videos(manifest)
    frames = []
    for video_id, windows in index[index.split.eq('train')].groupby('video_id', sort=True):
        if str(video_id) not in videos:
            raise ValueError('index video missing from manifest')
        entry = videos[str(video_id)]
        if entry['split'] != 'train':
            raise ValueError('train index references nontrain cache')
        frame = _load_video(entry)
        selected = np.zeros(len(frame), bool)
        for row in windows.itertuples(index=False):
            start, end = _validate_slice(row, frame)
            selected[start:end] = True
        frames.append(frame.loc[selected])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TEMPORAL_COLUMNS)


class SequenceDataset:
    """Lazy per-video immutable array cache; returned tensors never alias the cache."""

    def __init__(self, index: pd.DataFrame, *, derived_manifest: dict,
                 scaler: dict, feature_names: tuple[str, ...]):
        validate_scaler(scaler, feature_names)
        self._videos = _manifest_videos(derived_manifest)
        if not set(INDEX_DTYPES).issubset(index.columns):
            raise ValueError('incomplete window index schema')
        for key in ('schema_version', 'derived_hash', 'split_hash', 'snapshot_sha256', 'mode', 'outer_index'):
            if key in scaler and scaler[key] != derived_manifest.get(key):
                raise ValueError(f'scaler/manifest {key} mismatch')
        self._rows = list(index.itertuples(index=False))
        self._rows_by_video = {}
        for row in self._rows:
            if str(row.video_id) not in self._videos:
                raise ValueError('index video missing from manifest')
            entry = self._videos[str(row.video_id)]
            for key in ('subject_id', 'label_id', 'split', 'profile_hash'):
                if key in entry and getattr(row, key) != entry[key]:
                    raise ValueError(f'index/manifest {key} mismatch')
            self._rows_by_video.setdefault(str(row.video_id), []).append(row)
        self._scaler = dict(scaler)
        self._cache = {}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        row = self._rows[index]
        video_id = str(row.video_id)
        if video_id not in self._cache:
            frame = _load_video(self._videos[video_id])
            # Validate all this video's requested slices once, not on every access.
            for requested in self._rows_by_video[video_id]:
                _validate_slice(requested, frame)
            values = np.stack(frame['values'])
            validity = np.stack(frame['validity'])
            normalized = transform_values(values, validity, self._scaler)
            normalized.flags.writeable = False
            self._cache[video_id] = normalized
        x = self._cache[video_id][int(row.start_row):int(row.end_row)].copy()
        metadata = {key: getattr(row, key) for key in INDEX_DTYPES if key != 'label_id'}
        return x, int(row.label_id), metadata
