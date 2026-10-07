"""Shared index boundaries and lazy cached tensor isolation."""
import numpy as np
import pandas as pd
import pytest

from src.datasets.normalization import FEATURE_NAMES, fit_scaler
from src.datasets.sequence import SequenceDataset, build_window_index, unique_train_timesteps


def temporal(n=120, video='v'):
    values = np.ones((n, 16), np.float32)
    values[:, :10] = np.arange(n)[:, None]
    validity = np.ones((n, 16), bool)
    if n:
        values[0, 14] = 0  # Startup PERCLOS alone must not reject a window.
        validity[0, 6] = False
    return pd.DataFrame(dict(video_id=[video]*n, subject_id=['s']*n,
        label_id=[2]*n, split=['train']*n, timestamp_ms=np.arange(n)*100,
        segment_id=[0]*n, values=list(values),
        validity=list(validity), event_summaries=['{}']*n,
        profile_hash=['profile']*n))

def test_first_complete_window_and_stride_use_tick_endpoints():
    index = build_window_index(temporal(), split={}, config={})
    assert index.start_row.tolist() == [0, 10, 20]
    assert index.end_row.tolist() == [100, 110, 120]
    assert index.end_timestamp_ms.tolist() == [9900, 10900, 11900]
    assert index.label_id.tolist() == [2]*3
    assert index.attrs['coverage']['scheduled'] == 3


def test_sequence_grid_origin_is_source_relative_after_prefix():
    frame = temporal(100)
    frame['timestamp_ms'] += 30039
    index = build_window_index(frame, split={}, config={})
    assert index.start_timestamp_ms.tolist() == [30039]
    assert index.end_timestamp_ms.tolist() == [39939]


@pytest.mark.parametrize('missing,expected', [(20, 1), (21, 0)])
def test_missing_ratio_boundary(missing, expected):
    frame = temporal(100)
    for row in range(missing):
        values = frame.at[row, 'values'].copy()
        values[10] = 0
        frame.at[row, 'values'] = values
        validity = frame.at[row, 'validity'].copy()
        validity[0] = False
        frame.at[row, 'validity'] = validity
    index = build_window_index(frame, split={}, config={})
    assert len(index) == expected
    assert index.attrs['coverage']['scheduled'] == 1
    assert index.attrs['coverage']['rejected'] == 1-expected


def test_current_invalid_and_reserved_prefix_reject_complete_windows():
    frame = temporal(100)
    values = frame.at[99, 'values'].copy()
    values[13] = 0
    frame.at[99, 'values'] = values
    assert build_window_index(frame, split={}, config={}).empty
    split = {'reserved_ranges': {'P1': {'v': {'start_ms': 0, 'end_ms': 1000}}}}
    index = build_window_index(temporal(), split=split, config={'calibration_mode': 'P1'})
    assert index.start_row.tolist() == [10, 20]
    assert index.attrs['coverage']['reasons']['reserved_prefix'] == 1


def test_windows_never_cross_timestamp_gaps_or_segments():
    frame = temporal(120)
    frame.loc[60:, 'timestamp_ms'] += 1500
    index = build_window_index(frame, split={}, config={})
    assert index.empty
    assert index.attrs['coverage']['rejected'] == 3
    frame = temporal(120)
    frame.loc[60:, 'segment_id'] = 1
    assert build_window_index(frame, split={}, config={}).empty


def test_timestamp_duplicates_are_not_silently_sorted_or_padded():
    frame = temporal(100)
    frame.loc[40, 'timestamp_ms'] = 3900
    with pytest.raises(ValueError, match='timestamp'):
        build_window_index(frame, split={}, config={})
    empty = build_window_index(temporal(0), split={}, config={})
    assert empty.empty and empty.attrs['coverage']['scheduled'] == 0
    assert empty.end_row.dtype == np.dtype('int64')


def manifest(tmp_path, frame):
    path = tmp_path/'v.parquet'
    frame.to_parquet(path, index=False)
    return dict(schema_version='facial_features_v1', derived_hash='derived',
        split_hash='split', snapshot_sha256='snapshot', feature_names=list(FEATURE_NAMES),
        videos=[dict(video_id='v', subject_id='s', label_id=2, split='train',
                     profile_hash='profile', path=str(path), rows=len(frame))])


def test_lazy_load_union_train_positions_and_no_returned_cache_alias(tmp_path, monkeypatch):
    frame = temporal()
    index = build_window_index(frame, split={}, config={})
    derived = manifest(tmp_path, frame)
    train = unique_train_timesteps(index, derived)
    assert len(train) == 120
    scaler = fit_scaler(train, FEATURE_NAMES)
    reads = []
    original = pd.read_parquet
    def tracked(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', tracked)
    dataset = SequenceDataset(index, derived_manifest=derived, scaler=scaler,
                              feature_names=FEATURE_NAMES)
    assert reads == []
    x, label, metadata = dataset[0]
    assert label == 2 and x.shape == (100, 16) and x.dtype == np.float32
    assert metadata['video_id'] == 'v' and metadata['end_timestamp_ms'] == 9900
    assert all(not isinstance(value, (np.ndarray, list)) for value in metadata.values())
    x[:] = 999
    assert not np.any(dataset[0][0] == 999)
    dataset[1]
    assert len(reads) == 1


@pytest.mark.parametrize('kind', ['order', 'hash', 'nonfinite'])
def test_dataset_rejects_manifest_scaler_mismatch_or_corrupt_cache(tmp_path, kind):
    frame = temporal(100)
    index = build_window_index(frame, split={}, config={})
    scaler = fit_scaler(frame, FEATURE_NAMES)
    if kind == 'nonfinite':
        frame.at[4, 'values'][0] = np.nan
    derived = manifest(tmp_path, frame)
    if kind == 'order': derived['feature_names'].reverse()
    if kind == 'hash':
        from src.datasets.normalization import scaler_hash
        scaler['derived_hash'] = 'different'
        scaler['hash'] = scaler_hash(scaler)
    with pytest.raises(ValueError):
        dataset = SequenceDataset(index, derived_manifest=derived, scaler=scaler,
                                  feature_names=FEATURE_NAMES)
        dataset[0]


def test_future_rows_do_not_change_past_index_or_tensors(tmp_path):
    frame = temporal(120)
    scaler = fit_scaler(frame, FEATURE_NAMES)
    first_manifest = manifest(tmp_path, frame)
    first_index = build_window_index(frame, split={}, config={})
    before = SequenceDataset(first_index, derived_manifest=first_manifest,
                             scaler=scaler, feature_names=FEATURE_NAMES)[0][0]
    for row in range(100, 120):
        frame.at[row, 'values'][:10] += 500
    second_manifest = manifest(tmp_path, frame)
    second_index = build_window_index(frame, split={}, config={})
    after = SequenceDataset(second_index, derived_manifest=second_manifest,
                            scaler=scaler, feature_names=FEATURE_NAMES)[0][0]
    np.testing.assert_array_equal(before, after)
    pd.testing.assert_series_equal(first_index.iloc[0], second_index.iloc[0])


def test_train_union_never_includes_validation_video(tmp_path):
    frame = temporal(100)
    index = build_window_index(frame, split={}, config={})
    derived = manifest(tmp_path, frame)
    validation = index.copy()
    validation['split'] = 'validation'
    assert unique_train_timesteps(validation, derived).empty
