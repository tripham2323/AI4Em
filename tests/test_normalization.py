"""Train-only, unique-timestep normalization regressions."""
import numpy as np
import pandas as pd
import pytest

from src.datasets.normalization import FEATURE_NAMES, fit_scaler, transform_values


def rows():
    values = np.ones((3, 16), dtype=float)
    values[:, :10] = np.array([1., 3., 9.])[:, None]
    return pd.DataFrame(dict(video_id=['a', 'a', 'b'], segment_id=[0]*3,
        timestamp_ms=[0, 100, 0], split=['train']*3,
        values=list(values), validity=list(np.ones((3, 16), dtype=bool))))


def test_overlapping_windows_do_not_weight_duplicate_train_timesteps():
    frame = rows()
    scaler = fit_scaler(pd.concat([frame, frame.iloc[[0]]]), FEATURE_NAMES)
    assert scaler['counts'] == [3]*10
    assert scaler['means'] == pytest.approx([13/3]*10)
    assert scaler['scales'] == pytest.approx([np.std([1, 3, 9])]*10)


def test_conflicting_duplicate_identity_is_rejected():
    frame = rows()
    duplicate = frame.iloc[[0]].copy()
    duplicate['values'] = [np.zeros(16)]
    with pytest.raises(ValueError, match='conflict'):
        fit_scaler(pd.concat([frame, duplicate]), FEATURE_NAMES)


def test_validation_rows_cannot_enter_train_statistics():
    frame = rows()
    frame.loc[0, 'split'] = 'val'
    with pytest.raises(ValueError, match='train'):
        fit_scaler(frame, FEATURE_NAMES)


def test_zero_valid_observations_block_fit_and_constant_scale_is_one():
    frame = rows().iloc[[0]].copy()
    assert fit_scaler(frame, FEATURE_NAMES)['scales'] == [1.]*10
    validity = np.ones(16, bool)
    validity[4] = False
    frame['validity'] = [validity]
    with pytest.raises(ValueError, match='zero.*yaw_delta'):
        fit_scaler(frame, FEATURE_NAMES)


def test_imputation_happens_after_scaling_and_binary_values_are_not_scaled():
    scaler = fit_scaler(rows(), FEATURE_NAMES)
    values = np.ones((100, 16))
    validity = np.ones_like(values, bool)
    values[0, 0] = np.nan
    validity[0, 0] = False
    values[:, 14] = 0
    validity[:, 6] = False
    out = transform_values(values, validity, scaler)
    assert out.dtype == np.float32 and out.shape == (100, 16)
    assert out[0, 0] == 0 and np.isfinite(out).all()
    assert np.array_equal(out[:, 10:], values[:, 10:])


@pytest.mark.parametrize('kind', ['hash', 'order', 'nan', 'binary', 'validity'])
def test_transform_refuses_corrupt_schema_and_valid_values(kind):
    scaler = fit_scaler(rows(), FEATURE_NAMES)
    values, validity = np.ones((100, 16)), np.ones((100, 16), bool)
    if kind == 'hash': scaler['means'][0] += 1
    if kind == 'order': scaler['feature_names'].reverse()
    if kind == 'nan': values[0, 0] = np.inf
    if kind == 'binary': values[0, 12] = .5
    if kind == 'validity': validity[0, 12] = False
    with pytest.raises(ValueError):
        transform_values(values, validity, scaler)
