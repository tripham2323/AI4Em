"""Behavioral regression tests for the train-only RF baseline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.baseline import BaselineClassifier


def training_rows():
    return pd.DataFrame({"signal": np.repeat([0., 10., 20.], 12),
                         "sometimes": np.tile([1., np.nan, 3.], 12),
                         "never": np.full(36, np.nan)}), np.repeat([0, 1, 2], 12)


def test_train_only_medians_and_every_column_missing_flags():
    x, y = training_rows()
    model = BaselineClassifier({}).fit(x, y)
    assert model.feature_names_ == ("signal", "sometimes", "never")
    np.testing.assert_array_equal(model.medians_, [10., 2., 0.])
    transformed = model.transform(pd.DataFrame({"signal": [np.nan], "sometimes": [999.], "never": [np.nan]}))
    np.testing.assert_array_equal(transformed, [[10., 999., 0., 1., 0., 1.]])
    model.predict_proba(pd.DataFrame({"signal": [1e9], "sometimes": [np.nan], "never": [12.]}))
    np.testing.assert_array_equal(model.medians_, [10., 2., 0.])
    assert model.classifier_.n_features_in_ == 6


def test_actual_rf_cold_reload_probabilities_and_class_order(tmp_path):
    x, y = training_rows()
    model = BaselineClassifier({"rf_n_jobs": 1}).fit(x, y, sample_weight=np.ones(len(y)))
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (36, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
    assert model.class_order_ == (0, 1, 2)
    assert model.classifier_.n_estimators == 300
    assert model.classifier_.max_depth == 12
    assert model.classifier_.min_samples_leaf == 5
    assert model.classifier_.class_weight == "balanced"
    assert model.classifier_.random_state == 42
    assert model.classifier_.n_jobs == 1
    path = model.save(tmp_path / "rf.joblib")
    with pytest.raises(ValueError, match="trusted"):
        BaselineClassifier.load(path)
    restored = BaselineClassifier.load(path, trusted=True)
    assert restored.feature_names_ == model.feature_names_
    assert restored.class_order_ == (0, 1, 2)
    np.testing.assert_array_equal(restored.predict_proba(x), probabilities)


def test_numpy_features_and_explicit_schema():
    x, y = training_rows()
    model = BaselineClassifier({"feature_names": ["a", "b", "c"]}).fit(x.to_numpy(), y)
    assert model.feature_names_ == ("a", "b", "c")
    assert model.predict_proba(x.to_numpy()).shape == (36, 3)


def test_requires_all_train_classes_and_strict_input_schema():
    x, y = training_rows()
    with pytest.raises(NotFittedError):
        BaselineClassifier({}).predict_proba(x)
    with pytest.raises(ValueError, match="three"):
        BaselineClassifier({}).fit(x.iloc[:24], y[:24])
    model = BaselineClassifier({}).fit(x, y)
    with pytest.raises(ValueError, match="order"):
        model.predict_proba(x[["never", "sometimes", "signal"]])
    with pytest.raises(ValueError):
        model.predict_proba(np.ones((2, 2)))
    with pytest.raises(ValueError):
        model.predict_proba(np.array([[np.inf, 1., 0.]]))
    with pytest.raises(ValueError):
        BaselineClassifier({}).fit(x, y + .25)
    with pytest.raises(ValueError):
        BaselineClassifier({}).fit(x, y, sample_weight=np.full(len(y), -1.))
    with pytest.raises(ValueError):
        BaselineClassifier({}).fit(x, y, sample_weight=np.zeros(len(y)))
    assert model.predict_proba(np.empty((0, 3))).shape == (0, 3)


def test_nullable_pandas_and_failed_refit_do_not_reuse_old_forest():
    x, y = training_rows()
    x["sometimes"] = x["sometimes"].astype("Float64")
    model = BaselineClassifier({}).fit(x, y)
    assert model.feature_names_ == ("signal", "sometimes", "never")
    assert np.isfinite(model.predict_proba(x)).all()
    with pytest.raises(ValueError):
        model.fit(np.ones((2, 2)), [0, 1])
    with pytest.raises(NotFittedError):
        model.predict_proba(x)
