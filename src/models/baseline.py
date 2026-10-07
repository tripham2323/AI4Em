"""Train-only summary imputation and the three-class random-forest baseline.

Serialized joblib files execute Python when loaded: load only locally trusted
artifacts, opting in explicitly with ``load(path, trusted=True)``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


class BaselineClassifier:
    """RF with one missingness flag per input column, including empty columns."""

    class_order_ = (0, 1, 2)

    def __init__(self, config: Mapping):
        self.config = dict(config)

    def _matrix(self, X, *, fitting: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            names = tuple(X.columns)
            if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
                raise ValueError("Feature names must be unique strings")
            if not fitting and names != self.feature_names_:
                raise ValueError("Feature names/order differ from training")
            X = X.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            names = None
        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric values or NaN") from exc
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("X must be a two-dimensional nonzero-width matrix")
        if np.isinf(values).any():
            raise ValueError("X must not contain infinite values")
        if fitting:
            if len(values) == 0:
                raise ValueError("Training rows must not be empty")
            if names is None:
                configured = self.config.get("feature_names")
                names = tuple(configured) if configured is not None else tuple(
                    f"feature_{i}" for i in range(values.shape[1]))
                if len(names) != values.shape[1] or len(set(names)) != len(names) or not all(
                    isinstance(name, str) for name in names):
                    raise ValueError("Configured feature names/order must match X columns")
            self.feature_names_ = names
        elif values.shape[1] != len(self.feature_names_):
            raise ValueError("Feature count differs from training")
        return values

    def _require_fitted(self) -> None:
        if not hasattr(self, "classifier_"):
            raise NotFittedError("BaselineClassifier must be fitted first")

    def fit(self, X, y, sample_weight=None) -> BaselineClassifier:
        # A failed refit must not expose an old forest under a new schema.
        if hasattr(self, "classifier_"):
            del self.classifier_
        values = self._matrix(X, fitting=True)
        try:
            labels = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Labels must be class IDs 0/1/2") from exc
        if labels.ndim != 1 or len(labels) != len(values) or not np.isfinite(labels).all() or not np.isin(labels, self.class_order_).all():
            raise ValueError("Labels must be one-dimensional class IDs 0/1/2 matching X")
        if set(labels.tolist()) != set(self.class_order_):
            raise ValueError("All three training classes 0/1/2 are required")
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.ndim != 1 or len(weights) != len(labels) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Sample weights must be finite, nonnegative and match X")
            if any(weights[labels == cls].sum() <= 0 for cls in self.class_order_):
                raise ValueError("All three classes require positive sample weight")
        requested_jobs = self.config.get("rf_n_jobs", min(4, os.cpu_count() or 1))
        if isinstance(requested_jobs, bool) or not isinstance(requested_jobs, int) or requested_jobs < 1:
            raise ValueError("rf_n_jobs must be a positive bounded integer")
        n_jobs = min(requested_jobs, 4, os.cpu_count() or 1)
        medians = np.empty(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            observed = values[:, column][~np.isnan(values[:, column])]
            medians[column] = float(np.median(observed)) if len(observed) else 0.
        missing = np.isnan(values)
        transformed = np.concatenate((np.where(missing, medians, values), missing.astype(np.float64)), axis=1)
        classifier = RandomForestClassifier(
            n_estimators=self.config.get("rf_n_estimators", 300),
            max_depth=self.config.get("rf_max_depth", 12),
            min_samples_leaf=self.config.get("rf_min_samples_leaf", 5),
            class_weight="balanced", random_state=self.config.get("seed", 42), n_jobs=n_jobs)
        classifier.fit(transformed, labels.astype(np.int64), sample_weight=weights)
        self.medians_ = medians
        self.classifier_ = classifier
        self.transformed_feature_names_ = self.feature_names_ + tuple(f"{name}__missing" for name in self.feature_names_)
        return self

    def transform(self, X) -> np.ndarray:
        """Apply stored train medians and flags without any fit on new rows."""
        self._require_fitted()
        values = self._matrix(X)
        missing = np.isnan(values)
        return np.concatenate((np.where(missing, self.medians_, values), missing.astype(np.float64)), axis=1)

    def predict_proba(self, X) -> np.ndarray:
        transformed = self.transform(X)
        if len(transformed) == 0:
            return np.empty((0, 3), dtype=np.float64)
        probabilities = self.classifier_.predict_proba(transformed)
        positions = [int(np.flatnonzero(self.classifier_.classes_ == cls)[0]) for cls in self.class_order_]
        return probabilities[:, positions]

    def save(self, path: str | Path) -> Path:
        """Persist the fitted RF, medians and ordered schema to a local artifact."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> BaselineClassifier:
        """Cold-load a trusted joblib artifact (never use untrusted uploads)."""
        if trusted is not True:
            raise ValueError("Loading joblib requires trusted=True for a trusted local path")
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise ValueError("Artifact is not a BaselineClassifier")
        model._require_fitted()
        if tuple(model.classifier_.classes_) != model.class_order_:
            raise ValueError("Artifact class order is not 0/1/2")
        if model.medians_.shape != (len(model.feature_names_),) or not np.isfinite(model.medians_).all() or model.classifier_.n_features_in_ != 2 * len(model.feature_names_):
            raise ValueError("Artifact imputation/schema dimensions are inconsistent")
        return model
