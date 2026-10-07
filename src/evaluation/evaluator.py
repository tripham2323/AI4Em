"""Phase11 accepted-window metrics; no subject/video aggregation or bootstrap.

Rows are accepted predictions only, in fixed probability order [0, 1, 2].
Abstentions are counted in coverage, never encoded as an invented class. Metric
values are numbers or null; dotted keys in undefined_reasons explain nulls.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


CLASS_ORDER = (0, 1, 2)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Metadata/coverage dictionary keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("Metadata/coverage must contain finite JSON-safe values")


class ModelEvaluator:
    """Evaluate raw classifier probabilities with explicit support and coverage."""

    def evaluate(self, labels, probabilities, *, metadata, coverage) -> dict:
        if not isinstance(metadata, Mapping) or not isinstance(coverage, Mapping):
            raise ValueError("metadata and coverage must be mappings")
        metadata = _json_safe(metadata)
        coverage = _json_safe(coverage)
        if metadata.get("class_order", list(CLASS_ORDER)) != list(CLASS_ORDER):
            raise ValueError("Probability class order must be [0, 1, 2]")
        try:
            labels = np.asarray(labels, dtype=np.float64)
            probabilities = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Labels/probabilities must be numeric") from exc
        if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, CLASS_ORDER).all():
            raise ValueError("Labels must be one-dimensional class IDs 0/1/2")
        n = len(labels)
        if probabilities.shape != (n, 3):
            raise ValueError("Probabilities must have shape [N, 3] in class order 0/1/2")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Probabilities must be finite and between zero and one")
        if not np.allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1e-6):
            raise ValueError("Probability rows must sum to one within 1e-6")
        for key in ("scheduled", "accepted", "rejected"):
            count = coverage.get(key)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"Coverage {key} must be a nonnegative integer")
        if coverage["accepted"] != n or coverage["scheduled"] != n + coverage["rejected"]:
            raise ValueError("Coverage must match accepted rows and scheduled=accepted+rejected")
        reasons = {}

        def ratio(numerator, denominator, key, reason):
            if denominator == 0:
                reasons[key] = reason
                return None
            return float(numerator / denominator)

        coverage["acceptance_rate"] = ratio(n, coverage["scheduled"], "coverage.acceptance_rate", "No scheduled rows")
        confusion = np.zeros((3, 3), dtype=np.int64)
        if n:
            predictions = probabilities.argmax(axis=1)
            np.add.at(confusion, (labels.astype(np.int64), predictions), 1)
        supports = confusion.sum(axis=1)
        predicted_supports = confusion.sum(axis=0)
        per_class = []
        for cls in CLASS_ORDER:
            tp = int(confusion[cls, cls])
            true_count = int(supports[cls])
            predicted_count = int(predicted_supports[cls])
            per_class.append({
                "class_id": cls,
                "precision": ratio(tp, predicted_count, f"per_class.{cls}.precision", "No predicted rows for class"),
                "recall": ratio(tp, true_count, f"per_class.{cls}.recall", "No true rows for class"),
                "f1": ratio(2 * tp, true_count + predicted_count, f"per_class.{cls}.f1", "No true or predicted rows for class"),
                "support": true_count,
                "predicted_support": predicted_count,
            })
        supported = [item for item in per_class if item["support"] > 0]
        supported_count = len(supported)
        accuracy = ratio(int(np.trace(confusion)), n, "accuracy", "No accepted rows")
        supported_macro = ratio(sum(item["f1"] for item in supported), supported_count,
                                "supported_macro_f1", "No true class support")
        balanced = ratio(sum(item["recall"] for item in supported), supported_count,
                         "balanced_accuracy", "No true class support")
        weighted = ratio(sum(item["f1"] * item["support"] for item in supported), n,
                         "weighted_f1", "No accepted rows")
        macro = supported_macro if supported_count == 3 else None
        if macro is None:
            reasons["macro_f1"] = "Full three-class macro F1 requires true support for every class"
        return {
            "class_order": list(CLASS_ORDER), "support": n,
            "accuracy": accuracy, "macro_f1": macro, "weighted_f1": weighted,
            "balanced_accuracy": balanced, "balanced_accuracy_denominator": supported_count,
            "supported_macro_f1": supported_macro,
            "supported_macro_f1_denominator": supported_count,
            "per_class": per_class, "confusion_matrix": confusion.tolist(),
            "coverage": coverage, "metadata": metadata, "undefined_reasons": reasons,
        }
