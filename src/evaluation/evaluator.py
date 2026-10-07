"""Shared evaluator for rules / RF / LSTM raw predictions.

Design rules (docs/10_Evaluation_Strategy.md):
* Classifier metrics use raw predictions only; realtime smoothing is evaluated elsewhere.
* Class order is fixed to ``[0, 1, 2]`` = ``[ALERT, LOW_VIGILANCE, DROWSY]``.
* Undefined metrics are ``None`` (JSON ``null``), never a fake ``0``.
* Macro metrics average only classes with ``support > 0`` in the evaluated rows, and the
  number of such classes is always reported next to the metric.
* Rows are ``accepted`` / ``rejected`` / ``blocked``; coverage is reported beside every metric.
* Uncertainty uses a cluster bootstrap over subjects, never over overlapping windows.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "evaluation_v1"
CLASS_ORDER: tuple[int, ...] = (0, 1, 2)
CLASS_NAMES: tuple[str, ...] = ("alert", "low_vigilance", "drowsy")
N_CLASSES = len(CLASS_ORDER)
ACCEPTED, REJECTED, BLOCKED = "accepted", "rejected", "blocked"
ROW_STATUSES = (ACCEPTED, REJECTED, BLOCKED)
HEADLINE_METRICS = (
    "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
    "low_vigilance_recall", "drowsy_recall", "low_vigilance_precision",
)


# --------------------------------------------------------------------------- helpers
def _num(value: Any) -> float | None:
    """Python float or None; NaN/inf become None so JSON stays strict."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy types and NaN/inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _num(obj)
    return obj


def _labels(values: Sequence[Any], name: str, rows: np.ndarray | None = None) -> np.ndarray:
    """Validate integer class ids in ``CLASS_ORDER``; ``rows`` limits validation to a mask."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric class ids")
    out = np.full(arr.shape, -1, dtype=np.int64)
    check = np.ones(arr.shape, bool) if rows is None else rows
    sel = arr[check].astype(float)
    if not np.all(np.isfinite(sel)) or not np.all(sel == np.round(sel)):
        raise ValueError(f"{name} must contain finite integer class ids")
    if not np.all(np.isin(sel, CLASS_ORDER)):
        raise ValueError(f"{name} must only contain class ids {list(CLASS_ORDER)}")
    out[check] = sel.astype(np.int64)
    return out


def _resolve_status(n: int, accepted: Sequence[Any] | None, status: Sequence[Any] | None) -> np.ndarray:
    if status is not None:
        arr = np.asarray(list(status), dtype=object)
        if arr.shape != (n,):
            raise ValueError("status length must match labels")
        if not all(s in ROW_STATUSES for s in arr):
            raise ValueError(f"status values must be in {ROW_STATUSES}")
        if accepted is not None:
            mask = np.asarray(accepted, dtype=bool)
            if mask.shape != (n,) or not np.array_equal(mask, arr == ACCEPTED):
                raise ValueError("accepted mask contradicts status")
        return arr
    if accepted is None:
        return np.full(n, ACCEPTED, dtype=object)
    mask = np.asarray(accepted)
    if mask.dtype != bool or mask.shape != (n,):
        raise ValueError("accepted must be a boolean mask matching labels")
    return np.where(mask, ACCEPTED, REJECTED).astype(object)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows = true class, columns = predicted class, fixed ``CLASS_ORDER`` size."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    if len(y_true):
        np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    """All classification metrics from a 3x3 count matrix (rows=true, cols=predicted).

    Per-class: precision undefined if nothing predicted as the class; recall and F1 undefined
    if the class has no true support. Macro/weighted/balanced average only supported classes.
    """
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (N_CLASSES, N_CLASSES) or (cm < 0).any():
        raise ValueError("confusion matrix must be a non-negative 3x3 count matrix")
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    n = int(cm.sum())
    per_class: dict[str, dict[str, Any]] = {}
    f1s: dict[int, float] = {}
    recalls: dict[int, float] = {}
    for k, name in enumerate(CLASS_NAMES):
        precision = tp[k] / predicted[k] if predicted[k] > 0 else None
        recall = tp[k] / support[k] if support[k] > 0 else None
        f1 = 2 * tp[k] / (support[k] + predicted[k]) if support[k] > 0 else None
        if recall is not None:
            recalls[k] = float(recall)
        if f1 is not None:
            f1s[k] = float(f1)
        per_class[name] = {"precision": _num(precision), "recall": _num(recall), "f1": _num(f1),
                           "support": int(support[k]), "predicted": int(predicted[k])}
    present = [k for k in range(N_CLASSES) if support[k] > 0]
    metrics: dict[str, Any] = {k: None for k in HEADLINE_METRICS}
    if n > 0:
        metrics["accuracy"] = float(tp.sum() / n)
        metrics["macro_f1"] = float(np.mean([f1s[k] for k in present]))
        metrics["weighted_f1"] = float(sum(support[k] * f1s[k] for k in present) / sum(support[k] for k in present))
        metrics["balanced_accuracy"] = float(np.mean([recalls[k] for k in present]))
    metrics["low_vigilance_recall"] = per_class["low_vigilance"]["recall"]
    metrics["drowsy_recall"] = per_class["drowsy"]["recall"]
    metrics["low_vigilance_precision"] = per_class["low_vigilance"]["precision"]
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = [[_num(cm[i, j] / support[i]) if support[i] > 0 else None for j in range(N_CLASSES)]
                      for i in range(N_CLASSES)]
    return {
        "n": n, "metrics": metrics, "per_class": per_class,
        "classes_present": [CLASS_NAMES[k] for k in present],
        "n_classes_present": len(present),
        "full_class_coverage": len(present) == N_CLASSES,
        "confusion_matrix": {"labels": list(CLASS_ORDER), "class_names": list(CLASS_NAMES),
                             "counts": cm.tolist(), "normalized_by_true": normalized},
    }


# --------------------------------------------------------------------------- window level
def evaluate_classification(y_true: Sequence[int], y_pred: Sequence[int], *,
                            accepted: Sequence[bool] | None = None,
                            status: Sequence[str] | None = None) -> dict[str, Any]:
    """Metrics on accepted rows with explicit eligible/accepted/rejected/blocked counts.

    ``y_pred`` entries of non-accepted rows are ignored (may be -1/NaN).
    coverage = accepted / scheduled (all rows); eligible_coverage = accepted / (scheduled - blocked).
    Classifier metrics are *conditional on accepted rows*; ``abstained_true_class`` counts the
    true risky timestamps that were not predicted.
    """
    st = _resolve_status(len(np.asarray(y_true)), accepted, status)
    yt = _labels(y_true, "y_true")
    acc = st == ACCEPTED
    yp = _labels(y_pred, "y_pred", rows=acc)
    cm = confusion_counts(yt[acc], yp[acc])
    report = metrics_from_confusion(cm)
    scheduled = len(yt)
    blocked = int((st == BLOCKED).sum())
    n_acc = int(acc.sum())
    by_class = {}
    for k, name in enumerate(CLASS_NAMES):
        m = yt == k
        by_class[name] = {"scheduled": int(m.sum()), "blocked": int((m & (st == BLOCKED)).sum()),
                          "eligible": int((m & (st != BLOCKED)).sum()),
                          "accepted": int((m & acc).sum()), "rejected": int((m & (st == REJECTED)).sum())}
    report.update({
        "counts": {"scheduled": scheduled, "blocked": blocked, "eligible": scheduled - blocked,
                   "accepted": n_acc, "rejected": int((st == REJECTED).sum())},
        "by_class_counts": by_class,
        "coverage": _num(n_acc / scheduled) if scheduled else None,
        "eligible_coverage": _num(n_acc / (scheduled - blocked)) if scheduled - blocked else None,
        "abstained_true_class": {name: by_class[name]["rejected"] for name in CLASS_NAMES},
        "defined": n_acc > 0,
    })
    return report


# --------------------------------------------------------------------------- subject level
def _subject_confusions(subject_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                        accepted: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s in sorted({str(x) for x in subject_ids}):
        m = (subject_ids == s) & accepted
        out[s] = confusion_counts(y_true[m], y_pred[m])
    return out


def evaluate_subjects(subject_ids: Sequence[Any], y_true: Sequence[int], y_pred: Sequence[int], *,
                      status: Sequence[str]) -> dict[str, Any]:
    """Per-subject metrics and the mean over subjects where each metric is defined."""
    sid = np.asarray([str(s) for s in subject_ids], dtype=object)
    st = np.asarray(status, dtype=object)
    acc = st == ACCEPTED
    yt, yp = _labels(y_true, "y_true"), _labels(y_pred, "y_pred", rows=acc)
    per_subject: dict[str, Any] = {}
    for s in sorted(set(sid)):
        m = sid == s
        rep = evaluate_classification(yt[m], yp[m], status=st[m])
        per_subject[s] = {"metrics": rep["metrics"], "counts": rep["counts"], "coverage": rep["coverage"],
                          "classes_present": rep["classes_present"]}
    mean: dict[str, Any] = {}
    for key in HEADLINE_METRICS:
        vals = [v["metrics"][key] for v in per_subject.values() if v["metrics"][key] is not None]
        mean[key] = {"mean": _num(np.mean(vals)) if vals else None, "n_subjects_defined": len(vals)}
    return {"n_subjects": len(per_subject), "per_subject": per_subject, "mean_over_subjects": mean}


def bootstrap_subject_ci(subject_ids: Sequence[Any], y_true: Sequence[int], y_pred: Sequence[int], *,
                         status: Sequence[str], metrics: Sequence[str] = ("macro_f1", "balanced_accuracy",
                         "low_vigilance_recall", "drowsy_recall"), n_boot: int = 1000, seed: int = 0,
                         confidence: float = 0.95) -> dict[str, Any]:
    """Cluster (subject) bootstrap of pooled-confusion metrics.

    Subjects are resampled with replacement, their accepted windows are pooled, and the metric is
    recomputed. Draws where the metric is undefined are dropped and counted (``n_boot_valid``).
    """
    if n_boot < 1 or not 0 < confidence < 1:
        raise ValueError("n_boot must be >= 1 and confidence in (0, 1)")
    sid = np.asarray([str(s) for s in subject_ids], dtype=object)
    st = np.asarray(status, dtype=object)
    acc = st == ACCEPTED
    yt, yp = _labels(y_true, "y_true"), _labels(y_pred, "y_pred", rows=acc)
    per_subject = _subject_confusions(sid, yt, yp, acc)
    names = sorted(per_subject)
    stack = np.stack([per_subject[s] for s in names]) if names else np.zeros((0, N_CLASSES, N_CLASSES), np.int64)
    point = metrics_from_confusion(stack.sum(axis=0))["metrics"] if len(names) else {}
    result: dict[str, Any] = {"method": "cluster_bootstrap_by_subject", "n_subjects": len(names),
                              "n_boot": n_boot, "seed": seed, "confidence": confidence, "metrics": {}}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, max(len(names), 1), size=(n_boot, max(len(names), 1)))
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    if len(names) >= 2:
        for row in draws:
            sampled = metrics_from_confusion(stack[row].sum(axis=0))["metrics"]
            for m in metrics:
                if sampled[m] is not None:
                    samples[m].append(sampled[m])
    lo_q, hi_q = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    for m in metrics:
        vals = samples[m]
        result["metrics"][m] = {
            "estimate": _num(point.get(m)),
            "ci_low": _num(np.percentile(vals, lo_q)) if vals else None,
            "ci_high": _num(np.percentile(vals, hi_q)) if vals else None,
            "n_boot_valid": len(vals),
        }
    if len(names) < 2:
        result["note"] = "fewer than two subjects: confidence interval undefined"
    return result


def paired_subject_difference(metric_a: Mapping[str, float | None], metric_b: Mapping[str, float | None], *,
                              n_boot: int = 1000, seed: int = 0, confidence: float = 0.95) -> dict[str, Any]:
    """Paired per-subject gain ``b - a`` over subjects defined for both, with bootstrap CI of the mean."""
    common = sorted(s for s in set(metric_a) & set(metric_b)
                    if metric_a[s] is not None and metric_b[s] is not None)
    diffs = np.array([float(metric_b[s]) - float(metric_a[s]) for s in common])
    out: dict[str, Any] = {"n_subjects": len(common), "subjects": common,
                           "n_gain": int((diffs > 0).sum()), "n_loss": int((diffs < 0).sum()),
                           "n_tie": int((diffs == 0).sum()), "mean_difference": None,
                           "ci_low": None, "ci_high": None, "confidence": confidence, "seed": seed}
    if len(common) == 0:
        return out
    out["mean_difference"] = _num(diffs.mean())
    if len(common) >= 2:
        rng = np.random.default_rng(seed)
        means = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
        out["ci_low"] = _num(np.percentile(means, (1 - confidence) / 2 * 100))
        out["ci_high"] = _num(np.percentile(means, (1 + confidence) / 2 * 100))
    else:
        out["note"] = "fewer than two paired subjects: confidence interval undefined"
    return out


# --------------------------------------------------------------------------- video level
def aggregate_videos(video_ids: Sequence[Any], subject_ids: Sequence[Any], y_true: Sequence[int],
                     probabilities: np.ndarray, status: Sequence[str], *, min_video_coverage: float) -> dict[str, Any]:
    """Mean probability of accepted windows, then argmax.

    A video abstains (never defaults to Alert) when it has no accepted window or
    ``accepted / eligible < min_video_coverage``; a video whose windows are all blocked is blocked.
    """
    if not 0 <= min_video_coverage <= 1:
        raise ValueError("min_video_coverage must be between 0 and 1")
    vid = np.asarray([str(v) for v in video_ids], dtype=object)
    sid = np.asarray([str(s) for s in subject_ids], dtype=object)
    yt = _labels(y_true, "y_true")
    st = np.asarray(status, dtype=object)
    probs = np.asarray(probabilities, dtype=float)
    videos: list[dict[str, Any]] = []
    for v in sorted(set(vid)):
        m = vid == v
        if len(set(yt[m])) != 1:
            raise ValueError(f"video {v} has inconsistent labels")
        if len(set(sid[m])) != 1:
            raise ValueError(f"video {v} belongs to more than one subject")
        eligible = int((m & (st != BLOCKED)).sum())
        accepted = m & (st == ACCEPTED)
        n_acc = int(accepted.sum())
        cov = n_acc / eligible if eligible else None
        row: dict[str, Any] = {"video_id": v, "subject_id": sid[m][0], "label": int(yt[m][0]),
                               "n_windows": int(m.sum()), "n_eligible": eligible, "n_accepted": n_acc,
                               "coverage": _num(cov), "pred": None, "mean_probabilities": None}
        if eligible == 0:
            row["status"] = BLOCKED
        elif n_acc == 0 or cov < min_video_coverage:
            row["status"] = REJECTED
        else:
            mean_p = probs[accepted].mean(axis=0)
            row.update(status=ACCEPTED, pred=int(np.argmax(mean_p)), mean_probabilities=[float(x) for x in mean_p])
        videos.append(row)
    report = evaluate_classification([r["label"] for r in videos],
                                     [r["pred"] if r["pred"] is not None else -1 for r in videos],
                                     status=[r["status"] for r in videos])
    return {"min_video_coverage": min_video_coverage, "videos": videos, "report": report}


# --------------------------------------------------------------------------- split checks
def verify_outer_split(folds: Mapping[Any, Mapping[str, Sequence[Any]]],
                       expected_subjects: Sequence[Any] | None = None) -> dict[str, Any]:
    """Each outer subject is tested exactly once and partitions never overlap inside a fold."""
    tested: dict[str, Any] = {}
    for fold, parts in folds.items():
        sets = {name: {str(s) for s in parts.get(name, [])} for name in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            both = sets[a] & sets[b]
            if both:
                raise ValueError(f"fold {fold}: subjects in both {a} and {b}: {sorted(both)}")
        for s in sets["test"]:
            if s in tested:
                raise ValueError(f"subject {s} is a test subject in folds {tested[s]} and {fold}")
            tested[s] = fold
    if expected_subjects is not None:
        expected = {str(s) for s in expected_subjects}
        if set(tested) != expected:
            raise ValueError(f"missing test subjects {sorted(expected - set(tested))}; "
                             f"unexpected {sorted(set(tested) - expected)}")
    return {"n_folds": len(folds), "n_test_subjects": len(tested), "tested_once": True}


def verify_rows_belong_to_test(subject_ids: Sequence[Any], test_subjects: Sequence[Any]) -> None:
    extra = {str(s) for s in subject_ids} - {str(s) for s in test_subjects}
    if extra:
        raise ValueError(f"evaluation rows contain non-test subjects: {sorted(extra)}")


def verify_calibration_prefix_excluded(subject_ids: Sequence[Any], window_start_ms: Sequence[float],
                                       status: Sequence[str], prefix_end_ms: Mapping[Any, float]) -> None:
    """Accepted windows must start at/after the reserved calibration prefix of their subject."""
    for s, start, st in zip(subject_ids, window_start_ms, status):
        end = prefix_end_ms.get(s, prefix_end_ms.get(str(s)))
        if st == ACCEPTED and end is not None and start < end:
            raise ValueError(f"accepted window of subject {s} starts at {start} inside calibration prefix (<{end})")


# --------------------------------------------------------------------------- full evaluate
def _validate_probabilities(probabilities: Any, n: int, accepted: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (n, N_CLASSES):
        raise ValueError(f"probabilities must have shape ({n}, {N_CLASSES}) in class order {list(CLASS_ORDER)}")
    sel = probs[accepted]
    if sel.size and (not np.all(np.isfinite(sel)) or (sel < -1e-6).any() or not np.allclose(sel.sum(axis=1), 1, atol=1e-3)):
        raise ValueError("accepted rows need finite probabilities that sum to 1")
    return probs


def evaluate(labels: Sequence[int], probabilities: Any, *, subject_ids: Sequence[Any], video_ids: Sequence[Any],
             status: Sequence[str] | None = None, accepted: Sequence[bool] | None = None,
             fold: Any = None, min_video_coverage: float, n_boot: int = 1000, seed: int = 0,
             confidence: float = 0.95, test_subjects: Sequence[Any] | None = None,
             window_start_ms: Sequence[float] | None = None,
             calibration_prefix_end_ms: Mapping[Any, float] | None = None,
             model_id: str | None = None, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Window, video and subject reports for one outer fold. Raw prediction = argmax(probabilities)."""
    n = len(np.asarray(labels))
    for name, seq in (("subject_ids", subject_ids), ("video_ids", video_ids)):
        if len(seq) != n:
            raise ValueError(f"{name} length must match labels")
    st = _resolve_status(n, accepted, status)
    acc = st == ACCEPTED
    probs = _validate_probabilities(probabilities, n, acc)
    if test_subjects is not None:
        verify_rows_belong_to_test(subject_ids, test_subjects)
    if calibration_prefix_end_ms is not None:
        if window_start_ms is None:
            raise ValueError("window_start_ms is required to verify the calibration prefix")
        verify_calibration_prefix_excluded(subject_ids, window_start_ms, st, calibration_prefix_end_ms)
    pred = np.full(n, -1, dtype=np.int64)
    if acc.any():
        pred[acc] = np.argmax(probs[acc], axis=1)
    window = evaluate_classification(labels, pred, status=st)
    video = aggregate_videos(video_ids, subject_ids, labels, probs, st, min_video_coverage=min_video_coverage)
    subject = evaluate_subjects(subject_ids, labels, pred, status=st)
    boot = bootstrap_subject_ci(subject_ids, labels, pred, status=st, n_boot=n_boot, seed=seed, confidence=confidence)
    notes = []
    if not window["defined"]:
        notes.append("no accepted windows: metrics undefined")
    elif not window["full_class_coverage"]:
        notes.append("classes absent from test support: " + ", ".join(
            n_ for n_ in CLASS_NAMES if n_ not in window["classes_present"]) + "; macro metrics average present classes only")
    if window["counts"]["blocked"]:
        notes.append(f"{window['counts']['blocked']} blocked windows excluded from metrics but kept in coverage")
    return clean_for_json({
        "schema_version": SCHEMA_VERSION, "fold": fold, "model_id": model_id,
        "status": "ok" if window["defined"] else "undefined_no_accepted_windows",
        "class_order": list(CLASS_ORDER), "class_names": list(CLASS_NAMES),
        "window": window, "video": video, "subject": subject, "bootstrap": boot,
        "notes": notes, "provenance": dict(provenance or {}),
    })


# --------------------------------------------------------------------------- aggregate folds
def _aggregate_level(reports: Sequence[Mapping[str, Any]], level: str) -> dict[str, Any]:
    def section(r: Mapping[str, Any]) -> Mapping[str, Any]:
        return r[level] if level == "window" else r["video"]["report"]

    valid = [r for r in reports if r["status"] == "ok" and section(r)["defined"]]
    stats = {}
    for key in HEADLINE_METRICS:
        vals = [section(r)["metrics"][key] for r in valid if section(r)["metrics"][key] is not None]
        stats[key] = {"mean": _num(np.mean(vals)) if vals else None,
                      "std": _num(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                      "n_folds_defined": len(vals)}
    pooled_cm = sum((np.asarray(section(r)["confusion_matrix"]["counts"], dtype=np.int64) for r in valid),
                    np.zeros((N_CLASSES, N_CLASSES), np.int64))
    pooled = metrics_from_confusion(pooled_cm)
    pooled["denominators"] = {
        k: int(sum(section(r)["counts"][k] for r in valid))
        for k in ("scheduled", "blocked", "eligible", "accepted", "rejected")}
    return {"n_folds_valid": len(valid), "std_ddof": 1, "mean_std": stats, "pooled": pooled}


def aggregate_folds(reports: Sequence[Mapping[str, Any]], *, expected_folds: int = 5) -> dict[str, Any]:
    """mean/std over valid folds (with explicit counts) plus pooled out-of-fold confusion.

    Folds with undefined metrics are listed, never averaged as 0. ``complete_five_fold`` is only
    true when all expected folds are valid *and* each contains all three classes.
    """
    ids = [r["fold"] for r in reports]
    if len(set(map(str, ids))) != len(ids):
        raise ValueError("duplicate fold ids in reports")
    seen: dict[str, Any] = {}
    for r in reports:
        for s in r["subject"]["per_subject"]:
            if s in seen:
                raise ValueError(f"subject {s} was tested in folds {seen[s]} and {r['fold']}")
            seen[s] = r["fold"]
    fold_table = [{"fold": r["fold"], "status": r["status"],
                   "counts": r["window"]["counts"], "coverage": r["window"]["coverage"],
                   "classes_present": r["window"]["classes_present"],
                   "full_class_coverage": r["window"]["full_class_coverage"],
                   "n_subjects": r["subject"]["n_subjects"], "notes": r["notes"]} for r in reports]
    reasons = []
    if len(reports) != expected_folds:
        reasons.append(f"{len(reports)} of {expected_folds} outer folds reported")
    for row in fold_table:
        if row["status"] != "ok":
            reasons.append(f"fold {row['fold']}: {row['status']}")
        elif not row["full_class_coverage"]:
            reasons.append(f"fold {row['fold']}: only classes {row['classes_present']}")
    return clean_for_json({
        "schema_version": SCHEMA_VERSION, "expected_folds": expected_folds, "n_folds_reported": len(reports),
        "complete_five_fold": not reasons and expected_folds == 5, "scope_limitations": reasons,
        "folds": fold_table, "window": _aggregate_level(reports, "window"),
        "video": _aggregate_level(reports, "video"),
    })


# --------------------------------------------------------------------------- artifacts
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_confusion_matrix(report: Mapping[str, Any], path: str | Path, title: str = "Confusion matrix") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = report["confusion_matrix"]
    counts = np.asarray(cm["counts"])
    norm = np.array([[np.nan if v is None else v for v in row] for row in cm["normalized_by_true"]], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(np.nan_to_num(norm), vmin=0, vmax=1, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            frac = "n/a" if np.isnan(norm[i, j]) else f"{norm[i, j]:.2f}"
            ax.text(j, i, f"{counts[i, j]}\n({frac})", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=20)
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_run_artifacts(run_dir: str | Path, report: Mapping[str, Any], *, predictions: Any = None,
                        command: str | None = None, provenance: Mapping[str, Any] | None = None,
                        confusion_report: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Write metrics.json, predictions.parquet, confusion_matrix.png, command.log, provenance.json."""
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    (run / "metrics.json").write_text(
        json.dumps(clean_for_json(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    written["metrics"] = str(run / "metrics.json")
    if predictions is not None:
        try:
            predictions.to_parquet(run / "predictions.parquet", index=False)
        except ImportError as exc:  # pyarrow/fastparquet missing: fail loudly, no silent CSV swap
            raise RuntimeError("writing predictions.parquet needs pyarrow (see requirements.txt)") from exc
        written["predictions"] = str(run / "predictions.parquet")
    if confusion_report is not None:
        plot_confusion_matrix(confusion_report, run / "confusion_matrix.png")
        written["confusion_matrix"] = str(run / "confusion_matrix.png")
    if command is not None:
        (run / "command.log").write_text(command.rstrip() + "\n", encoding="utf-8")
        written["command"] = str(run / "command.log")
    (run / "provenance.json").write_text(
        json.dumps(clean_for_json(dict(provenance or {})), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    written["provenance"] = str(run / "provenance.json")
    return written
