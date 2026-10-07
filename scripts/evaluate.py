"""Evaluate raw model predictions (rules/RF/LSTM) per outer fold and write a run directory.

Input is a predictions table (.parquet or .csv) produced by the model owner, one row per window:
  required: subject_id, video_id, label (0/1/2), p_alert, p_low, p_drowsy
  optional: fold, status (accepted|rejected|blocked) or accepted (bool), window_start_ms
Probability columns map to class order [0, 1, 2] = [alert, low_vigilance, drowsy].
Output: <run-dir>/metrics.json, predictions.parquet, confusion_matrix.png, command.log, provenance.json
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluator as ev

PROB_COLUMNS = ["p_alert", "p_low", "p_drowsy"]
REQUIRED = ["subject_id", "video_id", "label", *PROB_COLUMNS]


class InputError(ValueError):
    """Bad config/schema; reported without a traceback and exit code 2."""


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise InputError(f"predictions file not found: {path}")
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise InputError("reading parquet needs pyarrow (pip install -r requirements.txt)") from exc
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise InputError("predictions must be .parquet or .csv")


def _read_json(path: Path | None, what: str):
    if path is None:
        return None
    if not path.is_file():
        raise InputError(f"{what} file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{what} is not valid JSON: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True, help="predictions .parquet/.csv")
    p.add_argument("--run-dir", type=Path, required=True, help="output directory, e.g. runs/<id>")
    p.add_argument("--min-video-coverage", type=float, default=0.5,
                   help="min accepted/eligible windows for a video prediction; below it the video abstains (default: 0.5)")
    p.add_argument("--n-boot", type=int, default=1000, help="subject-cluster bootstrap draws")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--model-id", default=None)
    p.add_argument("--splits", type=Path, help='JSON {"<fold>": {"train": [...], "val": [...], "test": [...]}}')
    p.add_argument("--expected-subjects", type=Path, help="JSON list of all outer subjects (each tested once)")
    p.add_argument("--calibration-prefix", type=Path, help='JSON {"<subject>": prefix_end_ms}; needs window_start_ms')
    p.add_argument("--provenance", type=Path, help="JSON with snapshot/split/model/config hashes to embed")
    return p


def run(args: argparse.Namespace, argv: list[str]) -> dict:
    if not 0 <= args.min_video_coverage <= 1:
        raise InputError("--min-video-coverage must be in [0, 1]")
    df = _read_table(args.predictions)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"predictions missing columns {missing}; required {REQUIRED}")
    if df.empty:
        raise InputError("predictions table is empty")
    if "fold" not in df.columns:
        if args.expected_folds > 1:
            raise InputError(
                "predictions must contain a 'fold' column when evaluating "
                "multiple outer folds"
            )
        df = df.assign(fold=1)
    splits = _read_json(args.splits, "splits")
    prefix = _read_json(args.calibration_prefix, "calibration prefix")
    expected = _read_json(args.expected_subjects, "expected subjects")
    split_check = ev.verify_outer_split(splits, expected) if splits is not None else None
    if splits is not None and expected is None:
        raise InputError(
            "--splits requires --expected-subjects"
        )
    if prefix is not None and "window_start_ms" not in df.columns:
        raise InputError("--calibration-prefix needs a window_start_ms column")
    reports, pred_frames = [], []
    for fold, part in df.groupby("fold", sort=True):
        test_subjects = None
        if splits is not None:
            key = str(fold)
            if key not in splits:
                raise InputError(f"fold {fold} not present in --splits")
            test_subjects = splits[key]["test"]
        probs = part[PROB_COLUMNS].to_numpy(dtype=float)
        status = part["status"].tolist() if "status" in part.columns else None
        accepted = part["accepted"].to_numpy(dtype=bool) if "accepted" in part.columns and status is None else None
        report = ev.evaluate(
            part["label"].to_numpy(), probs, subject_ids=part["subject_id"].tolist(),
            video_ids=part["video_id"].tolist(), status=status, accepted=accepted, fold=fold,
            min_video_coverage=args.min_video_coverage, n_boot=args.n_boot, seed=args.seed,
            confidence=args.confidence, test_subjects=test_subjects, model_id=args.model_id,
            window_start_ms=part["window_start_ms"].tolist() if prefix is not None else None,
            calibration_prefix_end_ms=prefix)
        reports.append(report)
        with np.errstate(invalid="ignore"):
            raw = np.where(np.isfinite(probs).all(axis=1), np.nanargmax(np.nan_to_num(probs, nan=-1), axis=1), -1)
        pred_frames.append(part.assign(raw_pred=raw))
    aggregate = ev.aggregate_folds(reports, expected_folds=args.expected_folds)
    pooled = aggregate["window"]["pooled"]
    provenance = {
        "schema_version": ev.SCHEMA_VERSION, "command": " ".join(shlex.quote(a) for a in argv),
        "predictions_sha256": ev.sha256_file(args.predictions), "seed": args.seed, "n_boot": args.n_boot,
        "min_video_coverage": args.min_video_coverage, "split_check": split_check,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "user_provenance": _read_json(args.provenance, "provenance") or {},
    }
    if args.splits:
        provenance["splits_sha256"] = ev.sha256_file(args.splits)
    full = {"schema_version": ev.SCHEMA_VERSION, "folds": {str(r["fold"]): r for r in reports},
            "aggregate": aggregate, "provenance": provenance}
    confusion_source = reports[0]["window"] if len(reports) == 1 else pooled
    try:
        ev.write_run_artifacts(args.run_dir, full, predictions=pd.concat(pred_frames, ignore_index=True),
                               command=provenance["command"], provenance=provenance,
                               confusion_report=confusion_source)
    except RuntimeError as exc:
        raise InputError(str(exc)) from exc
    return full


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    try:
        report = run(args, ["python", "-m", "scripts.evaluate", *argv])
    except (InputError, ValueError) as exc:
        print(f"evaluate: error: {exc}", file=sys.stderr)
        return 2
    agg = report["aggregate"]
    print(json.dumps({"run_dir": str(args.run_dir), "folds_reported": agg["n_folds_reported"],
                      "complete_five_fold": agg["complete_five_fold"],
                      "scope_limitations": agg["scope_limitations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
