"""Phase 21 external (NTHU-DDD) binary evaluation of a frozen UTA P0 model.

Binary risk proxy only: p_risk = p_low + p_drowsy against the source binary labels. Low Vigilance is never
created or reported for NTHU. If access is not attested, a BLOCKED report is written and no score exists.

Inputs
  --predictions  .csv/.parquet: scenario, video_id, risk_label (0/1), p_alert, p_low, p_drowsy [, status|accepted, subject_id]
  --policy       JSON frozen on UTA: threshold, threshold_source="uta_validation", calibration_mode="P0",
                 checkpoint_sha256, scaler_sha256, source_modality
  --access       JSON: dataset="NTHU-DDD", agreement_signed, official_partitions, labels_accessible, annotation_source
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import evaluator as ev

SCHEMA_VERSION = "external_nthu_v1"
REQUIRED_COLUMNS = ["scenario", "video_id", "risk_label", "p_alert", "p_low", "p_drowsy"]
POLICY_KEYS = ("threshold", "threshold_source", "calibration_mode", "checkpoint_sha256", "scaler_sha256",
               "source_modality")
EXIT_BLOCKED = 3


class ExternalInputError(ValueError):
    """Bad config/schema; exit 2."""


def check_access(access: dict | None) -> str | None:
    """Return a blocking reason, or None when legal access to labelled official partitions is attested."""
    if not access:
        return "no access attestation supplied"
    if access.get("dataset") != "NTHU-DDD":
        return "access file is not for NTHU-DDD"
    for key in ("agreement_signed", "official_partitions", "labels_accessible"):
        if access.get(key) is not True:
            return f"access attestation: {key} is not true"
    if not access.get("annotation_source"):
        return "access attestation: annotation_source missing"
    return None


def validate_policy(policy: dict) -> None:
    missing = [k for k in POLICY_KEYS if k not in policy]
    if missing:
        raise ExternalInputError(f"policy missing keys {missing}")
    if policy["calibration_mode"] != "P0":
        raise ExternalInputError("external protocol is frozen P0; NTHU labels must not be used for calibration")
    if policy["threshold_source"] != "uta_validation":
        raise ExternalInputError("threshold must be chosen on UTA validation, not on NTHU")
    t = policy["threshold"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1:
        raise ExternalInputError("threshold must be a number in [0, 1]")


def _binary_metrics(y: np.ndarray, p_risk: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Binary Macro F1 (classes present only), risk recall/precision, PR-AUC; undefined -> None."""
    n = len(y)
    out: dict[str, Any] = {"n_accepted": n, "n_risk": int((y == 1).sum()), "n_normal": int((y == 0).sum()),
                           "macro_f1": None, "risk_recall": None, "risk_precision": None,
                           "pr_auc": None, "risk_prevalence": None}
    if n == 0:
        return out
    tp, fp = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tn = int(((y == 1) & (pred == 0)).sum()), int(((y == 0) & (pred == 0)).sum())
    f1s = []
    if out["n_risk"]:
        f1s.append(2 * tp / (2 * tp + fp + fn))
        out["risk_recall"] = tp / (tp + fn)
    if out["n_normal"]:
        f1s.append(2 * tn / (2 * tn + fn + fp))
    out["macro_f1"] = float(np.mean(f1s))
    out["risk_precision"] = tp / (tp + fp) if tp + fp else None
    out["risk_prevalence"] = out["n_risk"] / n
    if out["n_risk"] and out["n_normal"]:  # AP is meaningless with one class: leave undefined
        from sklearn.metrics import average_precision_score
        out["pr_auc"] = float(average_precision_score(y, p_risk))
    out["confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return out


def evaluate_external(df: pd.DataFrame, policy: dict, *, expected_scenarios: int = 5) -> tuple[dict[str, Any], pd.DataFrame]:
    validate_policy(policy)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ExternalInputError(f"predictions missing columns {missing}; required {REQUIRED_COLUMNS}")
    if not df["risk_label"].isin([0, 1]).all():
        raise ExternalInputError("risk_label must be binary 0/1; Low Vigilance does not exist for NTHU")
    status = (df["status"].tolist() if "status" in df.columns else
              ["accepted" if a else "rejected" for a in df["accepted"]] if "accepted" in df.columns
              else ["accepted"] * len(df))
    if not set(status) <= set(ev.ROW_STATUSES):
        raise ExternalInputError(f"status must be in {ev.ROW_STATUSES}")
    work = df.assign(status=status)
    acc = work["status"] == ev.ACCEPTED
    probs = work[["p_alert", "p_low", "p_drowsy"]].to_numpy(dtype=float)
    if acc.any() and (not np.isfinite(probs[acc.to_numpy()]).all() or
                      not np.allclose(probs[acc.to_numpy()].sum(axis=1), 1, atol=1e-3)):
        raise ExternalInputError("accepted rows need finite probabilities summing to 1")
    work = work.assign(p_risk=probs[:, 1] + probs[:, 2])
    work = work.assign(pred_risk=(work["p_risk"] >= policy["threshold"]).astype(int))

    def scenario_block(part: pd.DataFrame) -> dict[str, Any]:
        a = part[part["status"] == ev.ACCEPTED]
        counts = {s: int((part["status"] == s).sum()) for s in ev.ROW_STATUSES}
        sched = len(part)
        return {"counts": {"scheduled": sched, **counts}, "coverage": ev._num(counts["accepted"] / sched) if sched else None,
                "risk_rows_abstained": int(((part["status"] == ev.REJECTED) & (part["risk_label"] == 1)).sum()),
                "metrics": _binary_metrics(a["risk_label"].to_numpy(), a["p_risk"].to_numpy(), a["pred_risk"].to_numpy())}

    scenarios = {str(k): scenario_block(g) for k, g in work.groupby("scenario", sort=True)}
    limits = []
    if len(scenarios) < expected_scenarios:
        limits.append(f"only {len(scenarios)} of {expected_scenarios} expected scenarios provided; "
                      "missing scenarios were not scored and nothing was substituted")
    for name, s in scenarios.items():
        if s["metrics"]["pr_auc"] is None:
            limits.append(f"scenario {name}: PR-AUC undefined (needs both risk and normal accepted rows)")
    return ev.clean_for_json({
        "schema_version": SCHEMA_VERSION, "status": "ok", "task": "binary_risk_proxy",
        "risk_definition": "p_risk = p_low + p_drowsy", "threshold": policy["threshold"],
        "policy": {k: policy[k] for k in POLICY_KEYS}, "scenarios": scenarios,
        "overall_pooled": scenario_block(work), "expected_scenarios": expected_scenarios,
        "n_scenarios_provided": len(scenarios), "limitations": limits,
        "domain_shift_note": "frozen UTA RGB-trained P0 model applied to NTHU data; modality differs",
        "not_reported": ["low_vigilance_recall", "three_class_metrics"],
    }), work


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--access", type=Path, help="access attestation JSON (without it the run is BLOCKED)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--target-modality", required=True, help='e.g. "NTHU-DDD infrared" (domain-shift note)')
    p.add_argument("--expected-scenarios", type=int, default=5)
    return p


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise ExternalInputError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExternalInputError(f"{what} is not valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    command = " ".join(shlex.quote(a) for a in ["python", "-m", "scripts.external_test", *argv])
    try:
        policy = _read_json(args.policy, "policy")
        validate_policy(policy)
        access = _read_json(args.access, "access") if args.access else None
        args.run_dir.mkdir(parents=True, exist_ok=True)
        reason = check_access(access)
        if reason:  # blocked: no scores, UTA results are unaffected
            report = {"schema_version": SCHEMA_VERSION, "status": "blocked", "reason": reason,
                      "note": "external generalization NOT complete; no NTHU score was produced or substituted"}
            (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (args.run_dir / "command.log").write_text(command + "\n", encoding="utf-8")
            print(f"external_test: BLOCKED: {reason}", file=sys.stderr)
            return EXIT_BLOCKED
        if not args.predictions.is_file():
            raise ExternalInputError(f"predictions not found: {args.predictions}")
        df = pd.read_parquet(args.predictions) if args.predictions.suffix == ".parquet" else pd.read_csv(args.predictions)
        report, work = evaluate_external(df, policy, expected_scenarios=args.expected_scenarios)
        report["target_modality"] = args.target_modality
        report["access"] = access
        report["provenance"] = {"predictions_sha256": ev.sha256_file(args.predictions), "command": command}
        (args.run_dir / "metrics.json").write_text(json.dumps(ev.clean_for_json(report), indent=2, sort_keys=True,
                                                              allow_nan=False), encoding="utf-8")
        work.to_csv(args.run_dir / "external_predictions.csv", index=False)
        (args.run_dir / "command.log").write_text(command + "\n", encoding="utf-8")
    except (ExternalInputError, ValueError, ImportError) as exc:
        print(f"external_test: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run_dir": str(args.run_dir), "scenarios": report["n_scenarios_provided"],
                      "limitations": report["limitations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
