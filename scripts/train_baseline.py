"""Train the RF control for one outer fold.

Supports the original Phase-13 CLI and the Phase-15 ablation contract. RF_D
receives the same D signal groups as the LSTM control, represented by causal
window summaries (no identifiers are fed to the classifier).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.datasets.derived import atomic_json
from src.datasets.normalization import FEATURE_NAMES
from src.models.baseline import BaselineClassifier
from src.training.experiment import (
    BlockedExperiment, build_summaries, check_eligibility, hashes, prepare_run,
    project_path, save_predictions, validate_dataset,
)


def _find_manifest(mode: str, outer_index: int, explicit: Path | None) -> Path:
    if explicit:
        return project_path(explicit)
    report = PROJECT_ROOT / "runs" / "phase11" / "derived" / f"{mode}.json"
    if not report.is_file():
        raise FileNotFoundError(f"No derived manifest report found at {report}; run build_derived first")
    data = json.loads(report.read_text(encoding="utf-8"))
    for artifact in data.get("artifacts", []):
        if int(artifact.get("outer_index", -1)) == outer_index:
            path = Path(artifact["path"])
            if path.is_file():
                return path
    raise FileNotFoundError(f"No {mode} derived manifest for outer index {outer_index} in {report}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--derived-manifest", type=Path)
    p.add_argument("--outer-index", type=int, choices=range(5))
    p.add_argument("--mode", choices=("P0", "P1"))
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--experiment-id", choices=("RF_D",))
    p.add_argument("--feature-names", help="comma-separated ordered raw feature names")
    p.add_argument("--calibration-mode", choices=("P0", "P1"))
    p.add_argument("--seed", type=int)
    p.add_argument("--outer-fold", type=int, choices=range(1, 6))
    args = p.parse_args(argv)
    ablation = args.experiment_id is not None or args.outer_fold is not None or args.feature_names is not None
    if ablation:
        if args.experiment_id != "RF_D" or not args.feature_names or args.calibration_mode is None or args.seed is None or args.outer_fold is None:
            p.error("RF_D mode requires --experiment-id RF_D, --feature-names, --calibration-mode, --seed and --outer-fold")
        args.mode = args.calibration_mode
        args.outer_index = args.outer_fold - 1
        args.feature_names = tuple(x.strip() for x in args.feature_names.split(",") if x.strip())
    else:
        if args.derived_manifest is None or args.outer_index is None or args.mode is None:
            p.error("original mode requires --derived-manifest, --outer-index and --mode")
        args.feature_names = tuple(FEATURE_NAMES)
    return args


# Raw temporal signal -> causal summary columns. Each raw signal stays within
# its own group; flags are represented by ratios/current readiness.
def _summary_columns(raw_names: tuple[str, ...], available: tuple[str, ...]) -> list[str]:
    mapping = {
        "ear_left_norm": ["ear_left_norm_mean", "ear_left_norm_std", "ear_left_norm_min", "ear_left_norm_max", "ear_left_norm_slope"],
        "ear_right_norm": ["ear_right_norm_mean", "ear_right_norm_std", "ear_right_norm_min", "ear_right_norm_max", "ear_right_norm_slope"],
        "mar_delta": ["mar_delta_mean", "mar_delta_std", "mar_delta_min", "mar_delta_max", "mar_delta_slope"],
        "pitch_delta": ["pitch_delta_mean", "pitch_delta_std", "pitch_delta_min", "pitch_delta_max", "pitch_delta_slope"],
        "yaw_delta": ["yaw_delta_mean", "yaw_delta_std", "yaw_delta_min", "yaw_delta_max", "yaw_delta_slope"],
        "roll_delta": ["roll_delta_mean", "roll_delta_std", "roll_delta_min", "roll_delta_max", "roll_delta_slope"],
        "perclos_60": ["perclos_60"],
        "closure_elapsed_s": ["closure_duration_s", "closure_max_duration_s", "closure_mean_duration_s"],
        "yawn_elapsed_s": ["mouth_open_duration_s", "yawn_count", "yawn_mean_duration_s", "yawn_max_duration_s"],
        "pitch_velocity_dps": ["pitch_velocity_dps_mean", "pitch_velocity_dps_std", "pitch_velocity_dps_min", "pitch_velocity_dps_max", "pitch_velocity_dps_slope"],
        "left_eye_valid": ["eye_valid_ratio"],
        "right_eye_valid": ["eye_valid_ratio"],
        "mouth_valid": ["mouth_valid_ratio"],
        "pose_valid": ["pose_valid_ratio"],
        "perclos_ready": ["perclos_ready"],
        "calibration_valid": ["calibration_valid"],
    }
    result = []
    available_set = set(available)
    for raw in raw_names:
        candidates = mapping.get(raw, [])
        selected = [c for c in candidates if c in available_set]
        if not selected:
            raise ValueError(f"No summary representation available for RF feature {raw}")
        result.extend(selected)
    # Preserve first occurrence only; left/right validity intentionally share
    # one observable ratio and therefore must not duplicate the column.
    return list(dict.fromkeys(result))


def _p95_predict(model, X: pd.DataFrame) -> float | None:
    if X.empty:
        return None
    row = X.iloc[[0]]
    for _ in range(5):
        model.predict_proba(row)
    times = []
    for _ in range(30):
        t0 = perf_counter(); model.predict_proba(row); times.append((perf_counter() - t0) * 1000.0)
    return float(np.percentile(times, 95))


def main(argv=None) -> int:
    args = _parse_args(argv)
    started = perf_counter()
    args.derived_manifest = _find_manifest(args.mode, args.outer_index, args.derived_manifest)
    report = None
    try:
        config, run, report = prepare_run(args, "baseline")
        config["seed"] = int(args.seed if args.seed is not None else config.get("seed", 42))
        config["experiment_id"] = args.experiment_id or "RF_D"
        atomic_json(run / "resolved_config.json", config)
        manifest, index, split = validate_dataset(args.derived_manifest, mode=args.mode,
                                                   outer_index=args.outer_index,
                                                   config={**config, "feature_names": list(FEATURE_NAMES)})
        identities = hashes(manifest)
        report.update(experiment_id=args.experiment_id or "RF_D", seed=config["seed"],
                      feature_names=list(args.feature_names), calibration_mode=args.mode,
                      hashes=identities, coverage=manifest["coverage"], counts=manifest["class_support"],
                      dataset_manifest_hash=manifest["dataset_manifest_hash"])
        atomic_json(run / "split.json", split)
        atomic_json(run / "dataset_manifest.json", manifest)
        check_eligibility(manifest, index)

        table, all_summary_names = build_summaries(index, manifest)
        selected_summary_names = _summary_columns(tuple(args.feature_names), tuple(all_summary_names))
        table.to_parquet(run / "summaries.parquet", index=False)
        schema = dict(schema_version=manifest["schema_version"], mode=args.mode,
                      outer_index=args.outer_index, feature_names=list(args.feature_names),
                      summary_feature_names=selected_summary_names, class_order=[0, 1, 2],
                      identifier_columns=[name for name in table.columns if name not in all_summary_names],
                      hashes=identities, dataset_manifest_hash=manifest["dataset_manifest_hash"],
                      imputation="train_column_median_plus_missing_flag; all_missing_train_column=0",
                      timing=dict(sequence_fps=10, steps=100, tick_span_ms=9900, nominal_duration_ms=10000,
                                  summary_interval="[end_timestamp_ms-10000,end_timestamp_ms)"))
        atomic_json(run / "summary_schema.json", schema)

        train = table[table.split.eq("train")]
        model = BaselineClassifier({**config, "feature_names": selected_summary_names})
        model.fit(train.loc[:, selected_summary_names], train.label_id.to_numpy())
        model.save(run / "model.joblib")
        validation = table[table.split.eq("validation")]
        val_X = validation.loc[:, selected_summary_names]
        val_probabilities = model.predict_proba(val_X)
        val_metrics, val_path = save_predictions(run, "validation", validation, val_probabilities, manifest, identities)
        atomic_json(run / "selection.json", dict(config_hash=report["config_hash"], hashes=identities,
                                                   selected_epoch=None, selection="fixed_RF_control; no_validation_tuning",
                                                   frozen_before_test=True, class_order=[0, 1, 2],
                                                   feature_names=list(args.feature_names), summary_feature_names=selected_summary_names))
        test = table[table.split.eq("test")]
        test_X = test.loc[:, selected_summary_names]
        test_probabilities = model.predict_proba(test_X)
        test_metrics, test_path = save_predictions(run, "test", test, test_probabilities, manifest, identities)
        pd.concat([pd.read_parquet(val_path), pd.read_parquet(test_path)], ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
        p95 = _p95_predict(model, test_X)
        train_subjects = sorted(set(train.subject_id.astype(str)))
        provenance = dict(experiment_id=args.experiment_id or "RF_D", seed=config["seed"], feature_names=list(args.feature_names),
                          summary_feature_names=selected_summary_names, calibration_mode=args.mode,
                          outer_fold=(args.outer_fold if args.outer_fold is not None else args.outer_index + 1),
                          outer_index=args.outer_index, p95_forward_ms=p95, train_subjects=train_subjects,
                          snapshot_sha256=identities["snapshot"], split_sha256=identities["split"],
                          config_sha256=report["config_hash"], model_sha256=None,
                          scaler_sha256=None, model=str(run / "model.joblib"), predictions=str(run / "predictions.parquet"))
        atomic_json(run / "provenance.json", provenance)
        atomic_json(run / "metrics.json", dict(validation=val_metrics, test=test_metrics))
        report.update(status="complete", metrics=dict(validation_macro_f1=val_metrics["macro_f1"], test_macro_f1=test_metrics["macro_f1"]),
                      p95_forward_ms=p95, artifacts=dict(model=str(run/"model.joblib"), summaries=str(run/"summaries.parquet"),
                      summary_schema=str(run/"summary_schema.json"), config=str(run/"resolved_config.json"), split=str(run/"split.json"),
                      seed=str(run/"seed.json"), environment=str(run/"environment.json"), selection=str(run/"selection.json"),
                      validation_predictions=str(val_path), test_predictions=str(test_path), predictions=str(run/"predictions.parquet"),
                      metrics=str(run/"metrics.json"), provenance=str(run/"provenance.json")))
    except BlockedExperiment as exc:
        if report is not None:
            report.update(status="blocked", reasons=exc.reasons, coverage=exc.coverage)
    except Exception as exc:
        if report is not None:
            report.update(status="failed", error=str(exc))
        else:
            print(f"Baseline failed: {exc}")
    if report is not None:
        report["elapsed_s"] = perf_counter() - started
        atomic_json(Path(args.run_dir).resolve() / "report.json", report)
        print(f"Baseline {args.mode} outer{args.outer_index}: {report['status']}; metrics={report.get('metrics')}; report={Path(args.run_dir).resolve()/'report.json'}", flush=True)
        return int(report["status"] != "complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
