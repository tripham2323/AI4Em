"""Train a frozen three-class LSTM for one outer fold.

Supports both the original Phase-13 CLI and the Phase-15 ablation contract:
  --derived-manifest ... --outer-index 0 --mode P0 ...
  --experiment-id A --feature-names a,b,... --calibration-mode P0
      --seed 42 --outer-fold 1 --run-dir ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.datasets.derived import atomic_json
from src.datasets.normalization import FEATURE_NAMES, fit_scaler, scaler_hash
from src.datasets.sequence import SequenceDataset
from src.models.bundle import ModelBundle
from src.models.lstm import LSTMClassifier
from src.training.experiment import (
    BlockedExperiment, check_eligibility, hashes, prepare_run, project_path,
    save_predictions, unique_train_timesteps, validate_dataset,
)
from src.training.trainer import ModelTrainer, seed_training


def _find_manifest(mode: str, outer_index: int, explicit: Path | None) -> Path:
    if explicit:
        return project_path(explicit)
    report = PROJECT_ROOT / "runs" / "phase11" / "derived" / f"{mode}.json"
    if not report.is_file():
        raise FileNotFoundError(
            f"No derived manifest report found at {report}. Run build_derived first or pass --derived-manifest.")
    data = json.loads(report.read_text(encoding="utf-8"))
    for artifact in data.get("artifacts", []):
        if int(artifact.get("outer_index", -1)) == outer_index:
            path = Path(artifact["path"])
            if path.is_file():
                return path
    raise FileNotFoundError(f"No {mode} derived manifest for outer index {outer_index} in {report}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # Original contract
    p.add_argument("--derived-manifest", type=Path)
    p.add_argument("--outer-index", type=int, choices=range(5))
    p.add_argument("--mode", choices=("P0", "P1"))
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    # Phase-15 contract
    p.add_argument("--experiment-id", choices=("A", "B", "C", "D", "E"))
    p.add_argument("--feature-names", help="comma-separated ordered feature names")
    p.add_argument("--calibration-mode", choices=("P0", "P1"))
    p.add_argument("--seed", type=int)
    p.add_argument("--outer-fold", type=int, choices=range(1, 6))
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args(argv)

    ablation = args.experiment_id is not None or args.outer_fold is not None or args.feature_names is not None
    if ablation:
        if args.experiment_id is None or args.feature_names is None or args.calibration_mode is None or args.seed is None or args.outer_fold is None:
            p.error("Phase-15 mode requires --experiment-id, --feature-names, --calibration-mode, --seed and --outer-fold")
        args.mode = args.calibration_mode
        args.outer_index = args.outer_fold - 1
        args.feature_names = tuple(x.strip() for x in args.feature_names.split(",") if x.strip())
        if not args.feature_names or any(x not in FEATURE_NAMES for x in args.feature_names):
            p.error("--feature-names contains an unknown/empty feature")
    else:
        if args.derived_manifest is None or args.outer_index is None or args.mode is None:
            p.error("original mode requires --derived-manifest, --outer-index and --mode")
        args.feature_names = None
    return args


def _loader(index, manifest, scaler, config, feature_names, *, shuffle=False):
    dataset = SequenceDataset(index, derived_manifest=manifest, scaler=scaler,
                              feature_names=tuple(feature_names))
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    return DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=shuffle,
                      num_workers=0, drop_last=False, generator=generator)


def _predict(model, loader, *, device, reload_path):
    model.eval()
    parts, labels = [], []
    with torch.inference_mode():
        for batch_index, (x, y, _) in enumerate(loader):
            if batch_index == 0:
                np.save(reload_path, x.numpy(), allow_pickle=False)
            probabilities = model(x.to(device=device, dtype=torch.float32)).softmax(dim=1)
            parts.append(probabilities.cpu().numpy())
            labels.append(y.numpy())
    return (np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.float32),
            np.concatenate(labels) if labels else np.empty(0, dtype=np.int64))


def _p95_forward_ms(model, x: np.ndarray) -> float | None:
    if len(x) == 0:
        return None
    tensor = torch.as_tensor(x[:1], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(tensor)
        times = []
        for _ in range(30):
            t0 = perf_counter(); model(tensor); times.append((perf_counter() - t0) * 1000.0)
    return float(np.percentile(times, 95))


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = perf_counter()
    # Resolve the ablation contract to the existing verified Phase-13 contract.
    if args.outer_index is None:
        args.outer_index = args.outer_fold - 1
    args.derived_manifest = _find_manifest(args.mode, args.outer_index, args.derived_manifest)
    if args.experiment_id is None:
        args.feature_names = tuple(FEATURE_NAMES)
    feature_names = tuple(args.feature_names)
    report = None
    try:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(args.cpu_threads)
        except RuntimeError:
            pass
        config, run, report = prepare_run(args, "lstm")
        config["feature_names"] = list(feature_names)
        config["seed"] = int(args.seed if args.seed is not None else config.get("seed", 42))
        config["experiment_id"] = args.experiment_id or "FULL"
        atomic_json(run / "resolved_config.json", config)
        manifest, index, split = validate_dataset(args.derived_manifest, mode=args.mode,
                                                   outer_index=args.outer_index, config={**config, "feature_names": list(FEATURE_NAMES)})
        report.update(experiment_id=args.experiment_id or "FULL", seed=config["seed"],
                     feature_names=list(feature_names), calibration_mode=args.mode,
                     hashes=hashes(manifest), coverage=manifest["coverage"],
                     counts=manifest["class_support"], dataset_manifest_hash=manifest["dataset_manifest_hash"])
        atomic_json(run / "split.json", split)
        atomic_json(run / "dataset_manifest.json", manifest)
        check_eligibility(manifest, index)

        # Fit only on training timesteps, then validate/test with the frozen scaler.
        scaler = fit_scaler(unique_train_timesteps(index, manifest), feature_names)
        for key in ("mode", "schema_version", "derived_hash", "split_hash", "snapshot_sha256", "outer_index"):
            scaler[key] = manifest[key]
        scaler["fit_scope"] = "unique_accepted_train_timesteps"
        scaler["index_sha256"] = manifest.get("index_sha256")
        scaler["hash"] = scaler_hash(scaler)
        atomic_json(run / "scaler.json", scaler)
        identities = hashes(manifest, scaler)
        train_index = index[index.split.eq("train")].reset_index(drop=True)
        val_index = index[index.split.eq("validation")].reset_index(drop=True)
        train_loader = _loader(train_index, manifest, scaler, config, feature_names, shuffle=True)
        val_loader = _loader(val_index, manifest, scaler, config, feature_names)

        seed_training(config["seed"])
        model = LSTMClassifier(tuple(feature_names), hidden_size=config.get("hidden_size", 64),
                               num_layers=config.get("num_layers", 1), head_dropout=config.get("head_dropout", .3))
        trainer_config = {**config, "val_coverage": dict(manifest["coverage"]["by_role"]["validation"])}
        result = ModelTrainer(model, train_loader, val_loader, trainer_config).fit()
        selected = result["best_val_predictions"]
        if not np.array_equal(selected["labels"], val_index.label_id.to_numpy()):
            raise ValueError("Selected validation predictions differ from accepted index")
        pd.DataFrame(result["history"]).to_csv(run / "history.csv", index=False)
        val_metrics, val_path = save_predictions(run, "validation", val_index, selected["probabilities"], manifest, identities)
        selection = dict(best_epoch=result["best_epoch"], metric=result["selection_metric"],
                         tie_policy=result["selection_tie_policy"], validation_metrics=val_metrics,
                         config_hash=report["config_hash"], hashes=identities, frozen_before_test=True,
                         epochs_executed=len(result["history"]), stopped_early=result["stopped_early"])
        atomic_json(run / "selection.json", selection)
        metadata = dict(mode=args.mode, schema_version=manifest["schema_version"],
                        feature_names=list(feature_names), class_order=[0, 1, 2], hashes=identities,
                        outer_index=args.outer_index, dataset_manifest_hash=manifest["dataset_manifest_hash"],
                        index_sha256=manifest["index_sha256"], config=config, config_hash=report["config_hash"],
                        selection=selection, environment=report["environment"], trainer_environment=result["environment"],
                        source_sha256=report["source_sha256"], derived_identity=manifest["identity"],
                        experiment_id=args.experiment_id or "FULL", seed=config["seed"],
                        timing=dict(sequence_fps=10, steps=100, stride_s=1, tick_span_ms=9900,
                                    nominal_duration_ms=10000, state_reset="every_window", max_missing_ratio=.20, max_gap_s=1),
                        training_elapsed_s=result["elapsed_s"], class_weights=result["class_weights"])
        bundle_path = ModelBundle.save(run / "model_bundle", model=model, scaler=scaler, metadata=metadata)

        test_index = index[index.split.eq("test")].reset_index(drop=True)
        test_loader = _loader(test_index, manifest, scaler, config, feature_names)
        test_probabilities, test_labels = _predict(model, test_loader, device=next(model.parameters()).device,
                                                    reload_path=run / "test_reload_windows.npy")
        if not np.array_equal(test_labels, test_index.label_id.to_numpy()):
            raise ValueError("Test predictions differ from accepted index")
        test_metrics, test_path = save_predictions(run, "test", test_index, test_probabilities, manifest, identities)
        atomic_json(run / "metrics.json", dict(validation=val_metrics, test=test_metrics))
        pd.concat([pd.read_parquet(val_path), pd.read_parquet(test_path)], ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
        latency_x = np.load(run / "test_reload_windows.npy", allow_pickle=False)
        p95 = _p95_forward_ms(model, latency_x)
        train_subjects = sorted(set(train_index.subject_id.astype(str)))
        provenance = dict(experiment_id=args.experiment_id or "FULL", seed=config["seed"], feature_names=list(feature_names),
                          calibration_mode=args.mode, outer_fold=(args.outer_fold if args.outer_fold is not None else args.outer_index + 1),
                          outer_index=args.outer_index, p95_forward_ms=p95, train_subjects=train_subjects,
                          snapshot_sha256=identities["snapshot"], split_sha256=identities["split"],
                          config_sha256=report["config_hash"], scaler_sha256=identities["scaler"],
                          model_bundle=str(bundle_path), predictions=str(run / "predictions.parquet"))
        atomic_json(run / "provenance.json", provenance)
        report.update(status="complete", selected_epoch=result["best_epoch"], fit_elapsed_s=result["elapsed_s"],
                      metrics=dict(validation_macro_f1=val_metrics["macro_f1"], test_macro_f1=test_metrics["macro_f1"]),
                      p95_forward_ms=p95, artifacts=dict(bundle=str(bundle_path), scaler=str(run/"scaler.json"),
                      config=str(run/"resolved_config.json"), split=str(run/"split.json"), seed=str(run/"seed.json"),
                      environment=str(run/"environment.json"), history=str(run/"history.csv"), selection=str(run/"selection.json"),
                      metrics=str(run/"metrics.json"), validation_predictions=str(val_path), test_predictions=str(test_path),
                      predictions=str(run/"predictions.parquet"), provenance=str(run/"provenance.json")))
    except BlockedExperiment as exc:
        if report is not None:
            report.update(status="blocked", reasons=exc.reasons, coverage=exc.coverage)
    except Exception as exc:
        if report is not None:
            report.update(status="failed", error=str(exc))
        else:
            print(f"LSTM failed: {exc}")
    if report is not None:
        report["elapsed_s"] = perf_counter() - started
        atomic_json(Path(args.run_dir).resolve() / "report.json", report)
        print(f"LSTM {args.mode} outer{args.outer_index}: {report['status']}; metrics={report.get('metrics')}; report={Path(args.run_dir).resolve()/'report.json'}", flush=True)
        return int(report["status"] != "complete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
