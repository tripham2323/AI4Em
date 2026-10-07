"""Integration tests. Sections are split by owner; do not edit another owner's section.

# ===== Người 2: realtime/replay section (not written here) =====
# ===== Người 1: evaluation / ablation / external section =====
"""
import json

import numpy as np
import pytest

from scripts import external_test as ext
from scripts import run_ablation as abl
from src.config import PROJECT_ROOT
from src.evaluation import evaluator as ev


def _fold(fold, subjects, wrong_first=True):
    labels, sids, vids, preds = [], [], [], []
    for s in subjects:
        for lab in (0, 1, 2):
            for w in range(6):
                labels.append(lab); sids.append(s); vids.append(f"{s}_v{lab}")
                preds.append((lab + 1) % 3 if (wrong_first and w == 0) else lab)
    probs = np.full((len(preds), 3), 0.1)
    probs[np.arange(len(preds)), preds] = 0.8
    return ev.evaluate(labels, probs, subject_ids=sids, video_ids=vids, fold=fold, min_video_coverage=0.5, n_boot=20)


def test_evaluate_to_ablation_collect_pipeline_with_cold_reload(tmp_path):
    """evaluate() -> run artifacts on disk -> run_ablation.collect from a cold read of those files."""
    matrix_path = abl.freeze("int", PROJECT_ROOT / "configs/training.yaml", [42], tmp_path)
    matrix = json.loads(matrix_path.read_text())
    specs = {v["experiment_id"]: v for v in matrix["variants"]}
    runs = []
    for exp in ("A", "B"):
        spec = specs[exp]
        report = _fold(1, ["s1", "s2", "s3"])
        full = {"folds": {"1": report}, "provenance": {"user_provenance": {
            "experiment_id": exp, "seed": 42, "feature_names": spec["feature_names"],
            "calibration_mode": spec["calibration_mode"], "p95_forward_ms": 2.5, "train_subjects": {"1": 8}}}}
        ev.write_run_artifacts(tmp_path / f"run_{exp}", full)
        runs.append(tmp_path / f"run_{exp}")
    code = abl.main(["collect", "--run-id", "int", "--out-root", str(tmp_path), "--eval-run", *map(str, runs), "--n-boot", "50"])
    assert code == 0
    lines = (tmp_path / "ablation_int" / "results.csv").read_text().splitlines()
    assert len(lines) == 3 and lines[0].split(",") == abl.RESULT_COLUMNS
    paired = json.loads((tmp_path / "ablation_int" / "paired.json").read_text())
    assert paired["lstm_B_vs_A"]["n_subjects"] == 3 and paired["lstm_B_vs_A"]["mean_difference"] == 0.0


def test_five_outer_folds_aggregate_and_every_subject_tested_once():
    folds = {i: {"train": [f"s{j}" for j in range(1, 11) if j not in (2 * i - 1, 2 * i)][:6],
                 "val": [], "test": [f"s{2 * i - 1}", f"s{2 * i}"]} for i in range(1, 6)}
    assert ev.verify_outer_split(folds, [f"s{j}" for j in range(1, 11)])["tested_once"]
    reports = [_fold(i, folds[i]["test"]) for i in range(1, 6)]
    agg = ev.aggregate_folds(reports)
    assert agg["complete_five_fold"] is True and agg["window"]["n_folds_valid"] == 5
    assert agg["window"]["pooled"]["denominators"]["accepted"] == 5 * 2 * 18


def test_external_blocked_does_not_affect_uta_reports(tmp_path):
    uta = _fold(1, ["s1", "s2"])
    policy = {"threshold": 0.5, "threshold_source": "uta_validation", "calibration_mode": "P0",
              "checkpoint_sha256": "a" * 64, "scaler_sha256": "b" * 64, "source_modality": "UTA RGB"}
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    (tmp_path / "p.csv").write_text("scenario,video_id,risk_label,p_alert,p_low,p_drowsy\n")
    code = ext.main(["--predictions", str(tmp_path / "p.csv"), "--policy", str(tmp_path / "policy.json"),
                     "--run-dir", str(tmp_path / "ext"), "--target-modality", "NTHU"])
    assert code == ext.EXIT_BLOCKED
    assert uta["status"] == "ok" and uta["window"]["metrics"]["macro_f1"] is not None
