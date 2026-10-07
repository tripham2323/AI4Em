"""Phase 15 ablation orchestration and Phase 21 external binary evaluation (behavioral tests)."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import external_test as ext
from scripts import run_ablation as abl
from src.config import PROJECT_ROOT

CONFIG = yaml.safe_load((PROJECT_ROOT / "configs/training.yaml").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- matrix
def test_matrix_variants_have_distinct_feature_orders_and_expected_groups():
    m = abl.build_matrix(CONFIG, seeds=[42])
    by = {v["experiment_id"]: v for v in m["variants"]}
    assert [by[k]["input_size"] for k in "ABCDE"] == [6, 8, 11, 16, 16]
    assert set(by["A"]["feature_names"]) < set(by["B"]["feature_names"]) < set(by["C"]["feature_names"]) < set(by["D"]["feature_names"])
    assert by["D"]["feature_names"] == CONFIG["feature_names"]  # config order preserved
    assert "perclos_ready" not in by["A"]["feature_names"] and "perclos_ready" in by["B"]["feature_names"]
    assert all("calibration_valid" in v["feature_names"] for v in m["variants"])
    assert by["E"]["calibration_mode"] == "P1" and all(by[k]["calibration_mode"] == "P0" for k in "ABCD")
    assert by["RF_D"]["model"] == "rf" and by["RF_D"]["feature_names"] == by["D"]["feature_names"]
    assert [e["experiment_id"] for e in m["extensions"]] == ["E03", "E04", "E06"]
    assert all(e["status"] == "chưa thực hiện" for e in m["extensions"])


def test_matrix_rejects_config_with_unknown_or_missing_features():
    bad = dict(CONFIG, feature_names=CONFIG["feature_names"][:-1])
    with pytest.raises(abl.AblationError):
        abl.build_matrix(bad, seeds=[1])
    with pytest.raises(abl.AblationError):
        abl.build_matrix(CONFIG, seeds=[])


def test_freeze_is_immutable_and_idempotent(tmp_path):
    cfg = PROJECT_ROOT / "configs/training.yaml"
    first = abl.freeze("t", cfg, [42], tmp_path)
    assert abl.freeze("t", cfg, [42], tmp_path) == first
    with pytest.raises(abl.AblationError):
        abl.freeze("t", cfg, [1, 2], tmp_path)  # different matrix under the same id


def test_train_is_blocked_not_faked_while_trainer_script_is_missing(tmp_path, monkeypatch):
    abl.freeze("tr", PROJECT_ROOT / "configs/training.yaml", [42], tmp_path)
    missing_trainer = tmp_path / "missing_train_lstm.py"
    monkeypatch.setitem(abl.TRAINER_SCRIPTS, "lstm", missing_trainer)
    assert not missing_trainer.exists()
    assert abl.main(["train", "--run-id", "tr", "--out-root", str(tmp_path), "--experiments", "A"]) == abl.EXIT_BLOCKED
    assert abl.main(["train", "--run-id", "missing", "--out-root", str(tmp_path)]) == 2  # no frozen matrix


def test_train_dry_run_plans_every_variant_seed_fold_with_own_feature_order(tmp_path):
    abl.freeze("pl", PROJECT_ROOT / "configs/training.yaml", [42, 7], tmp_path)
    assert abl.main(["train", "--run-id", "pl", "--out-root", str(tmp_path), "--dry-run"]) == 0
    plan = json.loads((tmp_path / "ablation_pl" / "train_plan.json").read_text())
    assert len(plan) == 6 * 2 * 5  # A-E + RF_D, two seeds, five folds
    a = next(p for p in plan if p["experiment_id"] == "A")["command"]
    assert a[a.index("--feature-names") + 1] == "ear_left_norm,ear_right_norm,closure_elapsed_s,left_eye_valid,right_eye_valid,calibration_valid"
    assert a[a.index("--calibration-mode") + 1] == "P0" and "scripts.train_lstm" in a
    rf = next(p for p in plan if p["experiment_id"] == "RF_D")["command"]
    assert "scripts.train_baseline" in rf
    e = next(p for p in plan if p["experiment_id"] == "E")["command"]
    assert e[e.index("--calibration-mode") + 1] == "P1"
    with pytest.raises(abl.AblationError):
        abl.plan_runs(json.loads((tmp_path / "ablation_pl" / "matrix.json").read_text()), tmp_path, ["Z"])


# ----------------------------------------------------------------- collect
def _fake_eval_run(path, matrix, exp, seed, subjects, *, fold=1, drop_feature=False, macro_by_subject=None):
    spec = {v["experiment_id"]: v for v in matrix["variants"]}[exp]
    feats = spec["feature_names"][:-1] if drop_feature else spec["feature_names"]
    per_subject = {s: {"metrics": {"macro_f1": (macro_by_subject or {}).get(s, 0.5)}} for s in subjects}
    metrics = {"low_vigilance_precision": 0.4, "low_vigilance_recall": 0.5, "drowsy_recall": None,
               "macro_f1": 0.5, "weighted_f1": 0.55, "accuracy": 0.6}
    rep = {"window": {"counts": {"accepted": 100}, "coverage": 0.9, "metrics": metrics},
           "video": {"report": {"metrics": {"accuracy": 0.7}}}, "subject": {"per_subject": per_subject}}
    prov = {"experiment_id": exp, "seed": seed, "feature_names": feats, "calibration_mode": spec["calibration_mode"],
            "p95_forward_ms": 3.2, "train_subjects": {str(fold): 8}}
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"folds": {str(fold): rep}, "provenance": {"user_provenance": prov}}))
    return path


def _matrix(tmp_path):
    return json.loads(abl.freeze("m", PROJECT_ROOT / "configs/training.yaml", [42], tmp_path).read_text())


def test_collect_writes_results_with_undefined_left_empty_and_lists_not_run(tmp_path):
    m = _matrix(tmp_path)
    runs = [_fake_eval_run(tmp_path / f"r{e}", m, e, 42, ["s1", "s2"]) for e in "AB"]
    rows, per_subject = abl.collect_rows(m, runs)
    abl.write_results(tmp_path / "out", m, rows, abl.compare_paired_subjects(per_subject, n_boot=50))
    text = (tmp_path / "out" / "results.csv").read_text().splitlines()
    header = text[0].split(",")
    assert header == abl.RESULT_COLUMNS and len(text) == 3
    assert dict(zip(header, text[1].split(",")))["drowsy_recall"] == ""  # undefined stays empty, not 0
    not_run = {r["experiment_id"] for r in json.loads((tmp_path / "out" / "not_run.json").read_text())}
    assert {"C", "D", "E", "RF_D", "E03", "E04", "E06"} <= not_run
    paired = json.loads((tmp_path / "out" / "paired.json").read_text(encoding="utf-8"))
    assert paired["lstm_C_vs_B"]["status"] == "chưa thực hiện" and paired["lstm_B_vs_A"]["n_subjects"] == 2


def test_collect_rejects_mismatched_features_seed_or_unfrozen_experiment(tmp_path):
    m = _matrix(tmp_path)
    with pytest.raises(abl.AblationError):
        abl.collect_rows(m, [_fake_eval_run(tmp_path / "a", m, "B", 42, ["s1"], drop_feature=True)])
    with pytest.raises(abl.AblationError):
        abl.collect_rows(m, [_fake_eval_run(tmp_path / "b", m, "B", 7, ["s1"])])
    run = _fake_eval_run(tmp_path / "c", m, "B", 42, ["s1"])
    meta = json.loads((run / "metrics.json").read_text())
    meta["provenance"]["user_provenance"]["experiment_id"] = "Z"
    (run / "metrics.json").write_text(json.dumps(meta))
    with pytest.raises(abl.AblationError):
        abl.collect_rows(m, [run])


def test_collect_requires_same_p0_cohort_but_allows_e_on_p1_intersection(tmp_path):
    m = _matrix(tmp_path)
    a = _fake_eval_run(tmp_path / "a", m, "A", 42, ["s1", "s2"])
    d_other = _fake_eval_run(tmp_path / "d", m, "D", 42, ["s1", "s3"])
    with pytest.raises(abl.AblationError):
        abl.collect_rows(m, [a, d_other])
    d = _fake_eval_run(tmp_path / "d2", m, "D", 42, ["s1", "s2"])
    e = _fake_eval_run(tmp_path / "e", m, "E", 42, ["s1"])  # P1 cohort is smaller: allowed
    rows, _ = abl.collect_rows(m, [a, d, e])
    assert len(rows) == 3


def test_collect_rejects_duplicate_experiment_seed_fold(tmp_path):
    m = _matrix(tmp_path)
    r1 = _fake_eval_run(tmp_path / "a1", m, "A", 42, ["s1"])
    r2 = _fake_eval_run(tmp_path / "a2", m, "A", 42, ["s1"])
    with pytest.raises(abl.AblationError):
        abl.collect_rows(m, [r1, r2])


def test_paired_comparison_uses_common_subjects_and_gives_evidence_hint(tmp_path):
    m = _matrix(tmp_path)
    subs = [f"s{i}" for i in range(8)]
    a = _fake_eval_run(tmp_path / "a", m, "A", 42, subs, macro_by_subject={s: 0.40 for s in subs})
    b = _fake_eval_run(tmp_path / "b", m, "B", 42, subs, macro_by_subject={s: 0.40 + 0.05 + 0.001 * i for i, s in enumerate(subs)})
    _, per_subject = abl.collect_rows(m, [a, b])
    out = abl.compare_paired_subjects(per_subject, n_boot=200)["lstm_B_vs_A"]
    assert out["n_subjects"] == 8 and out["n_gain"] == 8 and out["evidence_hint"] == "gain_ci_excludes_zero"


# ----------------------------------------------------------------- external
POLICY = {"threshold": 0.5, "threshold_source": "uta_validation", "calibration_mode": "P0",
          "checkpoint_sha256": "a" * 64, "scaler_sha256": "b" * 64, "source_modality": "UTA RGB"}
ACCESS = {"dataset": "NTHU-DDD", "agreement_signed": True, "official_partitions": True,
          "labels_accessible": True, "annotation_source": "official NTHU annotation files"}


def _ext_frame():
    rows = []
    for scen, (n_pos, n_neg) in {"s1": (3, 3), "s2": (2, 0)}.items():
        for i in range(n_pos):
            rows.append((scen, f"{scen}_p{i}", 1, 0.2, 0.3, 0.5))      # p_risk 0.8
        for i in range(n_neg):
            rows.append((scen, f"{scen}_n{i}", 0, 0.9, 0.05, 0.05))    # p_risk 0.1
    return pd.DataFrame(rows, columns=["scenario", "video_id", "risk_label", "p_alert", "p_low", "p_drowsy"])


def test_external_uses_p_risk_sum_and_reports_binary_metrics_per_scenario():
    report, work = ext.evaluate_external(_ext_frame(), POLICY, expected_scenarios=2)
    assert np.allclose(work["p_risk"].iloc[0], 0.8)
    s1 = report["scenarios"]["s1"]["metrics"]
    assert s1["macro_f1"] == 1.0 and s1["risk_recall"] == 1.0 and s1["pr_auc"] == 1.0
    assert report["risk_definition"] == "p_risk = p_low + p_drowsy"
    assert "low_vigilance_recall" in report["not_reported"] and "low_vigilance_recall" not in s1


def test_external_scenario_with_one_class_has_undefined_pr_auc_not_zero():
    report, _ = ext.evaluate_external(_ext_frame(), POLICY, expected_scenarios=2)
    s2 = report["scenarios"]["s2"]["metrics"]
    assert s2["pr_auc"] is None and s2["risk_recall"] == 1.0
    assert any("scenario s2" in lim and "PR-AUC" in lim for lim in report["limitations"])


def test_external_reports_missing_scenarios_without_substitution():
    report, _ = ext.evaluate_external(_ext_frame(), POLICY, expected_scenarios=5)
    assert report["n_scenarios_provided"] == 2 and any("2 of 5" in lim for lim in report["limitations"])


def test_external_abstained_rows_keep_coverage_and_count_missed_risk():
    df = _ext_frame().assign(status="accepted")
    df.loc[0, "status"] = "rejected"
    report, _ = ext.evaluate_external(df, POLICY, expected_scenarios=2)
    s1 = report["scenarios"]["s1"]
    assert s1["coverage"] == pytest.approx(5 / 6) and s1["risk_rows_abstained"] == 1


def test_external_rejects_three_class_labels_and_nonfrozen_policy():
    with pytest.raises(ext.ExternalInputError):
        ext.evaluate_external(_ext_frame().assign(risk_label=[2] + [1] * 7), POLICY)
    for bad in (dict(POLICY, calibration_mode="P1"), dict(POLICY, threshold_source="nthu_test"),
                dict(POLICY, threshold=1.5)):
        with pytest.raises(ext.ExternalInputError):
            ext.evaluate_external(_ext_frame(), bad)


def test_external_access_check_blocks_unattested_access():
    assert ext.check_access(None) is not None
    assert ext.check_access(dict(ACCESS, agreement_signed=False)) is not None
    assert ext.check_access(dict(ACCESS, official_partitions=False)) is not None
    assert ext.check_access(ACCESS) is None


def test_external_cli_blocked_writes_no_scores(tmp_path):
    (tmp_path / "policy.json").write_text(json.dumps(POLICY))
    _ext_frame().to_csv(tmp_path / "p.csv", index=False)
    code = ext.main(["--predictions", str(tmp_path / "p.csv"), "--policy", str(tmp_path / "policy.json"),
                     "--run-dir", str(tmp_path / "run"), "--target-modality", "NTHU infrared"])
    assert code == ext.EXIT_BLOCKED
    report = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert report["status"] == "blocked" and "scenarios" not in report
    assert not (tmp_path / "run" / "external_predictions.csv").exists()


def test_external_cli_end_to_end_with_attested_access(tmp_path):
    (tmp_path / "policy.json").write_text(json.dumps(POLICY))
    (tmp_path / "access.json").write_text(json.dumps(ACCESS))
    _ext_frame().to_csv(tmp_path / "p.csv", index=False)
    code = ext.main(["--predictions", str(tmp_path / "p.csv"), "--policy", str(tmp_path / "policy.json"),
                     "--access", str(tmp_path / "access.json"), "--run-dir", str(tmp_path / "run"),
                     "--target-modality", "NTHU infrared", "--expected-scenarios", "2"])
    assert code == 0
    report = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert report["status"] == "ok" and report["target_modality"] == "NTHU infrared"
