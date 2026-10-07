"""Behavioral tests for the shared evaluator (Phase 11 core, Phase 14 fold/video/subject)."""
import json

import numpy as np
import pytest

from src.evaluation import evaluator as ev


# ----------------------------------------------------------------- Phase 11 known answer
def test_known_confusion_fixture_matches_independent_hand_calculation():
    report = ev.evaluate_classification([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    m = report["metrics"]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["macro_f1"] == pytest.approx(59 / 90)
    assert [report["per_class"][n]["recall"] for n in ev.CLASS_NAMES] == pytest.approx([0.5, 1.0, 0.5])
    assert [report["per_class"][n]["support"] for n in ev.CLASS_NAMES] == [2, 2, 2]
    assert report["confusion_matrix"]["counts"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["weighted_f1"] == pytest.approx(59 / 90)  # equal supports
    assert m["low_vigilance_recall"] == pytest.approx(1.0)
    assert m["low_vigilance_precision"] == pytest.approx(2 / 3)
    assert m["drowsy_recall"] == pytest.approx(0.5)
    assert report["coverage"] == 1.0 and report["counts"]["accepted"] == 6


def test_metrics_agree_with_sklearn_on_random_data():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 3, 400)
    y_pred = np.where(rng.random(400) < 0.6, y_true, rng.integers(0, 3, 400))
    m = ev.evaluate_classification(y_true, y_pred)["metrics"]
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted"))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_confusion_rows_are_true_class_and_columns_predicted():
    cm = ev.evaluate_classification([1, 1, 2], [0, 0, 1])["confusion_matrix"]["counts"]
    assert cm[1] == [2, 0, 0] and cm[2] == [0, 1, 0]


def test_absent_class_metrics_are_undefined_not_zero():
    report = ev.evaluate_classification([1, 1, 1, 1], [1, 1, 0, 2])  # only Low Vigilance has support
    assert report["classes_present"] == ["low_vigilance"] and not report["full_class_coverage"]
    assert report["per_class"]["alert"]["recall"] is None and report["per_class"]["alert"]["f1"] is None
    assert report["per_class"]["drowsy"]["recall"] is None
    assert report["metrics"]["drowsy_recall"] is None
    assert report["per_class"]["alert"]["support"] == 0
    assert report["metrics"]["macro_f1"] == pytest.approx(2 / 3)  # precision 1, recall 1/2, one present class
    assert report["confusion_matrix"]["normalized_by_true"][0] == [None, None, None]


def test_zero_accepted_support_makes_every_metric_undefined():
    report = ev.evaluate_classification([0, 1, 2], [-1, -1, -1], accepted=[False, False, False])
    assert report["defined"] is False and report["coverage"] == 0.0
    assert all(v is None for v in report["metrics"].values())
    assert report["abstained_true_class"] == {"alert": 1, "low_vigilance": 1, "drowsy": 1}


def test_abstention_is_excluded_from_metrics_but_kept_in_coverage():
    report = ev.evaluate_classification([0, 1, 2, 2], [0, 1, 2, -1], accepted=[True, True, True, False])
    assert report["metrics"]["accuracy"] == 1.0
    assert report["coverage"] == pytest.approx(0.75)
    assert report["abstained_true_class"]["drowsy"] == 1
    assert report["by_class_counts"]["drowsy"] == {"scheduled": 2, "blocked": 0, "eligible": 2, "accepted": 1, "rejected": 1}


def test_blocked_rows_are_counted_separately_from_rejected():
    report = ev.evaluate_classification([0, 1, 2, 2], [0, 1, 2, -1],
                                        status=["accepted", "accepted", "blocked", "rejected"])
    c = report["counts"]
    assert (c["scheduled"], c["blocked"], c["eligible"], c["accepted"], c["rejected"]) == (4, 1, 3, 2, 1)
    assert report["coverage"] == pytest.approx(0.5) and report["eligible_coverage"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("y_true,y_pred", [([0, 3], [0, 0]), ([0, 1], [0, 5]), ([0.5, 1], [0, 1]), ([0, np.nan], [0, 1])])
def test_invalid_class_ids_are_rejected(y_true, y_pred):
    with pytest.raises(ValueError):
        ev.evaluate_classification(y_true, y_pred)


def test_accepted_mask_must_agree_with_status():
    with pytest.raises(ValueError):
        ev.evaluate_classification([0, 1], [0, 1], accepted=[True, True], status=["accepted", "rejected"])


# ----------------------------------------------------------------- evaluate() fixtures
def _probs(classes, confidence=0.8):
    p = np.full((len(classes), 3), (1 - confidence) / 2)
    p[np.arange(len(classes)), classes] = confidence
    return p


def _fold_rows(subjects=("s1", "s2"), windows_per_video=5):
    rows = {"labels": [], "subject_ids": [], "video_ids": [], "preds": []}
    for s in subjects:
        for label in (0, 1, 2):
            for w in range(windows_per_video):
                rows["labels"].append(label)
                rows["subject_ids"].append(s)
                rows["video_ids"].append(f"{s}_v{label}")
                rows["preds"].append(label if w != 0 else (label + 1) % 3)  # one wrong window per video
    return rows


def _evaluate(rows, **kw):
    kw.setdefault("min_video_coverage", 0.5)
    kw.setdefault("n_boot", 50)
    return ev.evaluate(rows["labels"], _probs(rows["preds"]), subject_ids=rows["subject_ids"],
                       video_ids=rows["video_ids"], **kw)


def test_evaluate_returns_window_video_subject_and_bootstrap_sections():
    report = _evaluate(_fold_rows(), fold=1)
    assert report["status"] == "ok" and report["window"]["counts"]["accepted"] == 30
    assert report["video"]["report"]["metrics"]["accuracy"] == 1.0  # mean prob fixes the 1 wrong window
    assert report["window"]["metrics"]["accuracy"] == pytest.approx(24 / 30)
    assert report["subject"]["n_subjects"] == 2 and report["bootstrap"]["n_subjects"] == 2
    json.dumps(report, allow_nan=False)  # strict JSON


def test_video_prediction_is_mean_probability_then_argmax_not_majority_vote():
    # windows: two weakly class-0, one strongly class-2 -> mean favours class 2, majority favours class 0
    probs = np.array([[0.40, 0.30, 0.30], [0.40, 0.30, 0.30], [0.02, 0.03, 0.95]])
    out = ev.aggregate_videos(["v"] * 3, ["s"] * 3, [2, 2, 2], probs, ["accepted"] * 3, min_video_coverage=0.5)
    assert out["videos"][0]["pred"] == 2 and out["report"]["metrics"]["accuracy"] == 1.0


def test_video_with_low_coverage_abstains_and_is_never_alert():
    probs = _probs([2, 2, 2, 2])
    status = ["accepted", "rejected", "rejected", "rejected"]
    out = ev.aggregate_videos(["v"] * 4, ["s"] * 4, [2] * 4, probs, status, min_video_coverage=0.5)
    video = out["videos"][0]
    assert video["status"] == "rejected" and video["pred"] is None
    assert out["report"]["counts"]["accepted"] == 0 and out["report"]["abstained_true_class"]["drowsy"] == 1


def test_all_blocked_video_is_blocked_not_rejected():
    out = ev.aggregate_videos(["v"] * 2, ["s"] * 2, [1, 1], _probs([1, 1]), ["blocked", "blocked"], min_video_coverage=0.5)
    assert out["videos"][0]["status"] == "blocked"


def test_video_with_inconsistent_label_or_subject_is_rejected():
    with pytest.raises(ValueError):
        ev.aggregate_videos(["v", "v"], ["s", "s"], [0, 1], _probs([0, 1]), ["accepted"] * 2, min_video_coverage=0.5)
    with pytest.raises(ValueError):
        ev.aggregate_videos(["v", "v"], ["s", "t"], [0, 0], _probs([0, 0]), ["accepted"] * 2, min_video_coverage=0.5)


def test_probabilities_must_have_class_order_shape_and_sum_to_one():
    rows = _fold_rows()
    with pytest.raises(ValueError):
        ev.evaluate(rows["labels"], np.ones((len(rows["labels"]), 2)) / 2, subject_ids=rows["subject_ids"],
                    video_ids=rows["video_ids"], min_video_coverage=0.5)
    with pytest.raises(ValueError):
        ev.evaluate(rows["labels"], np.full((len(rows["labels"]), 3), 0.5), subject_ids=rows["subject_ids"],
                    video_ids=rows["video_ids"], min_video_coverage=0.5)


def test_rejected_rows_may_carry_nan_probabilities():
    rows = _fold_rows()
    probs = _probs(rows["preds"])
    accepted = np.ones(len(probs), bool)
    accepted[::7] = False
    probs[~accepted] = np.nan
    report = ev.evaluate(rows["labels"], probs, subject_ids=rows["subject_ids"], video_ids=rows["video_ids"],
                         accepted=accepted, min_video_coverage=0.5, n_boot=20)
    assert report["window"]["counts"]["rejected"] == int((~accepted).sum())
    json.dumps(report, allow_nan=False)


def test_missing_class_fold_is_flagged_in_notes():
    rows = {"labels": [1] * 6, "subject_ids": ["s5"] * 6, "video_ids": ["v"] * 6, "preds": [1, 1, 1, 0, 1, 1]}
    report = _evaluate(rows, fold=5)
    assert report["window"]["full_class_coverage"] is False
    assert any("absent" in n for n in report["notes"])


# ----------------------------------------------------------------- subject level / bootstrap
def test_subject_mean_averages_only_defined_subjects():
    rows = {"labels": [0, 1, 2, 1, 1], "subject_ids": ["a", "a", "a", "b", "b"], "video_ids": list("pppqq"),
            "preds": [0, 1, 2, 1, 0]}
    rep = ev.evaluate_subjects(rows["subject_ids"], rows["labels"], rows["preds"], status=["accepted"] * 5)
    assert rep["per_subject"]["a"]["metrics"]["macro_f1"] == 1.0
    assert rep["per_subject"]["b"]["metrics"]["drowsy_recall"] is None
    assert rep["mean_over_subjects"]["drowsy_recall"]["n_subjects_defined"] == 1
    assert rep["mean_over_subjects"]["macro_f1"]["n_subjects_defined"] == 2


def test_bootstrap_is_deterministic_and_resamples_subjects_not_windows():
    rows = _fold_rows(subjects=("s1", "s2", "s3", "s4"))
    args = (rows["subject_ids"], rows["labels"], rows["preds"])
    st = ["accepted"] * len(rows["labels"])
    a = ev.bootstrap_subject_ci(*args, status=st, n_boot=200, seed=7)
    b = ev.bootstrap_subject_ci(*args, status=st, n_boot=200, seed=7)
    assert a == b and a["n_subjects"] == 4
    low, high = a["metrics"]["macro_f1"]["ci_low"], a["metrics"]["macro_f1"]["ci_high"]
    assert low <= high and a["metrics"]["macro_f1"]["n_boot_valid"] == 200


def test_bootstrap_interval_reflects_subject_heterogeneity():
    # subject a perfect, subject b always wrong: resampling subjects must give a wide interval
    labels = [0, 1, 2] * 10
    preds = labels[:15] + [(x + 1) % 3 for x in labels[15:]]
    sids = ["a"] * 15 + ["b"] * 15
    out = ev.bootstrap_subject_ci(sids, labels, preds, status=["accepted"] * 30, n_boot=400, seed=1)
    ci = out["metrics"]["macro_f1"]
    assert ci["ci_high"] - ci["ci_low"] > 0.5


def test_bootstrap_with_one_subject_has_no_interval():
    out = ev.bootstrap_subject_ci(["a"] * 3, [0, 1, 2], [0, 1, 2], status=["accepted"] * 3, n_boot=10)
    assert out["metrics"]["macro_f1"]["ci_low"] is None and out["metrics"]["macro_f1"]["estimate"] == 1.0


def test_paired_difference_counts_gain_loss_and_skips_undefined():
    a = {"s1": 0.5, "s2": 0.6, "s3": None, "s4": 0.7}
    b = {"s1": 0.6, "s2": 0.5, "s3": 0.9, "s4": 0.7}
    out = ev.paired_subject_difference(a, b, n_boot=100, seed=0)
    assert out["subjects"] == ["s1", "s2", "s4"]
    assert (out["n_gain"], out["n_loss"], out["n_tie"]) == (1, 1, 1)
    assert out["mean_difference"] == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------- split / leakage checks
def test_every_outer_subject_is_tested_exactly_once():
    folds = {1: {"train": ["c", "d"], "val": ["b"], "test": ["a"]}, 2: {"train": ["a", "d"], "val": ["c"], "test": ["b"]},
             3: {"train": ["a", "b"], "val": ["d"], "test": ["c"]}, 4: {"train": ["a", "b"], "val": ["c"], "test": ["d"]}}
    assert ev.verify_outer_split(folds, expected_subjects="abcd")["tested_once"]
    folds[4]["test"] = ["a"]
    with pytest.raises(ValueError):
        ev.verify_outer_split(folds)
    folds[4]["test"] = ["d"]
    with pytest.raises(ValueError):
        ev.verify_outer_split(folds, expected_subjects="abcde")


def test_subject_in_two_partitions_of_one_fold_is_leakage():
    with pytest.raises(ValueError):
        ev.verify_outer_split({1: {"train": ["a", "b"], "val": ["c"], "test": ["a"]}})


def test_rows_from_non_test_subjects_are_rejected():
    rows = _fold_rows()
    with pytest.raises(ValueError):
        _evaluate(rows, test_subjects=["s1"])


def test_calibration_prefix_windows_cannot_enter_metrics():
    rows = _fold_rows(subjects=("s1",))
    starts = list(range(len(rows["labels"])))
    with pytest.raises(ValueError):
        _evaluate(rows, window_start_ms=starts, calibration_prefix_end_ms={"s1": 3})
    status = ["rejected" if s < 3 else "accepted" for s in starts]
    _evaluate(rows, window_start_ms=starts, calibration_prefix_end_ms={"s1": 3}, status=status)


# ----------------------------------------------------------------- aggregate_folds
def _fold_report(fold, subjects, *, accepted=True, windows=5):
    rows = _fold_rows(subjects=subjects, windows_per_video=windows)
    kw = {} if accepted else {"accepted": np.zeros(len(rows["labels"]), bool)}
    probs = _probs(rows["preds"]) if accepted else np.full((len(rows["labels"]), 3), np.nan)
    return ev.evaluate(rows["labels"], probs, subject_ids=rows["subject_ids"], video_ids=rows["video_ids"],
                       fold=fold, min_video_coverage=0.5, n_boot=10, **kw)


def test_aggregate_folds_reports_valid_fold_count_and_does_not_average_undefined_as_zero():
    reports = [_fold_report(1, ["a", "b"]), _fold_report(2, ["c", "d"]), _fold_report(3, ["e"], accepted=False)]
    agg = ev.aggregate_folds(reports)
    f1 = agg["window"]["mean_std"]["macro_f1"]
    assert f1["n_folds_defined"] == 2 and f1["mean"] == pytest.approx(reports[0]["window"]["metrics"]["macro_f1"])
    assert agg["window"]["n_folds_valid"] == 2 and agg["window"]["std_ddof"] == 1
    assert agg["complete_five_fold"] is False and any("fold 3" in r for r in agg["scope_limitations"])
    pooled = np.array(agg["window"]["pooled"]["confusion_matrix"]["counts"])
    expected = sum(np.array(r["window"]["confusion_matrix"]["counts"]) for r in reports[:2])
    assert (pooled == expected).all()
    assert agg["window"]["pooled"]["denominators"]["accepted"] == 60


def test_aggregate_never_claims_complete_five_fold_with_missing_class_fold():
    low_only = ev.evaluate([1] * 6, _probs([1] * 6), subject_ids=["z"] * 6, video_ids=["zv"] * 6, fold=5,
                           min_video_coverage=0.5, n_boot=10)
    full = [_fold_report(i, [f"s{i}a", f"s{i}b"]) for i in range(1, 5)]
    agg = ev.aggregate_folds(full + [low_only])
    assert agg["n_folds_reported"] == 5 and agg["complete_five_fold"] is False
    assert any("fold 5" in r for r in agg["scope_limitations"])
    assert ev.aggregate_folds(full + [_fold_report(5, ["s5a", "s5b"])])["complete_five_fold"] is True


def test_aggregate_rejects_duplicate_folds_and_subjects_tested_twice():
    with pytest.raises(ValueError):
        ev.aggregate_folds([_fold_report(1, ["a"]), _fold_report(1, ["b"])])
    with pytest.raises(ValueError):
        ev.aggregate_folds([_fold_report(1, ["a"]), _fold_report(2, ["a"])])


# ----------------------------------------------------------------- artifacts
def test_write_run_artifacts_creates_strict_json_png_and_logs(tmp_path):
    report = _fold_report(1, ["a", "b"])
    out = ev.write_run_artifacts(tmp_path / "run", report, command="python -m scripts.evaluate --x",
                                 provenance={"seed": 1}, confusion_report=report["window"])
    loaded = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert loaded["window"]["metrics"]["accuracy"] == report["window"]["metrics"]["accuracy"]
    assert (tmp_path / "run" / "confusion_matrix.png").stat().st_size > 0
    assert "--x" in (tmp_path / "run" / "command.log").read_text()
    assert set(out) >= {"metrics", "confusion_matrix", "command", "provenance"}


def test_predictions_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    df = pd.DataFrame({"subject_id": ["a"], "label": [1], "p_low": [0.9]})
    ev.write_run_artifacts(tmp_path / "run", {"x": 1}, predictions=df)
    assert pd.read_parquet(tmp_path / "run" / "predictions.parquet").equals(df)


# ----------------------------------------------------------------- CLI
def _write_predictions_csv(path, folds=(1, 2)):
    import pandas as pd

    frames = []
    for f in folds:
        rows = _fold_rows(subjects=(f"f{f}a", f"f{f}b"))
        probs = _probs(rows["preds"])
        frames.append(pd.DataFrame({"fold": f, "subject_id": rows["subject_ids"], "video_id": rows["video_ids"],
                                    "label": rows["labels"], "p_alert": probs[:, 0], "p_low": probs[:, 1],
                                    "p_drowsy": probs[:, 2]}))
    pd.concat(frames).to_csv(path, index=False)


def test_cli_reports_missing_columns_and_files_without_traceback(tmp_path, capsys=None):
    from scripts import evaluate as cli

    bad = tmp_path / "bad.csv"
    bad.write_text("subject_id,label\na,0\n")
    assert cli.main(["--predictions", str(bad), "--run-dir", str(tmp_path / "r"), "--min-video-coverage", "0.5"]) == 2
    assert cli.main(["--predictions", str(tmp_path / "nope.csv"), "--run-dir", str(tmp_path / "r"),
                     "--min-video-coverage", "0.5"]) == 2
    assert not (tmp_path / "r" / "metrics.json").exists()  # no fake metrics on failure


def test_cli_end_to_end_writes_run_directory(tmp_path):
    pytest.importorskip("pyarrow")
    from scripts import evaluate as cli

    preds = tmp_path / "preds.csv"
    _write_predictions_csv(preds)
    run = tmp_path / "runs" / "x"
    code = cli.main(["--predictions", str(preds), "--run-dir", str(run), "--min-video-coverage", "0.5",
                     "--n-boot", "20", "--expected-folds", "2"])
    assert code == 0
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["folds"]) == {"1", "2"} and metrics["aggregate"]["complete_five_fold"] is False
    for name in ("predictions.parquet", "confusion_matrix.png", "command.log", "provenance.json"):
        assert (run / name).exists()


def test_cli_default_min_video_coverage_is_half():
    from scripts import evaluate as cli

    assert cli.build_parser().parse_args(["--predictions", "p.csv", "--run-dir", "r"]).min_video_coverage == 0.5
