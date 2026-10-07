"""Integration coverage for realtime replay and evaluation workflows."""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.contracts import (
    DetectionResult,
    DriverState,
    FeatureSample,
    FramePacket,
    Prediction,
    SystemStatus,
)
from src.evaluation.replay import ReplayRecord, compare_replays, replay_session
from scripts import external_test as ext
from scripts import run_ablation as abl
from src.config import PROJECT_ROOT
from src.evaluation import evaluator as ev


class Detector:
    def __init__(self):
        self.times = []
        self.resets = 0
        self.last_feature_sample = None

    def reset_session(self):
        self.resets += 1
        self.last_feature_sample = None

    def process(self, packet, *, current_time_ms=None):
        self.times.append(current_time_ms)
        valid = packet.frame_index != 1
        value = 0.3 if valid else np.nan
        self.last_feature_sample = FeatureSample(
            packet.timestamp_ms,
            packet.frame_index,
            packet.source_id,
            value,
            value,
            value,
            0.1 if valid else np.nan,
            0.0 if valid else np.nan,
            0.0 if valid else np.nan,
            0.0 if valid else np.nan,
            valid,
            valid,
            valid,
            valid,
            valid,
            0.0 if valid else np.nan,
        )
        if packet.frame_index == 1:
            return DetectionResult(None, None, SystemStatus.NO_FACE, {}, "COMPLETE")
        probabilities = np.array([.2, .3, .5], dtype=np.float32)
        raw = Prediction(packet.timestamp_ms, probabilities, DriverState.DROWSY,
                         True, "", "model")
        return DetectionResult(raw, None, SystemStatus.READY, {}, "COMPLETE")


def packets():
    return [
        FramePacket(np.zeros((2, 2, 3), dtype=np.uint8), timestamp, index, "video")
        for index, timestamp in enumerate((0, 67, 133))
    ]


def test_replay_uses_source_time_and_preserves_unavailable_timestamps():
    detector = Detector()
    records = replay_session(packets(), detector)
    assert detector.resets == 1
    assert detector.times == [0, 67, 133]
    assert [record.timestamp_ms for record in records] == [0, 67, 133]
    assert records[1].system_status == "NO_FACE"
    assert records[1].feature_values == (None,) * 8
    assert records[1].feature_validity == (False,) * 5
    assert records[1].raw_probabilities is None
    assert records[2].raw_class_id == 2


def test_replay_rejects_source_join_and_nonmonotonic_input():
    source_change = packets()
    source_change[1] = replace(source_change[1], source_id="other")
    with pytest.raises(ValueError, match="different sources"):
        replay_session(source_change, Detector())
    backwards = packets()
    backwards[2] = replace(backwards[2], timestamp_ms=50)
    with pytest.raises(ValueError, match="strictly increase"):
        replay_session(backwards, Detector())


@pytest.mark.parametrize("bad_prediction", [
    Prediction(134, np.array([.2, .3, .5], dtype=np.float32), DriverState.DROWSY, True, "", "model"),
    Prediction(133, np.array([.2, .2, .2], dtype=np.float32), DriverState.ALERT, True, "", "model"),
])
def test_replay_rejects_future_or_malformed_predictions(bad_prediction):
    class BadDetector(Detector):
        def process(self, packet, *, current_time_ms=None):
            return DetectionResult(bad_prediction, None, SystemStatus.READY, {}, "COMPLETE")

    with pytest.raises(ValueError):
        replay_session([packets()[-1]], BadDetector())


def record(timestamp, probabilities=(.2, .3, .5)):
    return ReplayRecord(
        source_id="video",
        frame_index=timestamp,
        timestamp_ms=timestamp,
        system_status="READY",
        calibration_status="COMPLETE",
        feature_values=(0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0),
        feature_validity=(True, True, True, True, True),
        raw_timestamp_ms=timestamp,
        raw_model_id="model",
        raw_probabilities=probabilities,
        raw_class_id=2,
        smoothed_timestamp_ms=None,
        smoothed_model_id=None,
        smoothed_probabilities=None,
        smoothed_class_id=None,
    )


def test_parity_accepts_declared_numeric_tolerance():
    expected = [record(0)]
    actual = [replace(
        record(0, (.2000004, .2999998, .4999998)),
        feature_values=(.3000004, .3, .3, .1, 0., 0., 0., 0.),
    )]
    report = compare_replays(expected, actual, atol=1e-6)
    assert report["match"]
    assert report["mismatches"] == []


def test_parity_reports_missing_timestamp_and_probability_difference():
    expected = [record(0), record(100)]
    actual = [record(0, (.1, .4, .5))]
    report = compare_replays(expected, actual, atol=1e-6)
    assert not report["match"]
    fields = [item["field"] for item in report["mismatches"]]
    assert "record_count" in fields
    assert "raw_probabilities" in fields
    assert "missing_record" in fields


def test_parity_rejects_prediction_cadence_or_model_identity_drift():
    expected = [record(100)]
    actual = [replace(record(100), raw_timestamp_ms=99, raw_model_id="other")]
    report = compare_replays(expected, actual)
    assert not report["match"]
    fields = [item["field"] for item in report["mismatches"]]
    assert "raw_timestamp_ms" in fields
    assert "raw_model_id" in fields


def test_parity_reports_feature_value_and_validity_drift():
    expected = [record(100)]
    actual = [replace(
        record(100),
        feature_values=(.2, .3, .3, .1, 0., 0., 0., 0.),
        feature_validity=(True, False, True, True, True),
    )]
    report = compare_replays(expected, actual)
    fields = [item["field"] for item in report["mismatches"]]
    assert "feature_values" in fields
    assert "feature_validity" in fields


def test_replay_rejects_feature_trace_from_another_packet():
    class MisalignedDetector(Detector):
        def process(self, packet, *, current_time_ms=None):
            result = super().process(packet, current_time_ms=current_time_ms)
            self.last_feature_sample = replace(self.last_feature_sample, frame_index=99)
            return result

    with pytest.raises(ValueError, match="packet identity"):
        replay_session([packets()[0]], MisalignedDetector())
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
