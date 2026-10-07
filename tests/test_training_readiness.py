import json
from pathlib import Path

import yaml

from scripts.check_training_readiness import inspect_training_readiness


def _workspace(tmp_path: Path, *, include_source: bool = True) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "raw_features").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "uta_rldd" / "04").mkdir(parents=True)
    (tmp_path / "configs" / "preprocessing.yaml").write_text(
        yaml.safe_dump({"dataset_root": "data/raw/uta_rldd"}), encoding="utf-8"
    )
    (tmp_path / "data" / "splits" / "outer_0.json").write_text(json.dumps({
        "sources": [{"video_id": "04_0", "relative_path": "04/0.mp4"}]
    }), encoding="utf-8")
    if include_source:
        (tmp_path / "data" / "raw" / "uta_rldd" / "04" / "0.mp4").write_bytes(b"video")
    (tmp_path / "data" / "processed" / "raw_features" / "04_0.parquet").write_bytes(b"parquet")
    (tmp_path / "data" / "processed" / "raw_features" / "04_0.metadata.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_readiness_lists_missing_source_and_frozen_reports(tmp_path):
    report = inspect_training_readiness(_workspace(tmp_path, include_source=False))
    assert report["status"] == "blocked"
    assert report["missing_raw_video_ids"] == ["04_0"]
    assert any("phase8" in blocker for blocker in report["blockers"])
    assert any("P0.json" in blocker for blocker in report["blockers"])


def test_readiness_counts_present_source_and_feature_pair(tmp_path):
    report = inspect_training_readiness(_workspace(tmp_path))
    assert report["required_raw_videos"] == 1
    assert report["present_raw_videos"] == 1
    assert report["missing_raw_feature_pair_ids"] == []
