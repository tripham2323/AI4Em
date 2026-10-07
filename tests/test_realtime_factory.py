from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.features.temporal import FEATURE_NAMES
from src.realtime.factory import build_detector, load_runtime_config, validate_runtime_artifacts


def _config(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    artifacts = {}
    for name in ("preprocessing.yaml", "temporal.yaml", "outer_0.json"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        artifacts[name] = path
    path = tmp_path / "realtime.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "facial_features_v1",
        "camera_index": 0,
        "camera_width": 640,
        "camera_height": 480,
        "calibration_mode": "P0",
        "model_path": str(model),
        "preprocessing_config": str(artifacts["preprocessing.yaml"]),
        "temporal_config": str(artifacts["temporal.yaml"]),
        "profiles_path": str(artifacts["outer_0.json"]),
        "ui_refresh_hz": 10,
    }), encoding="utf-8")
    return path


def test_runtime_config_resolves_and_preflights_artifacts(tmp_path):
    config = load_runtime_config(_config(tmp_path))
    assert all(Path(config[key]).is_absolute() for key in (
        "model_path", "preprocessing_config", "temporal_config", "profiles_path"
    ))
    validate_runtime_artifacts(config)


def test_runtime_preflight_reports_missing_model_bundle(tmp_path):
    config = load_runtime_config(_config(tmp_path))
    Path(config["model_path"]).rmdir()
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        validate_runtime_artifacts(config)
    validate_runtime_artifacts(config, require_model=False)


def test_fixture_preflight_does_not_require_ignored_runtime_artifacts(tmp_path):
    config = load_runtime_config(_config(tmp_path))
    Path(config["model_path"]).rmdir()
    for key in ("preprocessing_config", "temporal_config", "profiles_path"):
        Path(config[key]).unlink()
    validate_runtime_artifacts(config, require_model=False)


def test_runtime_config_requires_all_artifact_paths(tmp_path):
    path = _config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    del config["profiles_path"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="profiles_path"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("bundle_override", "message"),
    [
        ({"schema_version": "other"}, "schema differs"),
        ({"feature_names": FEATURE_NAMES[:-1]}, "complete ordered temporal feature schema"),
    ],
)
def test_detector_factory_rejects_incompatible_bundle_before_pipeline_allocation(
    tmp_path, monkeypatch, bundle_override, message,
):
    config = load_runtime_config(_config(tmp_path))
    bundle = SimpleNamespace(
        calibration_mode="P0",
        schema_version="facial_features_v1",
        feature_names=FEATURE_NAMES,
    )
    for key, value in bundle_override.items():
        setattr(bundle, key, value)
    monkeypatch.setattr("src.realtime.factory.ModelBundle.load", lambda *args, **kwargs: bundle)
    with pytest.raises(ValueError, match=message):
        build_detector(config)
