"""Construct the production realtime detector from frozen local artifacts."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.calibration.manager import CalibrationManager
from src.calibration.profile import estimate_profile
from src.config import PROJECT_ROOT, load_config, validate_config
from src.datasets.acquisition import digest_file
from src.datasets.normalization import transform_values
from src.datasets.splits import hash_payload, profile_from_dict
from src.features.pipeline import FeaturePipeline
from src.features.temporal import FEATURE_NAMES, TemporalFeatureExtractor, resolve_temporal_config, temporal_config_hash
from src.models.bundle import ModelBundle
from src.realtime.buffer import PredictionBuffer
from src.realtime.detector import DrowsinessDetector
from src.realtime.smoother import PredictionSmoother


_PATH_KEYS = ("model_path", "preprocessing_config", "temporal_config", "profiles_path")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    """Load realtime YAML and resolve every artifact path from project root."""
    config_path = _project_path(path)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid realtime YAML in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Realtime config must be a YAML mapping")
    for key in _PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Realtime config requires path key: {key}")
        config[key] = _project_path(value)
    validate_config(config)
    return config


def validate_runtime_artifacts(config: Mapping[str, Any], *, require_model: bool = True) -> None:
    """Fail early with a precise startup error before creating Qt/native state."""
    required = {
        "schema_version", "calibration_mode", "model_path", "preprocessing_config", "temporal_config",
        "profiles_path", "camera_index", "camera_width", "camera_height", "ui_refresh_hz",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Realtime config missing keys: {', '.join(missing)}")
    # Fixture mode intentionally exercises UI/thread wiring without requiring
    # ignored production data or model artifacts in a fresh checkout.
    if not require_model:
        return
    for key in ("preprocessing_config", "temporal_config", "profiles_path"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Realtime artifact not found: {config[key]}")
    if not Path(config["model_path"]).is_dir():
        raise FileNotFoundError(f"Model bundle not found: {config['model_path']}")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return value


def _load_split(path: Path) -> dict[str, Any]:
    split = _json(path)
    if split.get("split_hash") != hash_payload({k: v for k, v in split.items() if k != "split_hash"}):
        raise ValueError("Split/profile artifact hash mismatch")
    policy = split.get("qc_policy")
    if not isinstance(policy, dict) or policy.get("policy_hash") != hash_payload(
        {k: v for k, v in policy.items() if k != "policy_hash"}
    ):
        raise ValueError("Profile QC policy hash mismatch")
    return split


def _select_profile(split: Mapping[str, Any], mode: str, image_size: tuple[int, int], asset_sha256: str):
    records = list((split.get("profiles") or {}).get(mode, {}).values())
    if not records:
        raise ValueError(f"Split contains no {mode} calibration profiles")
    profiles = [profile_from_dict(record) for record in records]
    if mode == "P0":
        valid = [profile for profile in profiles if profile.valid]
        if not valid:
            raise ValueError("Split contains no valid P0 population profile")
        profile = valid[0]
        baseline = (
            profile.ear_left_baseline, profile.ear_right_baseline, profile.mar_baseline,
            profile.pitch_baseline, profile.yaw_baseline, profile.roll_baseline,
        )
        for candidate in valid[1:]:
            other = (
                candidate.ear_left_baseline, candidate.ear_right_baseline, candidate.mar_baseline,
                candidate.pitch_baseline, candidate.yaw_baseline, candidate.roll_baseline,
            )
            if candidate.asset_sha256 != profile.asset_sha256 or other != baseline:
                raise ValueError("P0 split contains inconsistent population profiles")
        if profile.asset_sha256 != asset_sha256:
            raise ValueError("P0 profile and current landmarker asset differ")
        return replace(
            profile,
            image_size=image_size,
            quality_stats={**profile.quality_stats, "population_numeric_only": True},
        )
    # The temporal extractor is rebound to the completed live P1 profile before
    # it receives any sample. This frozen record is only an inactive placeholder.
    placeholder = profiles[0]
    return replace(placeholder, image_size=image_size, asset_sha256=asset_sha256)


def build_detector(config: Mapping[str, Any]) -> DrowsinessDetector:
    """Load and cross-check every artifact, then allocate one detector session."""
    validate_runtime_artifacts(config)
    mode = str(config["calibration_mode"])
    bundle = ModelBundle.load(Path(config["model_path"]), trusted=True)
    if bundle.calibration_mode != mode:
        raise ValueError("Realtime calibration mode differs from model bundle")
    if bundle.schema_version != config["schema_version"]:
        raise ValueError("Realtime schema differs from model bundle")
    if tuple(bundle.feature_names) != FEATURE_NAMES:
        raise ValueError("Realtime production requires the complete ordered temporal feature schema")

    preprocessing = load_config(Path(config["preprocessing_config"]))
    temporal_raw = yaml.safe_load(Path(config["temporal_config"]).read_text(encoding="utf-8"))
    if not isinstance(temporal_raw, dict):
        raise ValueError("Temporal config must be a YAML mapping")
    temporal_config = resolve_temporal_config(temporal_raw)
    split = _load_split(Path(config["profiles_path"]))

    hashes = bundle.metadata["hashes"]
    if hashes["split"] != split["split_hash"]:
        raise ValueError("Model bundle and split/profile artifact differ")
    if hashes["profile_policy"] != split["qc_policy"]["policy_hash"]:
        raise ValueError("Model bundle and profile QC policy differ")
    if hashes["temporal"] != temporal_config_hash(temporal_config):
        raise ValueError("Model bundle and temporal policy differ")

    image_size = (int(config["camera_width"]), int(config["camera_height"]))
    asset_sha256 = digest_file(Path(preprocessing["asset_path"]))[0]
    seed_profile = _select_profile(split, mode, image_size, asset_sha256)
    calibration_kwargs: dict[str, Any] = dict(
        mode=mode,
        schema_version=bundle.schema_version,
        asset_sha256=asset_sha256,
        image_size=image_size,
        calibration_seconds=config.get("calibration_seconds", 30),
        min_valid_seconds=config.get("min_valid_seconds", 20),
        timeout_seconds=config.get("timeout_seconds", 60),
        max_sample_age_ms=int(temporal_config["max_sample_age_ms"]),
    )
    if mode == "P0":
        calibration_kwargs["population_profile"] = seed_profile
    else:
        calibration_kwargs.update(
            profile_estimator=estimate_profile,
            profile_estimator_kwargs={"qc_policy": split["qc_policy"]},
        )
    calibration = CalibrationManager(**calibration_kwargs)

    scaler = bundle.scaler

    def transform(values: np.ndarray, validity: np.ndarray) -> np.ndarray:
        return transform_values(values[np.newaxis, :], validity[np.newaxis, :], scaler)[0]

    timing = bundle.metadata.get("timing") or {}
    steps = int(timing.get("steps", 100))
    if steps != 100:
        raise ValueError("Realtime LSTM bundle must use the supported 100-step sequence")
    buffer = PredictionBuffer(
        feature_names=bundle.feature_names,
        schema_version=bundle.schema_version,
        sequence_steps=steps,
        max_missing_ratio=float(timing.get("max_missing_ratio", config.get("max_missing_ratio", 0.20))),
        max_gap_ms=round(float(timing.get("max_gap_s", config.get("max_gap_s", 1))) * 1000),
        transformer=transform,
    )
    smoother = PredictionSmoother(
        samples=int(config.get("smoothing_samples", 3)),
        expiry_ms=round(float(config.get("smoothing_expiry_s", 2)) * 1000),
    )
    return DrowsinessDetector(
        pipeline=FeaturePipeline(preprocessing),
        calibration=calibration,
        temporal=TemporalFeatureExtractor(seed_profile, temporal_config),
        buffer=buffer,
        model=bundle,
        smoother=smoother,
        prediction_interval_ms=round(float(config.get("prediction_interval_s", 1)) * 1000),
        stale_ms=int(config.get("stale_ms", 500)),
        max_gap_ms=round(float(config.get("max_gap_s", 1)) * 1000),
    )


def restart_calibration(detector: DrowsinessDetector) -> None:
    """Explicit retry requested by the UI; all causal session state is reset."""
    detector.reset_session()
