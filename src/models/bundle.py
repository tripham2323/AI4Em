"""CPU-loadable state_dict bundle with JSON provenance and SHA256 integrity.

A bundle is a directory containing weights.pt, metadata.json, scaler.json and
integrity.json. SHA256 detects accidental modification, not malicious replacement
of both payload and integrity file; explicit local trust is still required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch

from src.models.lstm import LSTMClassifier


_REQUIRED_HASHES = ("snapshot", "split", "profile_policy", "derived", "scaler", "temporal")
_PAYLOADS = ("weights.pt", "metadata.json", "scaler.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value) -> dict:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Bundle metadata/scaler must contain finite JSON values") from exc
    if not isinstance(result, dict):
        raise ValueError("Bundle metadata/scaler must be JSON mappings")
    return result


def _validate_metadata(metadata: dict, scaler: dict) -> None:
    if metadata.get("mode") not in ("P0", "P1"):
        raise ValueError("Bundle mode must be explicitly P0 or P1")
    if scaler.get("mode") != metadata["mode"]:
        raise ValueError("Scaler mode differs from bundle mode")
    schema = metadata.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ValueError("Bundle schema_version must be a nonempty string")
    if scaler.get("schema_version", schema) != schema:
        raise ValueError("Scaler schema_version differs from model schema")
    names = metadata.get("feature_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError("Bundle feature names/order must be nonempty unique strings")
    if metadata.get("class_order") != [0, 1, 2]:
        raise ValueError("Bundle class order must be [0,1,2]")
    hashes = metadata.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("Bundle hashes must be a mapping")
    for key in _REQUIRED_HASHES:
        value = hashes.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Bundle requires a lowercase SHA256 {key} identity")
    from src.datasets.normalization import validate_scaler
    validate_scaler(scaler, tuple(names))
    if hashes["scaler"] != scaler.get("hash"):
        raise ValueError("Bundle scaler identity differs from persisted scaler hash")
    if not isinstance(metadata.get("architecture"), dict):
        raise ValueError("Bundle architecture must be a mapping")


class ModelBundle:
    """A frozen classifier plus its train-fitted scaler and resolved metadata.

    predict_proba takes already scaled windows, never fits or reapplies scaling.
    validate_schema requires the complete six-hash consumer identity mapping;
    optional mode/schema_version entries additionally enforce those identities.
    """

    def __init__(self, model: LSTMClassifier, scaler: dict, metadata: dict, path: Path):
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.path = path

    @property
    def model_id(self) -> str:
        """Stable prediction identity tied to the serialized weights."""
        configured = self.metadata.get("model_id")
        if isinstance(configured, str) and configured:
            return configured
        weights = self.path / "weights.pt"
        if not weights.is_file():
            raise ValueError("Bundle weights are unavailable for model identity")
        return _sha256(weights)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_names"])

    @property
    def schema_version(self) -> str:
        return str(self.metadata["schema_version"])

    @property
    def calibration_mode(self) -> str:
        return str(self.metadata["mode"])

    @classmethod
    def save(cls, path: str | Path, *, model: LSTMClassifier, scaler: dict,
             metadata: dict) -> Path:
        if not isinstance(model, LSTMClassifier):
            raise ValueError("Bundle model must be an LSTMClassifier")
        metadata = _json_copy(metadata)
        scaler = _json_copy(scaler)
        supplied_architecture = metadata.get("architecture", model.architecture)
        if supplied_architecture != model.architecture:
            raise ValueError("Bundle architecture differs from model architecture")
        metadata["architecture"] = model.architecture
        _validate_metadata(metadata, scaler)
        if tuple(metadata["feature_names"]) != model.feature_names:
            raise ValueError("Bundle feature order differs from model feature order")
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Model weights must be finite")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save(state, path / "weights.pt")
        for filename, data in (("metadata.json", metadata), ("scaler.json", scaler)):
            (path / filename).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        integrity = {"format_version": 1, "sha256": {name: _sha256(path / name) for name in _PAYLOADS}}
        (path / "integrity.json").write_text(json.dumps(integrity, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> ModelBundle:
        if trusted is not True:
            raise ValueError("Loading a local bundle requires trusted=True")
        path = Path(path)
        try:
            integrity = json.loads((path / "integrity.json").read_text(encoding="utf-8"))
            if not isinstance(integrity, dict) or integrity.get("format_version") != 1 or set(integrity.get("sha256", {})) != set(_PAYLOADS):
                raise ValueError("Invalid bundle integrity manifest")
            for filename in _PAYLOADS:
                if _sha256(path / filename) != integrity["sha256"][filename]:
                    raise ValueError(f"Bundle integrity mismatch for {filename}")
            metadata = _json_copy(json.loads((path / "metadata.json").read_text(encoding="utf-8")))
            scaler = _json_copy(json.loads((path / "scaler.json").read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Cannot read bundle integrity/JSON payloads") from exc
        _validate_metadata(metadata, scaler)
        architecture = metadata["architecture"]
        try:
            model = LSTMClassifier(tuple(metadata["feature_names"]),
                                   hidden_size=architecture["hidden_size"],
                                   num_layers=architecture["num_layers"],
                                   head_dropout=architecture["head_dropout"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid bundle architecture") from exc
        if architecture != model.architecture:
            raise ValueError("Bundle architecture/schema is incompatible")
        # No pickled module or arbitrary globals are allowed, even for local data.
        try:
            state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
            if not isinstance(state, Mapping) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
                raise ValueError("Bundle weights must be a tensor state_dict")
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Bundle weights must be finite")
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, KeyError) as exc:
            raise ValueError("Bundle weights are incompatible with architecture") from exc
        model.cpu().eval()
        return cls(model, scaler, metadata, path)

    def validate_schema(self, names: tuple[str, ...], hashes: dict) -> None:
        if tuple(names) != tuple(self.metadata["feature_names"]):
            raise ValueError("Consumer feature names/order differ from bundle")
        if not isinstance(hashes, Mapping):
            raise ValueError("Consumer hashes must be a mapping")
        for key in _REQUIRED_HASHES:
            if key not in hashes:
                raise ValueError(f"Consumer identity missing {key}")
        for key, value in hashes.items():
            actual = self.metadata.get(key) if key in ("mode", "schema_version") else self.metadata["hashes"].get(key)
            if value != actual:
                raise ValueError(f"Consumer {key} identity differs from bundle")

    def predict_proba(self, scaled_x_np_or_tensor) -> np.ndarray:
        try:
            x = torch.as_tensor(scaled_x_np_or_tensor, dtype=torch.float32, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Scaled windows must be numeric tensors or arrays") from exc
        if x.ndim != 3 or x.shape[1:] != (100, len(self.model.feature_names)):
            raise ValueError(f"Expected scaled [N,100,{len(self.model.feature_names)}] windows")
        if not torch.isfinite(x).all():
            raise ValueError("Scaled windows must contain finite values")
        if len(x) == 0:
            return np.empty((0, 3), dtype=np.float32)
        self.model.eval()
        with torch.inference_mode():
            probabilities = self.model(x).softmax(dim=1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Model produced nonfinite probabilities")
        return probabilities.numpy()
