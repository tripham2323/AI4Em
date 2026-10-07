"""Real recurrent inference, trusted serialization and provenance regressions."""
import hashlib
import json

import numpy as np
import pytest
import torch

from src.models.bundle import ModelBundle
from src.datasets.normalization import FEATURE_NAMES, scaler_hash
from src.models.lstm import LSTMClassifier


NAMES = FEATURE_NAMES
HASHES = {name: hashlib.sha256(name.encode()).hexdigest() for name in
          ("snapshot", "split", "profile_policy", "derived", "scaler", "temporal")}


def bundle_inputs():
    model = LSTMClassifier(NAMES, hidden_size=8, head_dropout=0.)
    scaler = {"feature_names": list(NAMES), "schema_version": "temporal_v1", "mode": "P1",
              "means": [0.] * 10, "scales": [1.] * 10, "counts": [10] * 10}
    scaler["hash"] = scaler_hash(scaler)
    metadata = {"mode": "P1", "schema_version": "temporal_v1",
                "feature_names": list(NAMES), "class_order": [0, 1, 2],
                "hashes": {**HASHES, "scaler": scaler["hash"]}}
    return model, scaler, metadata


def test_bundle_rejects_cross_protocol_scaler_on_save(tmp_path):
    model, scaler, metadata = bundle_inputs()
    scaler["mode"] = "P0"
    scaler["hash"] = scaler_hash(scaler)
    metadata["hashes"]["scaler"] = scaler["hash"]
    with pytest.raises(ValueError, match="mode"):
        ModelBundle.save(tmp_path / "mixed", model=model, scaler=scaler, metadata=metadata)


def test_bundle_load_rejects_mode_mismatch_with_matching_integrity(tmp_path):
    model, scaler, metadata = bundle_inputs()
    path = ModelBundle.save(tmp_path / "bundle", model=model, scaler=scaler, metadata=metadata)
    file = path / "metadata.json"
    data = json.loads(file.read_text())
    data["mode"] = "P0"
    file.write_text(json.dumps(data))
    integrity_path = path / "integrity.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["sha256"]["metadata.json"] = hashlib.sha256(file.read_bytes()).hexdigest()
    integrity_path.write_text(json.dumps(integrity))
    with pytest.raises(ValueError, match="mode"):
        ModelBundle.load(path, trusted=True)


def test_real_forward_backward_and_window_state_reset():
    torch.manual_seed(42)
    model = LSTMClassifier(NAMES, hidden_size=8, head_dropout=0.)
    x = torch.randn(3, 100, 16)
    logits = model(x)
    assert logits.shape == (3, 3)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        first = model(x[:1]).clone()
        model(x[1:])
        repeated = model(x[:1])
    torch.testing.assert_close(first, repeated, rtol=0., atol=0.)


def test_bundle_cpu_cold_reload_and_prediction_parity(tmp_path):
    model, scaler, metadata = bundle_inputs()
    model.eval()
    x = torch.randn(4, 100, 16)
    with torch.no_grad():
        expected = torch.softmax(model(x), dim=1).numpy()
    path = ModelBundle.save(tmp_path / "P1_bundle", model=model, scaler=scaler, metadata=metadata)
    with pytest.raises(ValueError, match="trusted"):
        ModelBundle.load(path)
    restored = ModelBundle.load(path, trusted=True)
    restored.validate_schema(NAMES, metadata["hashes"])
    assert restored.feature_names == NAMES
    assert restored.schema_version == "temporal_v1"
    assert restored.calibration_mode == "P1"
    assert len(restored.model_id) == 64
    np.testing.assert_allclose(restored.predict_proba(x), expected, rtol=1e-6, atol=1e-7)
    assert next(restored.model.parameters()).device.type == "cpu"
    assert restored.scaler == scaler
    assert restored.metadata["mode"] == "P1"
    assert restored.predict_proba(np.empty((0, 100, 16), dtype=np.float32)).shape == (0, 3)


def test_bundle_rejects_wrong_mode_schema_feature_order_and_identity(tmp_path):
    model, scaler, metadata = bundle_inputs()
    with pytest.raises(ValueError, match="mode"):
        ModelBundle.save(tmp_path / "bad", model=model, scaler=scaler,
                         metadata={**metadata, "mode": "P2"})
    with pytest.raises(ValueError, match="schema"):
        ModelBundle.save(tmp_path / "bad", model=model,
                         scaler={**scaler, "schema_version": "other"}, metadata=metadata)
    path = ModelBundle.save(tmp_path / "valid", model=model, scaler=scaler, metadata=metadata)
    restored = ModelBundle.load(path, trusted=True)
    with pytest.raises(ValueError, match="order"):
        restored.validate_schema(tuple(reversed(NAMES)), metadata["hashes"])
    with pytest.raises(ValueError, match="split"):
        restored.validate_schema(NAMES, {**metadata["hashes"], "split": "f" * 64})
    with pytest.raises(ValueError, match="mode"):
        restored.validate_schema(NAMES, {**metadata["hashes"], "mode": "P0"})
    with pytest.raises(ValueError, match="schema"):
        restored.validate_schema(NAMES, {**metadata["hashes"], "schema_version": "other"})


@pytest.mark.parametrize("filename", ["weights.pt", "metadata.json", "scaler.json"])
def test_bundle_sha_integrity_catches_modified_payload(tmp_path, filename):
    model, scaler, metadata = bundle_inputs()
    path = ModelBundle.save(tmp_path / "bundle", model=model, scaler=scaler, metadata=metadata)
    file = path / filename
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        ModelBundle.load(path, trusted=True)


def test_bundle_rejects_missing_provenance_and_nonfinite_input(tmp_path):
    model, scaler, metadata = bundle_inputs()
    with pytest.raises(ValueError, match="temporal"):
        ModelBundle.save(tmp_path / "invalid", model=model, scaler=scaler,
                         metadata={**metadata, "hashes": {k: v for k, v in HASHES.items() if k != "temporal"}})
    path = ModelBundle.save(tmp_path / "bundle", model=model, scaler=scaler, metadata=metadata)
    restored = ModelBundle.load(path, trusted=True)
    with pytest.raises(ValueError, match="finite"):
        restored.predict_proba(np.full((1, 100, 16), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="100"):
        restored.predict_proba(np.ones((1, 99, 16), dtype=np.float32))


def test_load_checks_metadata_semantics_even_with_updated_integrity(tmp_path):
    model, scaler, metadata = bundle_inputs()
    path = ModelBundle.save(tmp_path / "bundle", model=model, scaler=scaler, metadata=metadata)
    file = path / "metadata.json"
    data = json.loads(file.read_text())
    data["class_order"] = [2, 1, 0]
    file.write_text(json.dumps(data))
    integrity_path = path / "integrity.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["sha256"]["metadata.json"] = hashlib.sha256(file.read_bytes()).hexdigest()
    integrity_path.write_text(json.dumps(integrity))
    with pytest.raises(ValueError, match="class"):
        ModelBundle.load(path, trusted=True)
