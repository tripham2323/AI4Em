"""Real CPU optimization and validation-only checkpoint selection regressions."""
import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import ModelEvaluator
from src.models.lstm import LSTMClassifier
from src.training.trainer import ModelTrainer


NAMES = tuple(f"feature_{i}" for i in range(16))


def loader(labels, *, shuffle=False):
    labels = torch.tensor(labels, dtype=torch.long)
    x = torch.zeros(len(labels), 100, 16)
    return DataLoader(TensorDataset(x, labels), batch_size=8, shuffle=shuffle, num_workers=0)


def initial_model():
    torch.manual_seed(42)
    model = LSTMClassifier(NAMES, hidden_size=4, head_dropout=0.)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
        last_linear.bias.copy_(torch.tensor([1., 0., 0.]))
    return model


def test_genuine_training_selects_validation_macro_f1_not_decreasing_train_loss():
    model = initial_model()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    train = loader([0] * 12 + [1, 1, 2, 2], shuffle=True)
    val = loader([0, 1, 2, 2, 2, 2])
    result = ModelTrainer(model, train, val, {"max_epochs": 4, "learning_rate": .01,
                                          "early_stopping_patience": 8}).fit()
    history = result["history"]
    assert len(history) == 4
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    # All windows have the same signal, so every epoch predicts class 0. The
    # validation macro F1 is 2/(6+1)/3 = 2/21, independently of train prevalence.
    np.testing.assert_allclose([row["val_macro_f1"] for row in history], 2. / 21.)
    assert result["best_epoch"] == 1
    assert result["best_val_metrics"]["macro_f1"] == pytest.approx(2. / 21.)
    assert all(value.device.type == "cpu" for value in result["state_dict"].values())
    assert any(not torch.equal(initial[key], value) for key, value in result["state_dict"].items())
    restored = initial_model()
    restored.load_state_dict(result["state_dict"])
    restored.eval()
    x, labels = next(iter(val))
    with torch.no_grad():
        probabilities = restored(x).softmax(dim=1).numpy()
    independent = ModelEvaluator().evaluate(labels.numpy(), probabilities,
                                            metadata={"class_order": [0, 1, 2]},
                                            coverage={"scheduled": 6, "accepted": 6, "rejected": 0})
    assert independent["macro_f1"] == result["best_val_metrics"]["macro_f1"]
    assert result["selection_metric"] == "val_macro_f1"
    assert result["config"]["device"] == "cpu"
    assert result["environment"]["seed"] == 42
    assert "test_loader" not in inspect.signature(ModelTrainer).parameters
    before = result["state_dict"][next(iter(result["state_dict"]))].clone()
    with torch.no_grad():
        next(model.parameters()).add_(100.)
    torch.testing.assert_close(result["state_dict"][next(iter(result["state_dict"]))], before)


@pytest.mark.parametrize("labels", [[], [0, 1, 1]])
def test_empty_or_missing_validation_class_blocks_before_optimizer_updates(labels):
    model = initial_model()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="validation"):
        ModelTrainer(model, loader([0, 1, 2]), loader(labels), {"max_epochs": 1}).fit()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key])


def test_training_support_and_nonfinite_inputs_are_not_hidden():
    with pytest.raises(ValueError, match="training"):
        ModelTrainer(initial_model(), loader([0, 1]), loader([0, 1, 2]), {"max_epochs": 1}).fit()
    bad = DataLoader(TensorDataset(torch.full((3, 100, 16), float("nan")),
                                  torch.tensor([0, 1, 2])), batch_size=3)
    with pytest.raises(ValueError, match="finite"):
        ModelTrainer(initial_model(), bad, loader([0, 1, 2]), {"max_epochs": 1}).fit()


def test_optional_class_weights_use_only_training_counts_and_default_is_unweighted():
    train, val = loader([0, 0, 0, 0, 1, 1, 2]), loader([0, 1, 2, 2, 2])
    result = ModelTrainer(initial_model(), train, val,
                          {"max_epochs": 1, "class_weights": True}).fit()
    np.testing.assert_allclose(result["class_weights"], [7. / 12., 7. / 6., 7. / 3.])
    plain = ModelTrainer(initial_model(), train, val, {"max_epochs": 1}).fit()
    assert plain["class_weights"] is None


def test_patience_stops_real_epochs_and_seed_repeats_optimization():
    train, val = loader([0, 0, 1, 2], shuffle=True), loader([0, 1, 2])
    config = {"max_epochs": 10, "early_stopping_patience": 2, "learning_rate": .001}
    first = ModelTrainer(initial_model(), train, val, config).fit()
    second = ModelTrainer(initial_model(), train, val, config).fit()
    assert len(first["history"]) == 3
    assert first["stopped_early"]
    assert first["best_epoch"] == second["best_epoch"]
    for key in first["state_dict"]:
        torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0., atol=0.)


def test_validation_coverage_and_window_metadata_survive_selection():
    rows = [(torch.zeros(100, 16), cls, {"video_id": f"video_{cls}", "timestamp_ms": 1000 + cls})
            for cls in range(3)]
    val = DataLoader(rows, batch_size=3)
    coverage = {"scheduled": 5, "accepted": 3, "rejected": 2}
    result = ModelTrainer(initial_model(), loader([0, 1, 2]), val,
                          {"max_epochs": 1, "val_coverage": coverage}).fit()
    assert result["best_val_metrics"]["coverage"]["acceptance_rate"] == .6
    assert result["best_val_predictions"]["metadata"] == [
        {"video_id": "video_0", "timestamp_ms": 1000},
        {"video_id": "video_1", "timestamp_ms": 1001},
        {"video_id": "video_2", "timestamp_ms": 1002}]
    with pytest.raises(ValueError, match="coverage"):
        ModelTrainer(initial_model(), loader([0, 1, 2]), val,
                     {"max_epochs": 1, "val_coverage": {**coverage, "accepted": 2}}).fit()


def test_validation_shuffling_is_rejected_instead_of_changing_selection_order():
    with pytest.raises(ValueError, match="validation"):
        ModelTrainer(initial_model(), loader([0, 1, 2]), loader([0, 1, 2], shuffle=True),
                     {"max_epochs": 1}).fit()
