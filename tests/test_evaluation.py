"""Independent known-answer and abstention tests for the metrics core."""
import json
import numpy as np
import pytest

from src.evaluation.evaluator import ModelEvaluator


def evaluate(labels, predictions, scheduled=None):
    accepted = len(labels)
    scheduled = accepted if scheduled is None else scheduled
    probabilities = np.eye(3)[predictions] if accepted else np.empty((0, 3))
    return ModelEvaluator().evaluate(labels, probabilities, metadata={"class_order": [0, 1, 2]},
                                    coverage={"accepted": accepted, "scheduled": scheduled,
                                              "rejected": scheduled - accepted})


def test_team01_known_answer_is_not_sklearn_generated_fixture():
    report = evaluate([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["macro_f1"] == pytest.approx(59 / 90)
    assert report["weighted_f1"] == pytest.approx(59 / 90)
    assert report["balanced_accuracy"] == pytest.approx(2 / 3)
    assert [item["recall"] for item in report["per_class"]] == [.5, 1., .5]
    assert [item["support"] for item in report["per_class"]] == [2, 2, 2]
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    json.dumps(report, allow_nan=False)


def test_absent_true_class_is_not_full_macro_and_abstentions_not_labels():
    report = evaluate([0, 0], [0, 1], scheduled=5)
    assert report["accuracy"] == .5
    assert report["macro_f1"] is None
    assert report["supported_macro_f1"] == pytest.approx(2 / 3)
    assert report["supported_macro_f1_denominator"] == 1
    assert report["per_class"][1]["recall"] is None
    assert report["per_class"][2]["precision"] is None
    assert report["per_class"][2]["f1"] is None
    assert "macro_f1" in report["undefined_reasons"]
    assert report["coverage"]["acceptance_rate"] == .4
    assert report["coverage"]["rejected"] == 3
    assert sum(map(sum, report["confusion_matrix"])) == 2


def test_no_accepted_support_and_zero_scheduled_have_explicit_null_reasons():
    report = evaluate([], [], scheduled=4)
    for metric in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "supported_macro_f1"]:
        assert report[metric] is None
        assert metric in report["undefined_reasons"]
    assert report["coverage"]["acceptance_rate"] == 0.
    empty = evaluate([], [])
    assert empty["coverage"]["acceptance_rate"] is None
    assert "coverage.acceptance_rate" in empty["undefined_reasons"]
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize("probabilities", [np.ones((2, 2)), [[.5, .5, .5], [0., 0., 1.]],
    [[np.nan, 0., 1.], [0., 1., 0.]], [[-.1, .1, 1.], [0., 1., 0.]]])
def test_invalid_probabilities_rejected_before_argmax(probabilities):
    with pytest.raises(ValueError):
        ModelEvaluator().evaluate([0, 1], probabilities, metadata={},
                                  coverage={"scheduled": 2, "accepted": 2, "rejected": 0})


def test_class_order_labels_and_coverage_validated():
    evaluator = ModelEvaluator()
    with pytest.raises(ValueError, match="order"):
        evaluator.evaluate([0], [[1., 0., 0.]], metadata={"class_order": [2, 1, 0]},
                           coverage={"scheduled": 1, "accepted": 1, "rejected": 0})
    with pytest.raises(ValueError):
        evaluator.evaluate([3], [[1., 0., 0.]], metadata={},
                           coverage={"scheduled": 1, "accepted": 1, "rejected": 0})
    with pytest.raises(ValueError):
        evaluator.evaluate([0], [[1., 0., 0.]], metadata={},
                           coverage={"scheduled": 4, "accepted": 2, "rejected": 2})
    report = evaluator.evaluate([0], [[.9999999, 0., 0.]], metadata={},
                               coverage={"scheduled": 1, "accepted": 1, "rejected": 0})
    assert report["accuracy"] == 1.
