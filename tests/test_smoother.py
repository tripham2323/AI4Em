import numpy as np
import pytest

from src.contracts import DriverState, Prediction
from src.realtime.smoother import PredictionSmoother


def prediction(timestamp, probabilities, *, model_id="model", valid=True):
    values = np.asarray(probabilities, dtype=np.float32)
    class_id = DriverState(int(np.argmax(values))) if valid and np.isfinite(values).all() else None
    return Prediction(timestamp, values, class_id, valid, "", model_id)


def test_returns_null_until_three_then_means_probability_vectors():
    smoother = PredictionSmoother(samples=3, expiry_ms=2000)
    assert smoother.update(prediction(0, [.7, .2, .1])) is None
    assert smoother.update(prediction(1000, [.5, .35, .15])) is None
    result = smoother.update(prediction(1999, [.4, .4, .2]))
    assert result is not None
    np.testing.assert_allclose(result.probabilities, [8 / 15, 19 / 60, 3 / 20], atol=1e-6)
    assert result.class_id is DriverState.ALERT
    assert result.timestamp_ms == 1999


def test_gap_exactly_expiry_clears_history():
    smoother = PredictionSmoother(samples=3, expiry_ms=2000)
    smoother.update(prediction(0, [.7, .2, .1]))
    smoother.update(prediction(1000, [.6, .3, .1]))
    assert smoother.update(prediction(3000, [.2, .3, .5])) is None
    assert len(smoother) == 1


def test_gap_just_before_expiry_keeps_history():
    smoother = PredictionSmoother(samples=3, expiry_ms=2000)
    smoother.update(prediction(0, [.7, .2, .1]))
    smoother.update(prediction(1000, [.6, .3, .1]))
    assert smoother.update(prediction(2999, [.2, .3, .5])) is not None


@pytest.mark.parametrize("bad", [
    prediction(0, [np.nan, .5, .5]),
    prediction(0, [.2, .8]),
    prediction(0, [.2, .2, .2]),
    prediction(0, [-.1, .6, .5]),
    prediction(0, [.2, .3, .5], valid=False),
])
def test_invalid_predictions_are_rejected(bad):
    with pytest.raises(ValueError):
        PredictionSmoother().update(bad)


def test_reset_and_model_identity_protect_sessions():
    smoother = PredictionSmoother()
    smoother.update(prediction(0, [.7, .2, .1]))
    with pytest.raises(ValueError, match="different models"):
        smoother.update(prediction(1000, [.7, .2, .1], model_id="other"))
    smoother.reset()
    assert smoother.update(prediction(0, [.7, .2, .1], model_id="other")) is None


def test_history_owns_probability_bytes_and_rejects_wrong_class():
    smoother = PredictionSmoother()
    first = prediction(0, [.7, .2, .1])
    smoother.update(first)
    first.probabilities[:] = [0., 0., 1.]
    smoother.update(prediction(1000, [.7, .2, .1]))
    result = smoother.update(prediction(1999, [.7, .2, .1]))
    assert result is not None
    np.testing.assert_allclose(result.probabilities, [.7, .2, .1], atol=1e-6)

    wrong = Prediction(3000, np.array([.7, .2, .1], dtype=np.float32),
                       DriverState.DROWSY, True, "", "model")
    with pytest.raises(ValueError, match="class_id"):
        PredictionSmoother().update(wrong)
