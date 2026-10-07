"""Rule boundaries remain diagnostic one-hot outputs, with explicit abstention."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.contracts import DriverState, TemporalSample
from src.features.temporal import FEATURE_NAMES
from src.models.rules import RuleBasedClassifier


def config():
    return yaml.safe_load((Path(__file__).parents[1] / 'configs/temporal.yaml').read_text())


def sample(*, perclos=None, closure=0., absolute=.3, absolute_duration=0., valid=True, face=True):
    values = np.array([1., 1., 0., 0., 0., 0., np.nan, closure, 0., 0.,
                       float(valid), float(valid), float(valid), float(valid), 0., 1.], dtype=np.float32)
    if perclos is not None:
        values[FEATURE_NAMES.index('perclos_60')] = perclos
        values[FEATURE_NAMES.index('perclos_ready')] = 1.
    validity = np.isfinite(values)
    if not valid:
        values[:10] = np.nan
        validity[:10] = False
    return TemporalSample(1000, values, validity, 0,
                          {'absolute_ear_mean': absolute, 'absolute_closure_elapsed_s': absolute_duration,
                           'face_detected': face, 'profile_valid': True})


@pytest.mark.parametrize('perclos,state', [(0., DriverState.ALERT), (.149, DriverState.ALERT),
                                         (.15, DriverState.LOW_VIGILANCE), (.249, DriverState.LOW_VIGILANCE),
                                         (.25, DriverState.DROWSY)])
def test_temporal_ready_perclos_boundaries_and_one_hot_diagnostic(perclos, state):
    prediction = RuleBasedClassifier(config(), mode='temporal').predict(sample(perclos=perclos))
    assert prediction.valid and prediction.class_id == state
    np.testing.assert_array_equal(prediction.probabilities, np.eye(3, dtype=np.float32)[int(state)])
    assert 'diagnostic' in prediction.reason and 'not_calibrated' in prediction.reason


@pytest.mark.parametrize('closure,valid', [(1.499, False), (1.5, True)])
def test_only_prolonged_valid_closure_can_bypass_startup_readiness(closure, valid):
    prediction = RuleBasedClassifier(config(), mode='temporal').predict(sample(closure=closure))
    assert prediction.valid is valid
    assert prediction.class_id == (DriverState.DROWSY if valid else None)
    if not valid:
        assert 'history' in prediction.reason
        assert np.isnan(prediction.probabilities).all()


@pytest.mark.parametrize('perclos,closure', [(1., 10.), (None, 10.)])
def test_missing_or_no_face_never_drowsy_even_with_stale_high_values(perclos, closure):
    for mode in ('ear_only', 'temporal'):
        classifier = RuleBasedClassifier(config(), mode=mode)
        for row in (sample(perclos=perclos, closure=closure, valid=False, absolute_duration=10.),
                    sample(perclos=perclos, closure=closure, face=False, absolute_duration=10.)):
            prediction = classifier.predict(row)
            assert not prediction.valid and prediction.class_id is None
            assert np.isnan(prediction.probabilities).all()


@pytest.mark.parametrize('ear,duration,state', [(.2, 2., DriverState.ALERT), (.199, .999, DriverState.ALERT),
                                              (.199, 1., DriverState.DROWSY), (.199, 1.5, DriverState.DROWSY)])
def test_ear_only_absolute_gate_and_one_second_duration(ear, duration, state):
    prediction = RuleBasedClassifier(config(), mode='ear_only').predict(
        sample(absolute=ear, absolute_duration=duration, closure=0.))
    assert prediction.valid and prediction.class_id == state
    assert prediction.class_id != DriverState.LOW_VIGILANCE


def test_normalized_duration_cannot_substitute_for_absolute_ear_duration():
    prediction = RuleBasedClassifier(config(), mode='ear_only').predict(sample(closure=20., absolute=.19))
    assert prediction.class_id == DriverState.ALERT


def test_one_invalid_eye_abstains_without_poisoning_other_valid_eye():
    row = sample(perclos=.8, closure=2.)
    row.values[FEATURE_NAMES.index('right_eye_valid')] = 0.
    row.validity[FEATURE_NAMES.index('ear_right_norm')] = False
    for mode in ('ear_only', 'temporal'):
        assert not RuleBasedClassifier(config(), mode=mode).predict(row).valid


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match='mode'):
        RuleBasedClassifier(config(), mode='checkpoint')
