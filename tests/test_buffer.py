import numpy as np
import pytest

from src.contracts import TemporalSample
from src.realtime.buffer import PredictionBuffer

FEATURES = tuple(f"f{i}" for i in range(16))


def transformer(values, validity):
    result = values.astype(np.float32)
    result[~np.isfinite(result)] = 0.0
    return result


def buffer(**overrides):
    values = {
        "feature_names": FEATURES,
        "schema_version": "facial_features_v1",
        "sequence_steps": 100,
        "max_missing_ratio": 0.2,
        "max_gap_ms": 1000,
        "transformer": transformer,
    }
    values.update(overrides)
    return PredictionBuffer(**values)


def temporal(timestamp, *, valid=True, segment=0):
    values = np.arange(16, dtype=np.float32)
    validity = np.ones(4, dtype=np.bool_)
    if not valid:
        validity[0] = False
        values[0] = np.nan
    return TemporalSample(timestamp, values, validity, segment, {})


def test_one_hundred_steps_from_zero_to_9900_create_window():
    target = buffer()
    for timestamp in range(0, 10_000, 100):
        target.append(temporal(timestamp))
    window = target.get_window()
    assert window is not None
    assert window.x.shape == (100, 16)
    assert window.x.dtype == np.float32
    assert (window.start_timestamp_ms, window.end_timestamp_ms) == (0, 9900)
    assert window.quality["missing_ratio"] == 0


def test_more_than_twenty_percent_missing_rejects_window():
    target = buffer()
    for index, timestamp in enumerate(range(0, 10_000, 100)):
        target.append(temporal(timestamp, valid=index >= 21))
    assert target.get_window() is None


def test_exactly_twenty_percent_missing_is_accepted_when_current_is_valid():
    target = buffer()
    for index, timestamp in enumerate(range(0, 10_000, 100)):
        target.append(temporal(timestamp, valid=index >= 20))
    assert target.get_window() is not None


def test_current_required_channel_invalid_rejects_window():
    target = buffer()
    for timestamp in range(0, 9900, 100):
        target.append(temporal(timestamp))
    target.append(temporal(9900, valid=False))
    assert target.get_window() is None


def test_gap_or_segment_change_resets_history():
    target = buffer(sequence_steps=3)
    target.append(temporal(0))
    target.append(temporal(100))
    target.append(temporal(1100))
    assert len(target) == 3
    target.append(temporal(2101))
    assert len(target) == 1
    target.append(temporal(2200, segment=1))
    assert len(target) == 1


def test_transformer_must_return_finite_ordered_row():
    target = buffer(transformer=lambda values, validity: np.full(16, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        target.append(temporal(0))
