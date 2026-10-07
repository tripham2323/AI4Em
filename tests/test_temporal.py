"""Causal native events and time-weighted receipts, independent of grid duplication."""
from dataclasses import replace
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.contracts import CalibrationProfile, FeatureSample
from src.features.temporal import FEATURE_NAMES, TemporalFeatureExtractor


def profile():
    return CalibrationProfile('P1', True, 0.25, 0.30, 0.10, 0., 0., 0.,
                              'facial_features_v1', 'a' * 64, (640, 480))


def config():
    return yaml.safe_load((Path(__file__).parents[1] / 'configs/temporal.yaml').read_text())


def raw(t, ratio=1., *, mouth=0., pitch=0., **changes):
    value = FeatureSample(t, t, 'fixture', ratio * .25, ratio * .30, ratio * .275,
                          .10 + mouth, pitch, 0., 0., True, True, True, True, True, 0.)
    return replace(value, **changes)


def value(sample, name):
    return float(sample.values[FEATURE_NAMES.index(name)])


def feed(engine, begin, end, *, step=50, **kwargs):
    rows = []
    for t in range(begin, end + 1, step):
        rows.extend(engine.update(raw(t, **kwargs)))
    return rows


def test_order_grid_native_causality_and_immutable_receipts():
    assert FEATURE_NAMES == tuple(yaml.safe_load((Path(__file__).parents[1] / 'configs/training.yaml').read_text())['feature_names'])
    engine = TemporalFeatureExtractor(profile(), config())
    first = engine.update(raw(25))[0]
    original = copy.deepcopy(first)
    engine.update(raw(75, ratio=.1))
    rows = engine.update(raw(225, ratio=1.))
    assert [r.timestamp_ms for r in rows] == [125, 225]
    assert value(rows[0], 'ear_left_norm') == pytest.approx(.1)
    assert value(rows[1], 'ear_left_norm') == pytest.approx(1.)
    np.testing.assert_array_equal(first.values, original.values)
    assert first.event_summaries == original.event_summaries
    assert not any(e['kind'] == 'closure' and e['candidate'] for r in rows for e in r.event_summaries['events'])
    assert any(e['censored'] for r in rows for e in r.event_summaries['events'])


@pytest.mark.parametrize('duration,candidate', [(50, None), (100, 'blink'), (800, 'blink'), (850, 'prolonged')])
def test_native_completed_closure_duration_boundaries(duration, candidate):
    engine = TemporalFeatureExtractor(profile(), config())
    rows = feed(engine, 0, duration - 50, ratio=.5)
    rows += engine.update(raw(duration))
    rows += engine.update(raw(duration + 100))
    events = [e for r in rows for e in r.event_summaries['events'] if e['kind'] == 'closure']
    assert len(events) == 1
    assert events[0]['duration_s'] == pytest.approx(duration / 1000)
    assert events[0]['classification'] == candidate and not events[0]['censored']
    assert events[0]['candidate'] is (candidate == 'blink')
    assert rows[-1].event_summaries['blink_count'] == int(candidate == 'blink')


def test_hysteresis_threshold_equality_does_not_trigger_transition():
    engine = TemporalFeatureExtractor(profile(), config())
    assert value(engine.update(raw(0, ratio=.65))[0], 'closure_elapsed_s') == 0.
    engine.update(raw(50, ratio=.6))
    row = engine.update(raw(100, ratio=.75))[0]
    assert value(row, 'closure_elapsed_s') == pytest.approx(.05)
    engine.update(raw(150, ratio=.8))
    row = engine.update(raw(200))[0]
    assert row.event_summaries['blink_count'] == 1


def test_one_invalid_eye_censors_native_closure_and_excludes_perclos_denominator():
    engine = TemporalFeatureExtractor(profile(), config())
    engine.update(raw(0, ratio=.1))
    engine.update(raw(50, ratio=.1))
    row = engine.update(raw(100, ratio=.1, right_eye_valid=False))[0]
    assert value(row, 'left_eye_valid') == 1 and value(row, 'right_eye_valid') == 0
    assert np.isnan(value(row, 'closure_elapsed_s'))
    event = [e for e in row.event_summaries['events'] if e['kind'] == 'closure'][0]
    assert event['censored'] and not event['candidate']
    row = engine.update(raw(200, ratio=.1, right_eye_valid=False))[0]
    assert row.event_summaries['eye_valid_duration_s'] == pytest.approx(.1)
    assert row.event_summaries['perclos_raw'] == 1.


@pytest.mark.parametrize('step', [25, 50, 100])
def test_native_frame_cadence_does_not_change_duration_or_duplicate_event_counts(step):
    engine = TemporalFeatureExtractor(profile(), config())
    rows = feed(engine, 0, 500 - step, step=step, ratio=.5)
    rows += feed(engine, 500, 1000, step=step)
    events = [e for r in rows for e in r.event_summaries['events'] if e['kind'] == 'closure' and e['candidate']]
    assert len(events) == 1 and events[0]['duration_s'] == pytest.approx(.5)
    assert rows[-1].event_summaries['blink_count'] == 1


def test_hand_computed_perclos_and_readiness_at_exact_30_seconds():
    engine = TemporalFeatureExtractor(profile(), config())
    feed(engine, 0, 1900, step=100, ratio=.2)
    row = feed(engine, 2000, 10000, step=100)[-1]
    assert row.event_summaries['perclos_raw'] == pytest.approx(.2)
    assert row.event_summaries['eye_valid_duration_s'] == 10.
    assert np.isnan(value(row, 'perclos_60')) and value(row, 'perclos_ready') == 0
    row = feed(engine, 10100, 29900, step=100)[-1]
    assert value(row, 'perclos_ready') == 0
    row = engine.update(raw(30000))[0]
    assert value(row, 'perclos_ready') == 1
    assert value(row, 'perclos_60') == pytest.approx(2 / 30)


@pytest.mark.parametrize('valid_end,ready', [(24000, True), (23900, False)])
def test_ready_coverage_inclusive_eighty_percent(valid_end, ready):
    engine = TemporalFeatureExtractor(profile(), config())
    feed(engine, 0, valid_end - 100, step=100)
    row = feed(engine, valid_end, 30000, step=100, left_eye_valid=False)[-1]
    assert row.event_summaries['coverage'] == pytest.approx(valid_end / 30000)
    assert bool(value(row, 'perclos_ready')) is ready


def test_perclos_window_clips_old_interval_not_sample_count():
    engine = TemporalFeatureExtractor(profile(), config())
    feed(engine, 0, 4900, step=100, ratio=.1)
    row = feed(engine, 5000, 62500, step=100)[-1]
    assert row.event_summaries['history_s'] == 60.
    assert row.event_summaries['closed_proxy_duration_s'] == pytest.approx(2.5)
    assert value(row, 'perclos_60') == pytest.approx(2.5 / 60)


def test_sparse_samples_hold_only_100ms_and_unknown_intervals_not_open():
    engine = TemporalFeatureExtractor(profile(), config())
    engine.update(raw(0, ratio=.1))
    rows = engine.update(raw(500))
    assert value(rows[0], 'left_eye_valid') == 1  # equality at age cap
    assert value(rows[1], 'left_eye_valid') == 0
    assert rows[-1].event_summaries['eye_valid_duration_s'] == pytest.approx(.1)
    assert rows[-1].event_summaries['perclos_raw'] == 1.
    intervals = [i for r in rows for i in r.event_summaries['held_intervals']]
    assert sum(i['end_ms'] - i['start_ms'] for i in intervals if i['eye_valid']) == 100


@pytest.mark.parametrize('duration,candidate', [(1950, None), (2000, 'yawn')])
def test_mouth_candidate_minimum_duration_and_missing_censor(duration, candidate):
    engine = TemporalFeatureExtractor(profile(), config())
    rows = feed(engine, 0, duration - 50, mouth=.4)
    rows += engine.update(raw(duration, mouth=.2))
    rows += engine.update(raw(duration + 100))
    mouth = [e for r in rows for e in r.event_summaries['events'] if e['kind'] == 'yawn']
    assert len(mouth) == 1 and mouth[0]['classification'] == candidate
    assert mouth[0]['candidate'] is (candidate == 'yawn')
    reopened_at = ((duration + 100) // 100 + 1) * 100
    engine.update(raw(reopened_at, mouth=.4))
    row = engine.update(raw(reopened_at + 100, mouth=.4, mouth_valid=False))[0]
    assert any(e['kind'] == 'yawn' and e['censored'] for e in row.event_summaries['events'])
    assert np.isnan(value(row, 'yawn_elapsed_s'))


def test_pose_velocity_wrap_first_invalid_gap_and_recovery():
    engine = TemporalFeatureExtractor(profile(), config())
    # Keep angles inside profile gates while crossing the wrapped equivalent.
    first = engine.update(raw(0, pitch=1.))[0]
    assert value(first, 'pose_valid') == 0 and np.isnan(value(first, 'pitch_delta'))
    row = engine.update(raw(100, pitch=359.))[0]
    assert value(row, 'pitch_velocity_dps') == pytest.approx(-20.)
    assert value(row, 'pose_valid') == 1
    rows = engine.update(raw(300, pitch=2.))
    assert value(rows[-1], 'pose_valid') == 0
    row = engine.update(raw(400, pitch=3.))[0]
    assert value(row, 'pitch_velocity_dps') == pytest.approx(10.)


def test_source_and_large_gap_reset_history_and_segment_not_one_second_boundary():
    engine = TemporalFeatureExtractor(profile(), config())
    first = engine.update(raw(0))[0]
    assert engine.update(raw(1000))[-1].segment_id == first.segment_id
    reset = engine.update(raw(2001))[0]
    assert reset.segment_id == first.segment_id + 1 and reset.event_summaries['history_s'] == 0
    switched = engine.update(raw(0, source_id='other'))[0]
    assert switched.segment_id == reset.segment_id + 1
    engine.reset()
    assert engine.update(raw(0))[0].event_summaries['history_s'] == 0


def test_duplicate_or_backward_native_timestamp_rejected_without_mutating_history():
    engine = TemporalFeatureExtractor(profile(), config())
    engine.update(raw(0))
    for timestamp in (0, -1):
        with pytest.raises(ValueError, match='timestamp'):
            engine.update(raw(timestamp))
    assert engine.update(raw(100))[0].event_summaries['eye_valid_duration_s'] == pytest.approx(.1)


def test_absolute_ear_duration_is_distinct_from_normalized_closure():
    engine = TemporalFeatureExtractor(replace(profile(), ear_left_baseline=.15, ear_right_baseline=.15), config())
    rows = []
    for t in range(0, 1600, 100):
        rows += engine.update(raw(t, ear_left=.18, ear_right=.18, ear_mean=.18))
    assert rows[-1].event_summaries['absolute_closure_elapsed_s'] == 1.5
    assert value(rows[-1], 'closure_elapsed_s') == 0


def test_no_face_and_invalid_profile_never_create_valid_continuous_values():
    for p, sample in [(profile(), raw(0, face_detected=False)), (replace(profile(), valid=False), raw(0))]:
        row = TemporalFeatureExtractor(p, config()).update(sample)[0]
        assert np.isnan(row.values[:10]).all()
        assert not row.validity[:10].any()
        assert not row.values[10:14].any()
        assert np.isfinite(row.values[10:]).all()


def test_native_state_receipts_intersect_exactly_without_repeated_history():
    engine = TemporalFeatureExtractor(profile(), config())
    rows = engine.update(raw(0))
    rows += engine.update(raw(50, ratio=.5, mouth=.4))
    rows += engine.update(raw(100, ratio=.5, mouth=.4))
    rows += engine.update(raw(150))
    rows += engine.update(raw(200))
    intervals = [interval for row in rows for interval in row.event_summaries['held_intervals']]
    assert sum(i['end_ms'] - i['start_ms'] for i in intervals) == 200
    assert sum(i['end_ms'] - i['start_ms'] for i in intervals if i['eye_closed']) == 100
    assert sum(i['end_ms'] - i['start_ms'] for i in intervals if i['mouth_open']) == 100
    assert not any(i['closed_proxy'] for i in intervals)  # .5 is blink gate, not proxy.
    assert all(i['start_ms'] < i['end_ms'] for i in intervals)
    assert len([e for row in rows for e in row.event_summaries['events'] if e['kind'] == 'closure']) == 1


def test_mouth_hysteresis_equality_and_pose_missing_mask_contract():
    engine = TemporalFeatureExtractor(replace(profile(), mar_baseline=0.), config())
    first = engine.update(raw(0, mar=.35))[0]
    assert value(first, 'yawn_elapsed_s') == 0
    assert first.event_summaries['yawn_start_ms'] is None
    engine.update(raw(50, mar=.4))
    row = engine.update(raw(100, mar=.25, pose_valid=False))[0]
    assert value(row, 'yawn_elapsed_s') == pytest.approx(.05)
    assert value(row, 'pose_valid') == 0
    assert np.isnan(row.values[[3, 4, 5, 9]]).all()
    assert not row.validity[[3, 4, 5, 9]].any()
    assert row.validity[10:].all()
    engine.update(raw(150, mar=.2))
    row = engine.update(raw(200))[0]
    assert value(row, 'yawn_elapsed_s') == 0


def test_p0_profile_normalizes_without_claiming_personal_calibration():
    engine = TemporalFeatureExtractor(replace(profile(), mode='P0'), config())
    row = feed(engine, 0, 100)[-1]
    assert value(row, 'ear_left_norm') == 1.
    assert value(row, 'calibration_valid') == 0
    assert row.validity[10:].all()


def test_resolved_policy_hash_covers_defaults_and_rule_thresholds():
    from src.features.temporal import resolve_temporal_config, temporal_config_hash
    defaults = resolve_temporal_config({})
    assert temporal_config_hash({}) == temporal_config_hash(defaults)
    changed = {**defaults, 'rule_closure_s': 2.}
    assert temporal_config_hash(changed) != temporal_config_hash(defaults)
    assert TemporalFeatureExtractor(profile(), config()).config_sha256 == temporal_config_hash(config())


@pytest.mark.parametrize('setting', [
    {'sequence_fps': 20}, {'max_sample_age_ms': 200}, {'blink_enter': .8},
    {'blink_min_s': 1.}, {'perclos_min_history_s': 61.},
    {'perclos_min_coverage': 1.1}, {'max_gap_s': float('nan')},
    {'rule_drowsy_perclos': .1}, {'yawn_exit': .4},
])
def test_invalid_temporal_policy_fails_before_processing(setting):
    with pytest.raises(ValueError):
        TemporalFeatureExtractor(profile(), setting)


def test_variable_native_cadence_integrates_valid_held_time_not_frame_fraction():
    engine = TemporalFeatureExtractor(profile(), config())
    rows = []
    for sample in (raw(0, ratio=.1), raw(25, ratio=.1), raw(95),
                   raw(160, right_eye_valid=False), raw(240), raw(300)):
        rows.extend(engine.update(sample))
    summary = rows[-1].event_summaries
    assert summary['closed_proxy_duration_s'] == pytest.approx(.095)
    assert summary['eye_valid_duration_s'] == pytest.approx(.220)
    assert summary['perclos_raw'] == pytest.approx(95 / 220)
    assert summary['coverage'] == pytest.approx(220 / 300)
    assert summary['blink_count'] == 0  # 95 ms is below the completed-blink minimum.


def test_rolling_event_mean_max_and_counts_clip_old_edge_incrementally():
    engine = TemporalFeatureExtractor(profile(), config())
    feed(engine, 0, 900, step=100, ratio=.5)
    engine.update(raw(1000))
    feed(engine, 1100, 1500, step=100, ratio=.5)
    engine.update(raw(1600))
    row = feed(engine, 1700, 60700, step=100)[-1]
    assert row.event_summaries['closure_mean_duration_s'] == pytest.approx(.4)
    assert row.event_summaries['closure_max_duration_s'] == pytest.approx(.5)
    assert row.event_summaries['blink_count'] == 1
    row = feed(engine, 60800, 61000, step=100)[-1]
    assert row.event_summaries['closure_mean_duration_s'] == pytest.approx(.5)
    row = feed(engine, 61100, 61600, step=100)[-1]
    assert row.event_summaries['blink_count'] == 0
    assert np.isnan(row.event_summaries['closure_mean_duration_s'])
    assert row.event_summaries['closure_max_duration_s'] == 0.
