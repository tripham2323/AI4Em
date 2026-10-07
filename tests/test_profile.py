from dataclasses import replace
import math

import pytest

from src.contracts import FeatureSample
from src.calibration.profile import estimate_profile, fit_qc_policy, transform_sample


def samples(source='alert', left=.25, right=.30, seconds=60):
    return [FeatureSample(t, t//50, source, left, right, (left+right)/2, .1,
                          0., 0., 0., True, True, True, True, True, 0.)
            for t in range(0, seconds*1000, 50)]


def policy(rows):
    return fit_qc_policy(rows)


def profile(rows, qc=None):
    return estimate_profile(rows, 'P1', image_size=(640,480), asset_sha256='a'*64,
                            qc_policy=qc or policy(rows))


def test_unreserved_endpoint_cannot_supply_calibration_duration():
    rows = [replace(s, mouth_valid=False) if s.timestamp_ms < 10000 else s for s in samples()]
    result = profile(rows, policy(samples()))
    assert result.quality_stats['prefix_end_ms'] == 31000
    # Last selected observation is30950: no observation at31000 may contribute.
    assert result.quality_stats['valid_duration_ms'] == 20950


def test_each_eye_uses_its_own_baseline_and_raw_invalid_stays_invalid():
    rows = samples()
    p = profile(rows)
    result = transform_sample(rows[0], p)
    assert result['ear_left_norm'] == result['ear_right_norm'] == 1.
    assert result['mar_delta'] == 0.
    bad = transform_sample(replace(rows[0], left_eye_valid=False), p)
    assert not bad['left_eye_valid'] and math.isnan(bad['ear_left_norm'])


def test_prefix_reserves_invalid_time_and_extends_for_bounded_valid_duration():
    rows = samples()
    rows = [replace(s, mouth_valid=False) if s.timestamp_ms < 15000 else s for s in rows]
    p = profile(rows)
    assert p.valid
    assert 35000 <= p.quality_stats['prefix_end_ms'] <= 36000
    assert p.quality_stats['valid_duration_ms'] >= 20000


def test_missing_intervals_cannot_count_as_valid_duration():
    rows = samples()[::10]
    p = profile(rows)
    assert not p.valid
    assert 'insufficient_valid_duration' in p.quality_stats['reasons']
    assert p.quality_stats['prefix_end_ms'] == 60000


@pytest.mark.parametrize('left', [0., float('nan')])
def test_nonpositive_and_nonfinite_eye_baselines_fail(left):
    rows = samples(left=left)
    assert not profile(rows).valid


def test_unstable_open_eye_candidates_fail():
    rows = samples()
    rows = [replace(s, ear_left=(.2 if i%4<2 else .6 if i%4==2 else 1.))
            for i,s in enumerate(rows)]
    assert not profile(rows).valid


def test_pose_delta_wraps_and_invalid_profile_never_emits_valid_channels():
    rows = [replace(s, roll=179.) for s in samples()]
    p = profile(rows)
    assert transform_sample(replace(rows[0], roll=-179.),p)['roll_delta'] == 2.
    out = transform_sample(rows[0], replace(p, valid=False))
    assert not out['pose_valid'] and math.isnan(out['roll_delta'])


def test_quantile_bounds_are_linear_inclusive_and_top_half_per_eye():
    rows = samples(seconds=1)
    rows = [replace(s,ear_left=float(i+1)) for i,s in enumerate(rows)]
    qc = policy(rows)
    assert qc['quantile_method'] == 'linear'
    assert qc['ear_left_bounds'] == pytest.approx([11.09,19.91])


def test_persistent_mouth_opening_rejects_profile():
    rows = samples()
    rows = [replace(s,mar=.6) if 10000<=s.timestamp_ms<13000 else s for s in rows]
    assert 'persistent_mouth_opening' in profile(rows).quality_stats['reasons']


@pytest.mark.parametrize('mask', ['left_eye_valid','right_eye_valid','mouth_valid','pose_valid'])
def test_each_required_channel_is_needed_for_calibration_duration(mask):
    rows = [replace(s, **{mask:False}) for s in samples()]
    p = profile(rows)
    assert not p.valid
    assert p.quality_stats['valid_duration_ms'] == 0


def test_population_transform_is_not_personal_calibration():
    rows = samples()
    p = estimate_profile(rows,'P0',image_size=(640,480),asset_sha256='a'*64,
                         qc_policy=policy(rows))
    out = transform_sample(rows[0],p)
    assert out['left_eye_valid']
    assert not out['calibration_valid']


def test_qc_endpoint_baselines_remain_accepted():
    rows = samples()
    p = profile(rows)
    assert p.valid
    assert p.ear_left_baseline == policy(rows)['ear_left_bounds'][0]


def test_population_pool_uses_each_training_sources_open_candidates():
    rows = samples('first',left=.2) + samples('second',left=.4)
    p = estimate_profile(rows,'P0',image_size=(640,480),asset_sha256='a'*64,
                         qc_policy=policy(rows))
    assert p.ear_left_baseline == pytest.approx(.3)
