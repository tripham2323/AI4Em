"""Consumer boundaries: live collection never fits unobserved future seconds."""
from dataclasses import replace

import io
import json
from types import MappingProxyType
import pytest

from src.contracts import CalibrationProfile, FeatureSample
from scripts.webcam_demo import attempt_live_profile, bind_profile, replay_prefix_end


def sample(t):
    return FeatureSample(t, t // 50, 'camera:0', .3, .3, .3, .2, 0., 0., 0.,
                         True, True, True, True, True, .001)


def policy():
    return dict(valid=True, reasons=[], initial_prefix_ms=30000, maximum_prefix_ms=60000,
                min_valid_duration_ms=20000, max_sample_age_ms=100,
                ear_left_bounds=[.2, .4], ear_right_bounds=[.2, .4], max_mad_ratio=.15)


def test_native_immutable_quality_metrics_are_strict_json_receipts():
    from scripts.webcam_demo import append_json
    handle = io.StringIO()
    append_json(handle, {'quality_metrics': MappingProxyType({'brightness': 90.0, 'blur': float('nan')})})
    assert json.loads(handle.getvalue()) == {'quality_metrics': {'brightness': 90.0, 'blur': None}}


def test_live_fit_waits_for_observed_initial_prefix_and_uses_relative_clock():
    rows = [sample(t) for t in range(0, 30001, 50)]
    assert attempt_live_profile(rows, 29999, image_size=(640, 480),
                                asset_sha256='asset', qc_policy=policy()) is None
    fitted = attempt_live_profile(rows, 30000, image_size=(640, 480),
                                  asset_sha256='asset', qc_policy=policy())
    assert fitted.valid
    assert fitted.quality_stats['prefix_end_ms'] == 30000
    assert fitted.quality_stats['sample_count'] == 600


def test_failed_live_collection_does_not_claim_future_prefix():
    rows = [replace(sample(t), pose_valid=False) for t in range(0, 30001, 50)]
    fitted = attempt_live_profile(rows, 30000, image_size=(640, 480),
                                  asset_sha256='asset', qc_policy=policy())
    assert not fitted.valid
    assert fitted.quality_stats['prefix_end_ms'] == 30000
    assert 'insufficient_valid_duration' in fitted.quality_stats['reasons']


def test_population_baselines_bind_resolution_without_becoming_personal():
    profile = CalibrationProfile('P0', True, .3, .3, .2, 0., 0., 0.,
                                 'facial_features_v1', 'asset', (848, 480), {})
    provenance = dict(schema_version='facial_features_v1', image_size=[640, 480],
                      artifact_hashes={'asset_sha256': 'asset'}, camera_metadata={'mode': 'approximate'})
    bound = bind_profile(profile, provenance)
    assert bound.image_size == (640, 480)
    assert bound.mode == 'P0'
    assert bound.ear_left_baseline == .3
    assert bound.quality_stats['population_numeric_only'] is True


def test_personal_profile_rejects_changed_camera_not_silently_rebinding():
    provenance = dict(schema_version='facial_features_v1', image_size=[640, 480],
                      artifact_hashes={'asset_sha256': 'asset'}, camera_metadata={'matrix': [[1.]]})
    profile = CalibrationProfile('P1', True, .3, .3, .2, 0., 0., 0.,
                                 'facial_features_v1', 'asset', (640, 480), {'provenance': provenance})
    with pytest.raises(ValueError, match='provenance'):
        bind_profile(profile, {**provenance, 'camera_metadata': {'matrix': [[2.]]}})


def test_used_alert_prefix_is_excluded_but_other_subject_video_is_not():
    profile = CalibrationProfile('P1', True, .3, .3, .2, 0., 0., 0.,
                                 'facial_features_v1', 'asset', (640, 480),
                                 {'alert_video_id': '04_0', 'prefix_end_ms': 42000})
    assert replay_prefix_end(profile, '04_0') == 42000
    assert replay_prefix_end(profile, '04_5') == 0


def test_overlay_abstains_on_current_profile_pose_failure_between_grid_ticks(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import scripts.webcam_demo as demo
    from src.features.temporal import TemporalFeatureExtractor
    from src.models.rules import RuleBasedClassifier

    profile = CalibrationProfile('P0', True, .3, .3, .2, -6.4, 0., 0.,
                                 'facial_features_v1', 'asset', (640, 480), {})
    engine = TemporalFeatureExtractor(profile, {})
    engine.update(sample(0))
    latest = engine.update(sample(100))[-1]
    pred = RuleBasedClassifier({}, mode='ear_only').predict(latest)
    assert pred.valid
    current = replace(sample(150), pitch=24.)
    assert current.pose_valid and engine.update(current) == []
    texts = []
    monkeypatch.setattr(demo, 'draw_overlay', lambda image, *_a, **_k: image.copy())
    monkeypatch.setattr(demo.cv2, 'putText', lambda _image, text, *_a, **_k: texts.append(text))
    packet = SimpleNamespace(image_bgr=np.zeros((480, 640, 3), dtype=np.uint8))
    pipeline = SimpleNamespace(last_quality=SimpleNamespace(reasons=[]),
        last_landmarks=SimpleNamespace(image_size=(640, 480)),
        pose_estimator=SimpleNamespace(camera_metadata=lambda _: {'approximate': True}))
    demo.render_overlay(packet, current, pipeline, 'P0_POPULATION_NOT_PERSONAL',
                        latest, {'ear_only': pred}, profile)
    assert any(text.startswith('ear_only: UNRELIABLE/WARMUP') for text in texts)
