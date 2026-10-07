"""Training consumers reject drift and never fit on validation/test observations."""
import numpy as np
import pandas as pd
import pytest

from src.datasets.normalization import FEATURE_NAMES
from src.datasets.splits import hash_payload
from src.training.experiment import (BlockedExperiment, check_eligibility,
    fit_train_scaler, summary_features, validate_manifest_identity)


def receipt():
    identity = dict(mode='P0', split_hash='a'*64, snapshot_sha256='b'*64,
                    feature_names=list(FEATURE_NAMES), temporal_config={'tick_ms': 100})
    manifest = dict(status='complete', schema_version='facial_features_v1', mode='P0',
        outer_index=0, feature_names=list(FEATURE_NAMES), identity=identity,
        derived_hash=hash_payload(identity), split_hash='a'*64, snapshot_sha256='b'*64,
        temporal_hash=hash_payload(identity['temporal_config']))
    manifest['dataset_manifest_hash'] = hash_payload(manifest)
    return manifest


def test_manifest_hash_rejects_changed_receipt_before_training():
    manifest = receipt()
    validate_manifest_identity(manifest, mode='P0', outer_index=0, config={'feature_names': list(FEATURE_NAMES)})
    manifest['outer_index'] = 1
    with pytest.raises(ValueError):
        validate_manifest_identity(manifest, mode='P0', outer_index=0, config={})


@pytest.mark.parametrize('field,value', [
    ('mode', 'P1'), ('outer_index', 4),
    ('feature_names', list(reversed(FEATURE_NAMES))),
    ('derived_hash', 'c'*64),
])
def test_rehashed_receipt_cannot_bypass_consumer_identity(field, value):
    manifest = receipt()
    manifest[field] = value
    manifest['dataset_manifest_hash'] = hash_payload({k:v for k,v in manifest.items() if k != 'dataset_manifest_hash'})
    with pytest.raises(ValueError):
        validate_manifest_identity(manifest, mode='P0', outer_index=0, config={})


def test_missing_validation_class_blocks_with_full_coverage():
    manifest = {'eligibility': {'status': 'eligible', 'reasons': []},
        'coverage': {'scheduled': 12, 'accepted': 9, 'rejected': 3}}
    index = pd.DataFrame({'split': ['train']*3+['validation']*2+['test'],
                          'label_id': [0, 1, 2, 0, 1, 2]})
    with pytest.raises(BlockedExperiment) as caught:
        check_eligibility(manifest, index)
    assert 'validation_missing_class_2' in caught.value.reasons
    assert caught.value.coverage == manifest['coverage']


def test_partial_test_class_support_does_not_block():
    manifest = {'eligibility': {'status': 'eligible', 'reasons': []}, 'coverage': {}}
    index = pd.DataFrame({'split': ['train']*3+['validation']*3+['test'],
                          'label_id': [0, 1, 2, 0, 1, 2, 2]})
    check_eligibility(manifest, index)


def test_summary_identifiers_are_not_classifier_inputs():
    table = pd.DataFrame({'video_id': ['v'], 'subject_id': ['s'], 'label_id': [2],
        'split': ['train'], 'start_timestamp_ms': [0], 'end_timestamp_ms': [9900],
        'ear_left_norm_mean': [0.8], 'blink_count': [1.]})
    X = summary_features(table, ('ear_left_norm_mean', 'blink_count'))
    assert X.columns.tolist() == ['ear_left_norm_mean', 'blink_count']
    with pytest.raises(ValueError):
        summary_features(table, ('label_id',))


def test_train_scaler_unions_overlap_and_ignores_validation_outlier(tmp_path):
    from src.datasets.sequence import build_window_index
    videos, indexes = [], []
    for role, value in [('train', 2.), ('validation', 2000.)]:
        n = 120
        values = np.ones((n, 16), dtype=np.float32)
        values[:, :10] = value
        frame = pd.DataFrame(dict(video_id=[role]*n, subject_id=[role]*n,
            label_id=[0]*n, split=[role]*n, timestamp_ms=np.arange(n)*100,
            segment_id=[0]*n, values=list(values), validity=list(np.ones((n, 16), bool)),
            event_summaries=['{}']*n, profile_hash=['profile']*n))
        path = tmp_path/(role+'.parquet')
        frame.to_parquet(path, index=False)
        videos.append(dict(video_id=role, subject_id=role, label_id=0, split=role,
                           profile_hash='profile', path=str(path), rows=n))
        indexes.append(build_window_index(frame, split={}, config={}))
    manifest = receipt()
    manifest['videos'] = videos
    scaler = fit_train_scaler(pd.concat(indexes, ignore_index=True), manifest)
    assert scaler['means'] == [2.]*10
    assert scaler['counts'] == [120]*10
    assert scaler['mode'] == 'P0' and scaler['derived_hash'] == manifest['derived_hash']


def artifact(tmp_path):
    """Write real source/raw/derived/index files with consistent provenance."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.config import PROJECT_ROOT
    from src.datasets.acquisition import digest_file
    from src.datasets.derived import DERIVED_SCHEMA, atomic_json
    from src.datasets.sequence import build_window_index
    from src.preprocessing.builder import STORAGE_SCHEMA, PRODUCER_VERSION
    from src.preprocessing.snapshot import extraction_program_hashes

    source_path = tmp_path/'source.bin'
    source_path.write_bytes(b'recorded-source')
    sha, _crc = digest_file(source_path)
    size = source_path.stat().st_size
    source = dict(video_id='v', subject_id='s', label_id=0,
                  relative_path=source_path.name, sha256=sha, size_bytes=size)
    signature = dict(artifact_hashes={}, dependency_versions={}, effective_config={},
                     extraction_fingerprint='e'*64)
    raw_dir = tmp_path/'raw'
    raw_dir.mkdir()
    raw_path = raw_dir/'v.parquet'
    pq.write_table(pa.Table.from_pylist([{}], schema=STORAGE_SCHEMA), raw_path)
    raw_sha = digest_file(raw_path)[0]
    metadata_path = raw_dir/'v.metadata.json'
    atomic_json(metadata_path, dict(schema_version='facial_features_v1',
        producer_version=PRODUCER_VERSION, complete_source_validation=True,
        **signature, manifest_row=source, source_sha256=sha, source_size_bytes=size,
        manifest_row_sha256=hash_payload(source), parquet_sha256=raw_sha,
        row_count=1, first_timestamp_ms=0, last_timestamp_ms=0,
        reader_stats=dict(status='EOF', capture_released=True, emitted_frames=1),
        pipeline_stats={}, validity_counts={}, quality_reason_counts={}, camera_metadata={}))
    snapshot = dict(sources=[source], source_root=str(tmp_path), output_dir=str(raw_dir),
                    signature=signature, extraction_program_sha256=extraction_program_hashes())
    snapshot['snapshot_sha256'] = hash_payload(snapshot)
    snapshot_path = tmp_path/'snapshot.json'
    atomic_json(snapshot_path, snapshot)
    policy = dict(train_subject_ids=['s'])
    policy['policy_hash'] = hash_payload(policy)
    profiles = {'s':dict(valid=True, quality_stats={'reasons': []})}
    split = dict(outer_index=0, snapshot_sha256=snapshot['snapshot_sha256'], sources=[source],
        qc_policy=policy, profiles={'P0': profiles}, reserved_ranges={'P0': {}},
        roles={'train':dict(subject_ids=['s'], video_ids=['v']),
               'validation':dict(subject_ids=['val'], video_ids=[]),
               'test':dict(subject_ids=['test'], video_ids=[])})
    split['split_hash'] = hash_payload(split)
    split_path = tmp_path/'split.json'
    atomic_json(split_path, split)
    profile_hash = hash_payload(profiles['s'])
    records = [dict(video_id='v', subject_id='s', label_id=0, split='train',
        timestamp_ms=i*100, segment_id=0, values=[2.]*10+[1.]*6,
        validity=[True]*16, event_summaries='{}', profile_hash=profile_hash) for i in range(120)]
    derived_path = tmp_path/'derived.parquet'
    pq.write_table(pa.Table.from_pylist(records, schema=DERIVED_SCHEMA), derived_path)
    index = build_window_index(pd.DataFrame(records), split=split, config={'calibration_mode': 'P0'})
    index_path = tmp_path/'index.parquet'
    index.to_parquet(index_path, index=False)
    code = {name:digest_file(PROJECT_ROOT/name)[0] for name in (
        'src/calibration/profile.py', 'src/datasets/splits.py', 'src/features/temporal.py',
        'src/datasets/derived.py', 'src/datasets/sequence.py', 'src/datasets/summaries.py')}
    manifest = receipt()
    manifest.update(raw_dir=str(raw_dir), source_root=str(tmp_path), snapshot_path=str(snapshot_path),
        split_path=str(split_path), split_hash=split['split_hash'],
        snapshot_sha256=snapshot['snapshot_sha256'], profile_policy_hash=policy['policy_hash'],
        index_path=str(index_path), index_sha256=digest_file(index_path)[0],
        coverage=index.attrs['coverage'],
        class_support={'train':{'0':3,'1':0,'2':0},
                       'validation':{'0':0,'1':0,'2':0}, 'test':{'0':0,'1':0,'2':0}},
        videos=[dict(video_id='v', subject_id='s', label_id=0, split='train', rows=120,
            profile_hash=profile_hash, profile_valid=True, profile_reasons=[],
            path=str(derived_path), parquet_sha256=digest_file(derived_path)[0])])
    manifest['identity'].update(split_hash=split['split_hash'],
        snapshot_sha256=snapshot['snapshot_sha256'], code_sha256=code,
        profiles_hash=hash_payload(profiles), raw_commits=[dict(video_id='v',
        parquet_sha256=raw_sha, metadata_sha256=digest_file(metadata_path)[0])])
    manifest['derived_hash'] = hash_payload(manifest['identity'])
    manifest['dataset_manifest_hash'] = hash_payload({k:v for k,v in manifest.items() if k != 'dataset_manifest_hash'})
    manifest_path = tmp_path/'dataset_manifest.json'
    atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def test_verified_consumer_reopens_real_artifacts(tmp_path):
    from src.training.experiment import validate_dataset
    path, _ = artifact(tmp_path)
    manifest, index, split = validate_dataset(path, mode='P0', outer_index=0, config={'calibration_mode':'P0'})
    assert index.start_row.tolist() == [0, 10, 20]
    assert manifest['coverage']['accepted'] == 3
    assert split['roles']['train']['video_ids'] == ['v']


@pytest.mark.parametrize('file', [
    'source.bin', 'raw/v.parquet', 'raw/v.metadata.json',
    'derived.parquet', 'index.parquet'])
def test_changed_artifact_bytes_are_rejected_before_fit(tmp_path, file):
    from src.training.experiment import validate_dataset
    path, _ = artifact(tmp_path)
    changed = tmp_path/file
    changed.write_bytes(changed.read_bytes()+b'drift')
    with pytest.raises(ValueError):
        validate_dataset(path, mode='P0', outer_index=0, config={'calibration_mode':'P0'})


def test_rehashed_wrong_index_cannot_change_accepted_fit_support(tmp_path):
    from src.datasets.acquisition import digest_file
    from src.datasets.derived import atomic_json
    from src.training.experiment import validate_dataset
    path, manifest = artifact(tmp_path)
    index_path = tmp_path/'index.parquet'
    index = pd.read_parquet(index_path)
    index.loc[0, 'end_row'] = 101
    index.to_parquet(index_path, index=False)
    manifest['index_sha256'] = digest_file(index_path)[0]
    manifest['dataset_manifest_hash'] = hash_payload({k:v for k,v in manifest.items() if k != 'dataset_manifest_hash'})
    atomic_json(path, manifest)
    with pytest.raises(ValueError):
        validate_dataset(path, mode='P0', outer_index=0, config={'calibration_mode':'P0'})


def test_saved_partial_test_metrics_remain_undefined_with_scheduled_coverage(tmp_path):
    import json
    from src.training.experiment import save_predictions
    path, manifest = artifact(tmp_path)
    rows = pd.read_parquet(manifest['index_path'])
    rows['split'] = 'test'
    manifest['coverage']['by_role']['test'] = dict(scheduled=5, accepted=3, rejected=2, reasons={'gap':2})
    probabilities = np.asarray([[1.,0.,0.]]*3)
    metrics, predictions = save_predictions(tmp_path, 'test', rows, probabilities, manifest, {})
    assert metrics['macro_f1'] is None
    assert metrics['coverage']['scheduled'] == 5 and metrics['coverage']['rejected'] == 2
    assert json.loads((tmp_path/'test_metrics.json').read_text())['macro_f1'] is None
    saved = pd.read_parquet(predictions)
    assert saved.label_id.tolist() == [0,0,0]
    assert saved.subject_id.tolist() == ['s']*3
    assert saved[['probability_0','probability_1','probability_2']].to_numpy().tolist() == [[1.,0.,0.]]*3
