import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.datasets.manifest import MANIFEST_COLUMNS
from src.preprocessing import snapshot

_program_hashes = snapshot.extraction_program_hashes


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    root = tmp_path / 'raw'
    (root / '04').mkdir(parents=True)
    rows = []
    for label, internal in ((0, 0), (5, 1), (10, 2)):
        source = root / '04' / f'{label}.avi'
        source.write_bytes(f'unchanged source {label}'.encode())
        row = dict.fromkeys(MANIFEST_COLUMNS)
        row.update(dataset_name='uta_rldd', subject_id='04', video_id=f'04_{label}',
                   relative_path=f'04/{label}.avi', source_label=label, label_id=internal,
                   label_source='video_weak', fold_id=1, fold_source='official_archive_index',
                   fps_reported=15., duration_s=1., width=64, height=48, frame_count=15,
                   size_bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   image_publishable=False, sampled_decode_ok=True, sample_timestamps_ms=[0],
                   status='ok', error=None)
        rows.append(row)
    manifest = tmp_path / 'manifest.parquet'
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    config = tmp_path / 'preprocessing.yaml'
    config.write_text(yaml.safe_dump(dict(dataset_root=str(root), output_dir=str(tmp_path / 'output'),
                                         manifest_path=str(manifest))), encoding='utf-8')
    monkeypatch.setattr(snapshot.FeatureDatasetBuilder, '_signature', lambda self: {'extraction_fingerprint': 'frozen'})
    acquisition = tmp_path / 'acquisition.json'
    acquisition.write_text(json.dumps({'video_count': 45, 'records': [{'video_id': row['video_id']} for row in rows] + [{'video_id': '51_0'}]}))
    monkeypatch.setattr(snapshot, 'ACQUISITION_PLAN_PATH', acquisition)
    program = tmp_path / 'builder.py'
    program.write_bytes(b'unchanged extraction program')
    monkeypatch.setattr(snapshot, 'extraction_program_hashes', lambda: {'builder.py': hashlib.sha256(program.read_bytes()).hexdigest()})
    return config, manifest, tmp_path / 'run', tmp_path / 'output', root, acquisition, program


def test_derived_module_addition_keeps_raw_snapshot_current(frozen_inputs, monkeypatch, tmp_path):
    config, manifest, run, output, *_ = frozen_inputs
    producer_paths = _program_hashes().keys()
    project = tmp_path / 'project'
    for name in producer_paths:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'raw producer {name}'.encode())
    monkeypatch.setattr(snapshot, 'PROJECT_ROOT', project)
    monkeypatch.setattr(snapshot, 'extraction_program_hashes', _program_hashes)
    before = snapshot.freeze_snapshot(config, manifest, run, output)
    (project / 'src/features/temporal.py').write_bytes(b'derived-only temporal engine')
    after = snapshot.freeze_snapshot(config, manifest, run, output)
    assert after['snapshot_sha256'] == before['snapshot_sha256']
    (project / 'src/features/eye.py').write_bytes(b'changed raw eye producer')
    with pytest.raises(ValueError, match='[Dd]rift|[Mm]ismatch'):
        snapshot.freeze_snapshot(config, manifest, run, output)


@pytest.mark.parametrize('change', ['config', 'manifest', 'program', 'signature', 'frozen_copy'])
def test_existing_freeze_refuses_drift_without_overwriting(frozen_inputs, monkeypatch, change):
    config, manifest, run, output, root, acquisition, program = frozen_inputs
    snapshot.freeze_snapshot(config, manifest, run, output)
    frozen_path = run / 'snapshot' / 'snapshot.json'
    before = frozen_path.read_bytes()
    if change == 'config':
        config.write_bytes(config.read_bytes() + b'\n# drift\n')
    elif change == 'manifest':
        frame = pd.read_parquet(manifest)
        frame['image_publishable'] = True
        frame.to_parquet(manifest, index=False)
    elif change == 'program':
        program.write_bytes(b'changed extraction program')
    elif change == 'signature':
        monkeypatch.setattr(snapshot.FeatureDatasetBuilder, '_signature', lambda self: {'extraction_fingerprint': 'changed'})
    else:
        (run / 'snapshot' / 'preprocessing.yaml').write_bytes(b'changed frozen copy')
    with pytest.raises(ValueError, match='[Dd]rift|[Mm]ismatch'):
        snapshot.freeze_snapshot(config, manifest, run, output)
    assert frozen_path.read_bytes() == before


def test_freeze_checks_actual_source_before_publishing(frozen_inputs):
    config, manifest, run, output, root, *_ = frozen_inputs
    (root / '04/5.avi').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='[Ss]ource'):
        snapshot.freeze_snapshot(config, manifest, run, output)
    assert not (run / 'snapshot/snapshot.json').exists()


def test_sequential_checkpoints_preserve_denominator_and_all_input_bytes(frozen_inputs, monkeypatch):
    config, manifest, run, output, root, acquisition, _ = frozen_inputs
    originals = {path: path.read_bytes() for path in [config, manifest, acquisition, *root.rglob('*.avi')]}
    checkpoints = []
    def build(self, full_manifest, output_dir, *, video_ids):
        checkpoint = json.loads((run / 'report.json').read_text())
        checkpoints.append(checkpoint)
        assert full_manifest['video_id'].tolist() == ['04_0', '04_5', '04_10']
        assert len(video_ids) == 1
        video_id = video_ids[0]
        item = dict(video_id=video_id, subject_id='04', status='failed' if video_id == '04_5' else 'cached',
                    output_current=video_id != '04_5', row_count=15, reader_stats={'status': 'EOF'},
                    extraction_fingerprint='frozen')
        if video_id == '04_5':
            item['error'] = 'native decoder failure'
        return {'videos': [item]}
    monkeypatch.setattr(snapshot.FeatureDatasetBuilder, 'build', build)
    report = snapshot.run_snapshot(config, manifest, run, output)
    assert [item['status'] for item in checkpoints[0]['videos']] == ['pending'] * 3
    assert [item['status'] for item in checkpoints[1]['videos']] == ['cached', 'pending', 'pending']
    assert report['status'] == 'failed'
    assert report['totals'] == {'pending': 0, 'completed': 0, 'cached': 2, 'failed': 1}
    assert report['videos'][1]['error'] == 'native decoder failure'
    assert all(item['elapsed_seconds'] >= 0 for item in report['videos'])
    freeze = json.loads((run / 'snapshot/snapshot.json').read_text())
    assert freeze['missing_acquisition_video_ids'] == ['51_0']
    assert freeze['working_snapshot_count'] == 3
    assert (run / 'snapshot/manifest.parquet').read_bytes() == originals[manifest]
    assert (run / 'snapshot/preprocessing.yaml').read_bytes() == originals[config]
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert json.loads((run / 'report.json').read_text()) == report


def test_unexpected_builder_exception_is_failed_and_remaining_members_run(frozen_inputs, monkeypatch):
    config, manifest, run, output, *_ = frozen_inputs
    def build(self, full_manifest, output_dir, *, video_ids):
        if video_ids == ('04_0',):
            raise RuntimeError('builder boundary failure')
        return {'videos': [{'video_id': video_ids[0], 'status': 'completed', 'output_current': True,
                            'extraction_fingerprint': 'frozen'}]}
    monkeypatch.setattr(snapshot.FeatureDatasetBuilder, 'build', build)
    report = snapshot.run_snapshot(config, manifest, run, output)
    assert report['totals']['failed'] == 1
    assert report['totals']['completed'] == 2
    assert report['videos'][0]['error'] == 'builder boundary failure'


def test_snapshot_identity_tampering_is_refused(frozen_inputs):
    config, manifest, run, output, *_ = frozen_inputs
    snapshot.freeze_snapshot(config, manifest, run, output)
    marker = run / 'snapshot/snapshot.json'
    identity = json.loads(marker.read_text())
    identity['subject_count'] = 99
    marker.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='hash mismatch'):
        snapshot.freeze_snapshot(config, manifest, run, output)


@pytest.mark.parametrize('change', ['signature', 'config', 'program'])
def test_mid_run_drift_fails_remaining_members_without_building(frozen_inputs, monkeypatch, change):
    config, manifest, run, output, root, acquisition, program = frozen_inputs
    calls = []
    def build(self, full_manifest, output_dir, *, video_ids):
        calls.append(video_ids)
        if change == 'signature':
            monkeypatch.setattr(snapshot.FeatureDatasetBuilder, '_signature',
                                lambda self: {'extraction_fingerprint': 'drifted'})
        elif change == 'config':
            frozen_config = run / 'snapshot/preprocessing.yaml'
            frozen_config.write_bytes(frozen_config.read_bytes() + b'\n# changed\n')
        else:
            program.write_bytes(b'changed program')
        output_dir.mkdir()
        (output_dir / '04_0.parquet').write_bytes(b'pair data')
        (output_dir / '04_0.metadata.json').write_bytes(b'pair marker')
        (output_dir / 'unrelated.parquet').write_bytes(b'unrelated')
        return {'videos': [{'video_id': video_ids[0], 'status': 'completed', 'output_current': True,
                            'extraction_fingerprint': 'frozen'}]}
    monkeypatch.setattr(snapshot.FeatureDatasetBuilder, 'build', build)
    report = snapshot.run_snapshot(config, manifest, run, output)
    assert calls == [('04_0',)]
    assert report['totals'] == {'pending': 0, 'completed': 1, 'cached': 0, 'failed': 2}
    assert all('drift' in item['error'].lower() for item in report['videos'][1:])
    assert report['output_size_bytes'] == len(b'pair data') + len(b'pair marker')
    assert report['elapsed_seconds'] >= sum(item['elapsed_seconds'] for item in report['videos'])
    assert report['subject_count'] == 1 and report['selected_count'] == 3
    assert report['class_counts'] == {'0': 1, '1': 1, '2': 1}
    assert report['output_dir'] == str(output.resolve())


def test_manifest_replaced_during_read_refuses_inconsistent_freeze(frozen_inputs, monkeypatch):
    config, manifest, run, output, *_ = frozen_inputs
    original_read = pd.read_parquet
    def replacing_read(path, *args, **kwargs):
        parsed = original_read(path, *args, **kwargs)
        replacement = parsed.copy()
        replacement['image_publishable'] = True
        replacement.to_parquet(path, index=False)
        return parsed
    monkeypatch.setattr(snapshot.pd, 'read_parquet', replacing_read)
    with pytest.raises(ValueError, match='changed.*parsing'):
        snapshot.freeze_snapshot(config, manifest, run, output)
    assert not (run / 'snapshot/snapshot.json').exists()


@pytest.mark.parametrize('status', ['completed', 'cached'])
@pytest.mark.parametrize('change', ['fingerprint', 'output_current'])
def test_successful_builder_result_must_match_frozen_provenance(frozen_inputs, monkeypatch, status, change):
    config, manifest, run, output, *_ = frozen_inputs
    def build(self, full_manifest, output_dir, *, video_ids):
        return {'videos': [dict(video_id=video_ids[0], status=status,
                                output_current=change != 'output_current',
                                extraction_fingerprint='changed' if change == 'fingerprint' else 'frozen',
                                row_count=17, reader_stats={'status': 'EOF'})]}
    monkeypatch.setattr(snapshot.FeatureDatasetBuilder, 'build', build)
    report = snapshot.run_snapshot(config, manifest, run, output)
    assert report['totals']['failed'] == 3
    for item in report['videos']:
        assert item['status'] == 'failed' and item['output_current'] is False
        assert 'frozen provenance' in item['error']
        assert item['row_count'] == 17 and item['reader_stats']['status'] == 'EOF'
