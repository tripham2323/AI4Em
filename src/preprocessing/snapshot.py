"""Freeze extraction inputs and checkpoint sequential full-manifest selections."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.datasets.acquisition import digest_file
from src.datasets.manifest import validate_working_manifest
from src.preprocessing.builder import FeatureDatasetBuilder, _hash_json, _json_value, _read_json

ACQUISITION_PLAN_PATH = PROJECT_ROOT / 'data/acquisition/uta_subset_plan.json'


def extraction_program_hashes() -> dict:
    paths = [PROJECT_ROOT / name for name in (
        'src/features/__init__.py', 'src/features/eye.py', 'src/features/head_pose.py',
        'src/features/landmarks.py', 'src/features/mouth.py', 'src/features/pipeline.py',
        'src/features/quality.py', 'src/preprocessing/builder.py',
        'src/preprocessing/video_reader.py', 'src/config.py', 'src/contracts.py',
        'src/datasets/manifest.py', 'src/datasets/acquisition.py', 'scripts/preprocess.py')]
    return {path.relative_to(PROJECT_ROOT).as_posix(): digest_file(path)[0]
            for path in sorted(paths)}


def _write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def freeze_snapshot(config_path: Path, manifest_path: Path, run_dir: Path,
                    output_dir: Path) -> dict:
    """Create a byte-copy freeze or refuse any mismatch with its existing identity.

    Source bytes are checked before a new freeze is published. On resume the
    unchanged builder remains responsible for actual per-video source SHA checks.
    """
    config_path, manifest_path = Path(config_path).resolve(), Path(manifest_path).resolve()
    directory = Path(run_dir).resolve() / 'snapshot'
    marker = directory / 'snapshot.json'
    config_sha = digest_file(config_path)[0]
    manifest_sha = digest_file(manifest_path)[0]
    config = load_config(config_path)
    manifest = pd.read_parquet(manifest_path)
    if (digest_file(config_path)[0] != config_sha
            or digest_file(manifest_path)[0] != manifest_sha):
        raise ValueError('Input changed while parsing snapshot inputs')
    validate_working_manifest(manifest)
    rows = _json_value(manifest.to_dict('records'))
    acquisition = _read_json(ACQUISITION_PLAN_PATH)
    planned_ids = {row['video_id'] for row in acquisition['records']}
    working_ids = {row['video_id'] for row in rows}
    if not working_ids <= planned_ids:
        raise ValueError('Working snapshot contains IDs outside acquisition plan')
    expected = dict(schema_version='working_snapshot_v1',
        original_manifest_path=str(manifest_path),
        manifest_file_sha256=manifest_sha,
        working_manifest_sha256=_hash_json(sorted(rows, key=lambda row: row['video_id'])),
        config_file_sha256=config_sha,
        signature=FeatureDatasetBuilder(config)._signature(),
        source_root=str(Path(config['dataset_root']).resolve()),
        output_dir=str(Path(output_dir).resolve()), working_snapshot_count=len(rows),
        subject_count=len({row['subject_id'] for row in rows}),
        acquisition_planned_count=45,
        missing_acquisition_video_ids=sorted(planned_ids - working_ids), sources=rows,
        extraction_program_sha256=extraction_program_hashes())
    expected = _json_value(expected)
    if marker.exists():
        frozen = _read_json(marker)
        identity = {key: value for key, value in frozen.items() if key != 'snapshot_sha256'}
        if frozen.get('snapshot_sha256') != _hash_json(identity):
            raise ValueError('Snapshot identity hash mismatch')
        current = {key: value for key, value in frozen.items()
                   if key not in ('created_at_utc', 'snapshot_sha256')}
        if current != expected:
            raise ValueError('Frozen snapshot input drift mismatch')
        for name, key in (('manifest.parquet', 'manifest_file_sha256'),
                          ('preprocessing.yaml', 'config_file_sha256')):
            copy = directory / name
            if not copy.is_file() or digest_file(copy)[0] != frozen[key]:
                raise ValueError(f'Frozen {name} hash mismatch')
        return frozen
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Incomplete existing freeze mismatch; refusing overwrite')
    root = Path(expected['source_root'])
    for row in rows:
        source = FeatureDatasetBuilder._source(root, row)
        if (digest_file(source)[0].lower() != row['sha256'].lower()
                or source.stat().st_size != int(row['size_bytes'])):
            raise ValueError(f"Source hash/size mismatch: {row['video_id']}")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, directory / 'manifest.parquet')
    shutil.copyfile(config_path, directory / 'preprocessing.yaml')
    for name, key in (('manifest.parquet', 'manifest_file_sha256'),
                      ('preprocessing.yaml', 'config_file_sha256')):
        if digest_file(directory / name)[0] != expected[key]:
            raise ValueError(f'Input changed while freezing {name}')
    if (digest_file(config_path)[0] != config_sha
            or digest_file(manifest_path)[0] != manifest_sha):
        raise ValueError('Input changed while copying snapshot inputs')
    expected['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    expected['snapshot_sha256'] = _hash_json(expected)
    _write_json(marker, expected)
    return expected


def run_snapshot(config_path: Path, manifest_path: Path, run_dir: Path,
                 output_dir: Path) -> dict:
    """Verify the freeze, then revalidate every pair through the existing builder."""
    run_started = perf_counter()
    frozen = freeze_snapshot(config_path, manifest_path, run_dir, output_dir)
    directory = Path(run_dir).resolve()
    config = load_config(directory / 'snapshot/preprocessing.yaml')
    manifest = pd.read_parquet(directory / 'snapshot/manifest.parquet')
    builder = FeatureDatasetBuilder(config)
    report = dict(schema_version='working_snapshot_run_v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        snapshot_sha256=frozen['snapshot_sha256'],
        extraction_fingerprint=frozen['signature']['extraction_fingerprint'],
        working_manifest_sha256=frozen['working_manifest_sha256'],
        working_snapshot_count=frozen['working_snapshot_count'],
        subject_count=frozen['subject_count'],
        selected_count=frozen['working_snapshot_count'],
        class_counts=dict(Counter(str(row['label_id']) for row in frozen['sources'])),
        output_dir=frozen['output_dir'],
        output_size_bytes=0,
        acquisition_planned_count=frozen['acquisition_planned_count'],
        missing_acquisition_video_ids=frozen['missing_acquisition_video_ids'],
        videos=[dict(video_id=row['video_id'], subject_id=row['subject_id'],
                     status='pending', output_current=False) for row in frozen['sources']])

    def checkpoint() -> None:
        counts = Counter(item['status'] for item in report['videos'])
        report['totals'] = {status: counts[status] for status in ('pending', 'completed', 'cached', 'failed')}
        report['status'] = ('running' if counts['pending'] else
                            'failed' if counts['failed'] else 'completed')
        report['elapsed_seconds'] = perf_counter() - run_started
        _write_json(directory / 'report.json', report)

    checkpoint()
    for index, row in enumerate(frozen['sources']):
        started = perf_counter()
        try:
            if (digest_file(directory / 'snapshot/preprocessing.yaml')[0] != frozen['config_file_sha256']
                    or _json_value(builder._signature()) != frozen['signature']
                    or extraction_program_hashes() != frozen['extraction_program_sha256']):
                raise ValueError('Frozen extraction provenance drift before member')
            result = builder.build(manifest, Path(frozen['output_dir']), video_ids=(row['video_id'],))
            items = result['videos']
            if (len(items) != 1 or items[0]['video_id'] != row['video_id']
                    or items[0]['status'] not in ('completed', 'cached', 'failed')):
                raise ValueError('Builder returned an invalid selected-member report')
            item = dict(items[0])
            item.setdefault('subject_id', row['subject_id'])
            if (item['status'] in ('completed', 'cached')
                    and (item.get('output_current') is not True
                         or item.get('extraction_fingerprint') != frozen['signature']['extraction_fingerprint'])):
                item.update(status='failed', output_current=False,
                            error='Builder success does not match frozen provenance')
        except Exception as exc:
            item = dict(video_id=row['video_id'], subject_id=row['subject_id'],
                        status='failed', output_current=False, error=str(exc))
        item['elapsed_seconds'] = perf_counter() - started
        report['videos'][index] = _json_value(item)
        checkpoint()
    output = Path(frozen['output_dir'])
    report['output_size_bytes'] = sum(
        path.stat().st_size
        for row in frozen['sources']
        for path in builder._paths(output, row['video_id'])
        if path.is_file())
    checkpoint()
    return report
