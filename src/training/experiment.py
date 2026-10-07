"""Verified artifact boundary and shared preparation for recorded-fold experiments.

All checks run before fitting. Identifiers stay outside classifier inputs; rejected
windows remain in coverage, and test support never controls model selection.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import PROJECT_ROOT, load_config
from src.datasets.acquisition import digest_file
from src.datasets.derived import DERIVED_SCHEMA, atomic_json, read_temporal
from src.datasets.normalization import FEATURE_NAMES, fit_scaler, scaler_hash
from src.datasets.sequence import (INDEX_DTYPES, _load_video, build_window_index,
                                   unique_train_timesteps)
from src.datasets.splits import hash_payload
from src.datasets.summaries import summarize_window
from src.evaluation.evaluator import ModelEvaluator
from src.preprocessing.builder import FeatureDatasetBuilder, _read_json
from src.preprocessing.snapshot import extraction_program_hashes

ROLES = ('train', 'validation', 'test')
IDENTIFIERS = frozenset(INDEX_DTYPES) | {'mode', 'outer_index', 'schema_version',
    'derived_hash', 'split_hash', 'snapshot_sha256', 'dataset_manifest_hash'}


class BlockedExperiment(ValueError):
    def __init__(self, reasons: list[str], coverage: dict):
        self.reasons = sorted(set(reasons))
        self.coverage = coverage
        super().__init__('Blocked experiment: ' + ', '.join(self.reasons))


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def parser(description: str, *, lstm: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--derived-manifest', type=Path, required=True)
    result.add_argument('--outer-index', type=int, choices=range(5), required=True)
    result.add_argument('--mode', choices=('P0', 'P1'), required=True)
    result.add_argument('--config', type=Path, default=Path('configs/training.yaml'))
    result.add_argument('--run-dir', type=Path, required=True)
    if lstm:
        result.add_argument('--cpu-threads', type=int, default=min(4, os.cpu_count() or 1))
    return result


def validate_manifest_identity(manifest: dict, *, mode: str, outer_index: int, config: dict) -> None:
    if manifest.get('dataset_manifest_hash') != hash_payload({k:v for k,v in manifest.items() if k != 'dataset_manifest_hash'}):
        raise ValueError('Dataset manifest hash mismatch')
    if manifest.get('status') != 'complete':
        raise ValueError('Derived manifest is not complete')
    if manifest.get('mode') != mode:
        raise ValueError('Derived manifest mode mismatch')
    if manifest.get('outer_index') != outer_index or type(manifest.get('outer_index')) is not int:
        raise ValueError('Derived manifest outer index mismatch')
    if manifest.get('schema_version') != 'facial_features_v1' or config.get('schema_version', 'facial_features_v1') != 'facial_features_v1':
        raise ValueError('Unsupported derived/config schema')
    if tuple(manifest.get('feature_names', ())) != FEATURE_NAMES or tuple(config.get('feature_names', FEATURE_NAMES)) != FEATURE_NAMES:
        raise ValueError('Derived/config feature order mismatch')
    identity = manifest.get('identity', {})
    if hash_payload(identity) != manifest.get('derived_hash'):
        raise ValueError('Derived identity hash mismatch')
    for key in ('mode', 'split_hash', 'snapshot_sha256', 'feature_names'):
        if identity.get(key) != manifest.get(key):
            raise ValueError(f'Derived identity {key} mismatch')
    if hash_payload(identity.get('temporal_config')) != manifest.get('temporal_hash'):
        raise ValueError('Temporal policy hash mismatch')
    for key, expected in (('sequence_fps', 10), ('sequence_window_s', 10), ('stride_s', 1),
                          ('max_missing_ratio', .20), ('max_gap_s', 1)):
        if float(config.get(key, expected)) != expected:
            raise ValueError(f'{key} differs from frozen window contract')


def _check_hash(path: Path, expected: str, description: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError(f'Invalid {description} SHA256')
    if digest_file(path)[0] != expected:
        raise ValueError(f'{description} SHA256 mismatch: {path}')


def _hashed_json(path: Path, key: str) -> dict:
    value = _read_json(path)
    if value.get(key) != hash_payload({k:v for k,v in value.items() if k != key}):
        raise ValueError(f'{key} identity mismatch: {path}')
    return value


def validate_dataset(path: Path, *, mode: str, outer_index: int, config: dict) -> tuple[dict, pd.DataFrame, dict]:
    """Reopen bytes, source membership, producer identities and every window receipt."""
    manifest = _read_json(project_path(path))
    validate_manifest_identity(manifest, mode=mode, outer_index=outer_index, config=config)
    split = _hashed_json(project_path(manifest['split_path']), 'split_hash')
    snapshot = _hashed_json(project_path(manifest['snapshot_path']), 'snapshot_sha256')
    for key in ('split_hash', 'snapshot_sha256', 'outer_index'):
        if split.get(key) != manifest.get(key):
            raise ValueError(f'Split/manifest {key} mismatch')
    if snapshot['snapshot_sha256'] != manifest['snapshot_sha256'] or snapshot['sources'] != split['sources']:
        raise ValueError('Snapshot/source membership mismatch')
    if snapshot['extraction_program_sha256'] != extraction_program_hashes():
        raise ValueError('Frozen raw producer identity drift')
    raw_dir = project_path(manifest['raw_dir'])
    code = manifest['identity']['code_sha256']
    expected_code = {'src/calibration/profile.py', 'src/datasets/splits.py',
        'src/features/temporal.py', 'src/datasets/derived.py',
        'src/datasets/sequence.py', 'src/datasets/summaries.py'}
    if set(code) != expected_code:
        raise ValueError('Incomplete derived producer identities')
    for name, expected in code.items():
        _check_hash(project_path(name), expected, 'Derived producer '+name)
    source_root = project_path(manifest['source_root'])
    if raw_dir != project_path(snapshot['output_dir']) or source_root != project_path(snapshot['source_root']):
        raise ValueError('Raw/source root differs from frozen snapshot')
    policy = split['qc_policy']
    if policy['policy_hash'] != hash_payload({k:v for k,v in policy.items() if k != 'policy_hash'}) or policy['policy_hash'] != manifest['profile_policy_hash']:
        raise ValueError('Profile QC policy identity mismatch')
    if manifest['identity']['profiles_hash'] != hash_payload(split['profiles'][mode]):
        raise ValueError('Calibration profile identity mismatch')
    commits = {entry['video_id']: entry for entry in manifest['identity']['raw_commits']}
    videos = {entry['video_id']: entry for entry in manifest['videos']}
    sources = {entry['video_id']: entry for entry in split['sources']}
    if len(commits) != len(manifest['identity']['raw_commits']) or len(videos) != len(manifest['videos']) or len(sources) != len(split['sources']) or set(commits) != set(sources) or set(videos) != set(sources):
        raise ValueError('Raw/derived/source membership mismatch or duplicate video')
    membership = {video:role for role, members in split['roles'].items() for video in members['video_ids']}
    if set(membership) != set(sources) or sum(len(m['video_ids']) for m in split['roles'].values()) != len(sources):
        raise ValueError('Split video role membership mismatch')
    subjects = [set(split['roles'][role]['subject_ids']) for role in ROLES]
    if any(subjects[a] & subjects[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Split subjects overlap')
    index_path = project_path(manifest['index_path'])
    _check_hash(index_path, manifest['index_sha256'], 'Window index')
    index = pd.read_parquet(index_path)
    if list(index.columns) != list(INDEX_DTYPES) or index.duplicated().any() or not index['split'].isin(ROLES).all() or not index['label_id'].isin((0,1,2)).all():
        raise ValueError('Invalid shared window index schema/rows')
    reconstructed = []
    coverage = manifest['coverage']
    totals = {key:0 for key in ('scheduled', 'accepted', 'rejected')}
    role_totals = {role:dict(totals) for role in ROLES}
    for video, source in sources.items():
        raw, marker = FeatureDatasetBuilder._paths(raw_dir, video)
        _check_hash(raw, commits[video]['parquet_sha256'], 'Raw Parquet '+video)
        _check_hash(marker, commits[video]['metadata_sha256'], 'Raw commit '+video)
        source_path = FeatureDatasetBuilder._source(source_root, source)
        sha, _crc = digest_file(source_path)
        size = source_path.stat().st_size
        if sha != source['sha256'] or size != source['size_bytes']:
            raise ValueError('Source SHA256/size drift: '+video)
        if FeatureDatasetBuilder._cached(raw, marker, source, sha, size, snapshot['signature']) is None:
            raise ValueError('Invalid committed raw artifact: '+video)
        entry = videos[video]
        profile = split['profiles'][mode][source['subject_id']]
        if entry['profile_valid'] != profile['valid'] or entry['profile_reasons'] != profile['quality_stats'].get('reasons', []):
            raise ValueError('Derived profile validity receipt mismatch: '+video)
        for key in ('subject_id', 'label_id'):
            if entry[key] != source[key]:
                raise ValueError('Derived/source '+key+' mismatch: '+video)
        if entry['split'] != membership[video] or source['subject_id'] not in split['roles'][entry['split']]['subject_ids']:
            raise ValueError('Derived/source role mismatch: '+video)
        if entry['profile_hash'] != hash_payload(split['profiles'][mode][source['subject_id']]):
            raise ValueError('Derived profile mismatch: '+video)
        entry['path'] = str(project_path(entry['path']))
        _check_hash(Path(entry['path']), entry['parquet_sha256'], 'Derived Parquet '+video)
        schema = pq.read_schema(entry['path'])
        # Parquet normalizes list child names from "item" to "element".
        # Logical field order, types, fixed widths and top-level metadata remain strict.
        if not schema.equals(DERIVED_SCHEMA) or schema.metadata != DERIVED_SCHEMA.metadata:
            raise ValueError('Derived Arrow schema mismatch: '+video)
        frame = _load_video(entry)
        rebuilt = build_window_index(frame, split=split, config=config)
        reconstructed.append(rebuilt)
        counts = rebuilt.attrs['coverage']['by_video'].get(video, dict(scheduled=0, accepted=0, rejected=0, reasons={}))
        if counts != coverage['by_video'].get(video):
            raise ValueError('Video coverage receipt mismatch: '+video)
        for key in totals:
            totals[key] += counts[key]
            role_totals[entry['split']][key] += counts[key]
    rebuilt = pd.concat(reconstructed, ignore_index=True)
    sort_columns = ['video_id', 'segment_id', 'start_row']
    try:
        pd.testing.assert_frame_equal(index.sort_values(sort_columns).reset_index(drop=True),
            rebuilt.sort_values(sort_columns).reset_index(drop=True), check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Shared window index differs from verified derived data') from exc
    if set(coverage['by_video']) != set(sources) or any(coverage.get(k) != v for k,v in totals.items()):
        raise ValueError('Global coverage receipt mismatch')
    for role in ROLES:
        if any(coverage['by_role'][role].get(k) != v for k,v in role_totals[role].items()):
            raise ValueError('Role coverage receipt mismatch: '+role)
        support = {str(label):int((index.split.eq(role) & index.label_id.eq(label)).sum()) for label in (0,1,2)}
        if manifest['class_support'][role] != support:
            raise ValueError('Class support receipt mismatch: '+role)
    # Deterministic chronological validation/test order, independent of producer order.
    index = index.sort_values(['video_id', 'segment_id', 'start_timestamp_ms']).reset_index(drop=True)
    return manifest, index, split


def check_eligibility(manifest: dict, index: pd.DataFrame) -> None:
    eligibility = manifest['eligibility']
    if eligibility.get('status') not in ('eligible', 'blocked'):
        raise ValueError('Invalid eligibility status')
    reasons = list(eligibility.get('reasons', []))
    for role in ('train', 'validation'):
        for label in (0,1,2):
            if not (index.split.eq(role) & index.label_id.eq(label)).any():
                reasons.append(f'{role}_missing_class_{label}')
    if not index.split.eq('test').any():
        reasons.append('no_accepted_test_windows')
    if eligibility['status'] == 'blocked' and not reasons:
        reasons.append('manifest_declares_blocked')
    if reasons:
        raise BlockedExperiment(reasons, manifest['coverage'])


def fit_train_scaler(index: pd.DataFrame, manifest: dict) -> dict:
    try:
        scaler = fit_scaler(unique_train_timesteps(index, manifest), FEATURE_NAMES)
    except ValueError as exc:
        if str(exc).startswith('blocked scaler:'):
            raise BlockedExperiment([str(exc)], manifest['coverage']) from exc
        raise
    for key in ('mode', 'schema_version', 'derived_hash', 'split_hash', 'snapshot_sha256', 'outer_index'):
        scaler[key] = manifest[key]
    scaler['fit_scope'] = 'unique_accepted_train_timesteps'
    scaler['index_sha256'] = manifest.get('index_sha256')
    scaler['hash'] = scaler_hash(scaler)
    return scaler


def build_summaries(index: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, tuple[str, ...]]:
    entries = {entry['video_id']:entry for entry in manifest['videos']}
    records, names = [], None
    for video, windows in index.groupby('video_id', sort=False):
        samples = read_temporal(Path(entries[video]['path']))
        timestamps = [sample.timestamp_ms for sample in samples]
        for row in windows.itertuples(index=False):
            # The nominal interval is [end-10000,end); not the 9.9s tick span.
            start_ms, end_ms = int(row.end_timestamp_ms)-10000, int(row.end_timestamp_ms)
            selected = samples[bisect_left(timestamps, start_ms):bisect_right(timestamps, end_ms)]
            summary = summarize_window(selected, start_ms=start_ms, end_ms=end_ms)
            if names is None:
                names = tuple(summary)
            if tuple(summary) != names:
                raise ValueError('Summary schema changed between accepted windows')
            records.append({**row._asdict(), **summary})
    table = pd.DataFrame(records)
    if len(table) != len(index) or names is None:
        raise ValueError('Accepted summary support mismatch')
    return table, names


def summary_features(table: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    if not names or set(names) & IDENTIFIERS:
        raise ValueError('Summary identifier metadata cannot enter X')
    return table.loc[:, list(names)]


def hashes(manifest: dict, scaler: dict | None = None) -> dict:
    result = {name:manifest[key] for name,key in dict(snapshot='snapshot_sha256',
        split='split_hash', profile_policy='profile_policy_hash', derived='derived_hash',
        temporal='temporal_hash').items()}
    if scaler is not None:
        result['scaler'] = scaler['hash']
    return result


def environment() -> dict:
    packages = {}
    for name in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'torch', 'joblib'):
        packages[name] = importlib.metadata.version(name)
    return dict(python=platform.python_version(), platform=platform.platform(),
                executable=sys.executable, versions=packages, cpu_count=os.cpu_count())


def source_hashes(model: str) -> dict:
    names = ['src/training/experiment.py', 'src/datasets/normalization.py',
             'src/evaluation/evaluator.py', f'scripts/train_{model}.py', f'src/models/{model}.py']
    if model == 'lstm':
        names += ['src/training/trainer.py', 'src/models/bundle.py']
    return {name:digest_file(PROJECT_ROOT/name)[0] for name in names}


def prepare_run(args, model: str) -> tuple[dict, Path, dict]:
    run = project_path(args.run_dir)
    # Never overwrite a frozen run or leave an old checkpoint beside a blocker.
    if run.exists() and any(run.iterdir()):
        raise ValueError('Run directory must be new or empty: '+str(run))
    run.mkdir(parents=True, exist_ok=True)
    config = load_config(project_path(args.config))
    config['calibration_mode'] = args.mode
    config['feature_names'] = list(config.get('feature_names', FEATURE_NAMES))
    if hasattr(args, 'cpu_threads'):
        config['cpu_threads'] = args.cpu_threads
        config['num_workers'] = 0
        config['class_weights'] = config.get('class_weights', False)
    report = dict(status='failed', model=model, mode=args.mode, outer_index=args.outer_index,
        config=config, config_hash=hash_payload(config), environment=environment(),
        command=[sys.executable, '-m', 'scripts.train_'+model,
            '--derived-manifest', str(project_path(args.derived_manifest)),
            '--outer-index', str(args.outer_index), '--mode', args.mode,
            '--config', str(project_path(args.config)), '--run-dir', str(run)],
        derived_manifest=str(project_path(args.derived_manifest)), artifacts={},
        source_sha256=source_hashes(model))
    if hasattr(args, 'cpu_threads'):
        import torch
        report['command'] += ['--cpu-threads', str(args.cpu_threads)]
        report['environment'].update(torch_num_threads=torch.get_num_threads(),
                                     torch_interop_threads=torch.get_num_interop_threads())
    atomic_json(run/'resolved_config.json', config)
    atomic_json(run/'seed.json', dict(seed=config.get('seed', 42)))
    atomic_json(run/'environment.json', report['environment'])
    return config, run, report


def save_predictions(run: Path, role: str, rows: pd.DataFrame, probabilities: np.ndarray,
                     manifest: dict, identities: dict) -> tuple[dict, Path]:
    if probabilities.shape != (len(rows), 3):
        raise ValueError('Accepted prediction support mismatch')
    metadata = dict(class_order=[0,1,2], mode=manifest['mode'], outer_index=manifest['outer_index'],
        schema_version=manifest['schema_version'], hashes=identities,
        dataset_manifest_hash=manifest['dataset_manifest_hash'], split=role)
    metrics = ModelEvaluator().evaluate(rows.label_id.to_numpy(), probabilities,
        metadata=metadata, coverage=dict(manifest['coverage']['by_role'][role]))
    table = rows.loc[:, list(INDEX_DTYPES)].copy()
    for name, value in metadata.items():
        if name in ('class_order', 'hashes'):
            continue
        table[name] = value
    for label in (0,1,2):
        table[f'probability_{label}'] = probabilities[:, label]
    table['prediction_id'] = probabilities.argmax(axis=1) if len(probabilities) else np.empty(0, dtype=int)
    path = run/(role+'_predictions.parquet')
    table.to_parquet(path, index=False)
    atomic_json(run/(role+'_metrics.json'), metrics)
    return metrics, path
