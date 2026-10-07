"""Derive shared temporal caches without changing frozen raw feature commits."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import PROJECT_ROOT
from src.contracts import FeatureSample, TemporalSample
from src.datasets.acquisition import digest_file
from src.datasets.splits import hash_payload, profile_from_dict, strict_json_value
from src.features.temporal import FEATURE_NAMES, TemporalFeatureExtractor, resolve_temporal_config
from src.preprocessing.builder import FeatureDatasetBuilder, _read_json
from src.preprocessing.snapshot import extraction_program_hashes

DERIVED_SCHEMA = pa.schema([
    pa.field('video_id', pa.string()), pa.field('subject_id', pa.string()),
    pa.field('label_id', pa.int8()), pa.field('split', pa.string()),
    pa.field('timestamp_ms', pa.int64()), pa.field('segment_id', pa.int32()),
    pa.field('values', pa.list_(pa.float32(), 16)),
    pa.field('validity', pa.list_(pa.bool_(), 16)),
    pa.field('event_summaries', pa.string()), pa.field('profile_hash', pa.string())])


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(strict_json_value(value), sort_keys=True, indent=2, allow_nan=False)
    staged = path.with_name(path.name + '.tmp')
    with staged.open('w', encoding='utf-8') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)


def raw_sample(record: dict, video_id: str) -> FeatureSample:
    converted = {}
    for field in fields(FeatureSample):
        value = video_id if field.name == 'source_id' else record[field.name]
        converted[field.name] = float('nan') if value is None else value
    return FeatureSample(**converted)


def read_temporal(path: Path) -> list[TemporalSample]:
    rows = pq.read_table(path).to_pylist()
    return [TemporalSample(row['timestamp_ms'], np.asarray(row['values'], dtype=np.float32),
                           np.asarray(row['validity'], dtype=np.bool_), row['segment_id'],
                           json.loads(row['event_summaries'])) for row in rows]


def build_derived(split: dict, profiles: dict, *, raw_dir: Path, snapshot: dict,
                  temporal_config: dict, output_dir: Path, mode: str) -> dict:
    if mode not in ('P0', 'P1') or split['snapshot_sha256'] != snapshot['snapshot_sha256']:
        raise ValueError('Derived protocol/snapshot mismatch')
    if hash_payload({k:v for k,v in split.items() if k != 'split_hash'}) != split['split_hash']:
        raise ValueError('Split identity drift')
    if profiles != split['profiles'][mode]:
        raise ValueError('Profile map differs from frozen split')
    if extraction_program_hashes() != snapshot['extraction_program_sha256']:
        raise ValueError('Frozen raw producer drift')
    if split['sources'] != snapshot['sources']:
        raise ValueError('Derived source membership drift')
    policy = resolve_temporal_config(temporal_config)
    raw_dir = Path(raw_dir).resolve()
    if raw_dir != Path(snapshot['output_dir']).resolve():
        raise ValueError('Raw output path mismatch')
    roles = {video:role for role,members in split['roles'].items() for video in members['video_ids']}
    inputs = []
    for source in split['sources']:
        video = source['video_id']
        parquet, marker = FeatureDatasetBuilder._paths(raw_dir, video)
        raw_source = Path(snapshot['source_root']) / source['relative_path']
        if digest_file(raw_source)[0] != source['sha256']:
            raise ValueError(f'Source drift: {video}')
        metadata = FeatureDatasetBuilder._cached(parquet, marker, source, source['sha256'],
                                                 source['size_bytes'], snapshot['signature'])
        if metadata is None:
            raise ValueError(f'Invalid raw commit: {video}')
        inputs.append(dict(video_id=video, parquet_sha256=digest_file(parquet)[0],
                           metadata_sha256=digest_file(marker)[0]))
    code = {name:digest_file(PROJECT_ROOT/name)[0] for name in (
        'src/calibration/profile.py', 'src/datasets/splits.py', 'src/features/temporal.py',
        'src/datasets/derived.py', 'src/datasets/sequence.py', 'src/datasets/summaries.py')}
    identity = dict(version='derived_temporal_v1', snapshot_sha256=snapshot['snapshot_sha256'],
                    split_hash=split['split_hash'], mode=mode, profiles_hash=hash_payload(profiles),
                    temporal_config=policy, raw_commits=inputs, code_sha256=code,
                    feature_names=list(FEATURE_NAMES))
    derived_hash = hash_payload(identity)
    directory = Path(output_dir).resolve() / ('derived_' + derived_hash)
    if directory == raw_dir or directory.is_relative_to(raw_dir):
        raise ValueError('Derived destination must not overwrite raw features')
    marker = directory / 'manifest.json'
    if marker.exists():
        previous = _read_json(marker)
        if hash_payload({k:v for k,v in previous.items() if k != 'manifest_hash'}) != previous.get('manifest_hash'):
            raise ValueError('Derived manifest bytes drift')
        if [entry['video_id'] for entry in previous['videos']] != [s['video_id'] for s in split['sources']]:
            raise ValueError('Derived cache membership drift')
        if previous.get('identity') != identity or previous.get('derived_hash') != derived_hash:
            raise ValueError('Derived cache identity drift')
        for entry in previous['videos']:
            if digest_file(Path(entry['path']))[0] != entry['parquet_sha256']:
                raise ValueError(f'Derived cache bytes drift: {entry["video_id"]}')
            with pq.ParquetFile(entry['path']) as stored:
                if stored.metadata.num_rows != entry['rows'] or not stored.schema_arrow.equals(DERIVED_SCHEMA):
                    raise ValueError(f'Derived cache schema/count drift: {entry["video_id"]}')
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    videos = []
    for source, commit in zip(split['sources'], inputs):
        video = source['video_id']
        profile_record = profiles[source['subject_id']]
        profile = profile_from_dict(profile_record)
        profile_hash = hash_payload(profile_record)
        engine = TemporalFeatureExtractor(profile, policy)
        reserved = split['reserved_ranges'][mode].get(video)
        prefix_end = (reserved or {}).get('end_ms') or 0
        rows = []
        raw_parquet, raw_marker = FeatureDatasetBuilder._paths(raw_dir, video)
        for record in pq.read_table(raw_parquet).to_pylist():
            if record['timestamp_ms'] < prefix_end:
                continue
            for sample in engine.update(raw_sample(record, video)):
                rows.append(dict(video_id=video, subject_id=source['subject_id'],
                    label_id=source['label_id'], split=roles[video], timestamp_ms=sample.timestamp_ms,
                    segment_id=sample.segment_id, values=sample.values.tolist(),
                    validity=sample.validity.tolist(), profile_hash=profile_hash,
                    event_summaries=json.dumps(strict_json_value(sample.event_summaries),
                                               separators=(',', ':'), allow_nan=False)))
        if (digest_file(raw_parquet)[0] != commit['parquet_sha256'] or
                digest_file(raw_marker)[0] != commit['metadata_sha256']):
            raise ValueError(f'Raw commit changed during derivation: {video}')
        path = directory / (video + '.parquet')
        staged = path.with_suffix('.parquet.tmp')
        pq.write_table(pa.Table.from_pylist(rows, schema=DERIVED_SCHEMA), staged)
        os.replace(staged, path)
        videos.append(dict(video_id=video, subject_id=source['subject_id'], label_id=source['label_id'],
                           split=roles[video], path=str(path), rows=len(rows), profile_hash=profile_hash,
                           profile_valid=profile.valid, profile_reasons=profile.quality_stats.get('reasons', []),
                           reserved_prefix_end_ms=prefix_end, parquet_sha256=digest_file(path)[0]))
    if any(digest_file(PROJECT_ROOT/name)[0] != expected for name,expected in code.items()):
        raise ValueError('Derived program changed during run')
    manifest = dict(schema_version='facial_features_v1', status='complete', derived_hash=derived_hash,
                    snapshot_sha256=snapshot['snapshot_sha256'], split_hash=split['split_hash'],
                    mode=mode, outer_index=split['outer_index'], feature_names=list(FEATURE_NAMES),
                    identity=identity, videos=videos, wall_seconds=perf_counter()-started,
                    manifest_path=str(marker))
    manifest['manifest_hash'] = hash_payload(manifest)
    atomic_json(marker, manifest)
    return manifest
