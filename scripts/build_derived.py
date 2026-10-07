"""Publish all restricted outer-slot temporal caches and shared window indexes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.datasets.acquisition import digest_file
from src.datasets.derived import atomic_json, build_derived
from src.datasets.sequence import build_window_index
from src.datasets.splits import hash_payload
from src.preprocessing.audit import audit_snapshot
from src.preprocessing.builder import _read_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, default=PROJECT_ROOT/'data/splits')
    parser.add_argument('--snapshot-run', type=Path, default=PROJECT_ROOT/'runs/phase8')
    parser.add_argument('--mode', choices=('P0','P1'), required=True)
    parser.add_argument('--outer-index', type=int, choices=range(5))
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT/'data/processed')
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/training.yaml')
    parser.add_argument('--temporal-config', type=Path, default=PROJECT_ROOT/'configs/temporal.yaml')
    parser.add_argument('--run-dir', type=Path, default=PROJECT_ROOT/'runs/phase11/derived')
    args = parser.parse_args(argv)
    for name in ('split_dir', 'snapshot_run', 'output_root', 'config', 'temporal_config', 'run_dir'):
        value = getattr(args, name)
        setattr(args, name, value.resolve() if value.is_absolute() else (PROJECT_ROOT/value).resolve())
    report = dict(mode=args.mode, status='failed', artifacts=[])
    try:
        frozen = args.snapshot_run/'snapshot'
        snapshot = _read_json(frozen/'snapshot.json')
        audit = audit_snapshot(frozen/'preprocessing.yaml', frozen/'manifest.parquet',
            Path(snapshot['output_dir']), snapshot, _read_json(args.snapshot_run/'report.json'))
        if audit['status'] != 'complete':
            raise ValueError('Frozen raw audit failed before deriving')
        config = load_config(args.config)
        config['calibration_mode'] = args.mode
        policy = load_config(args.temporal_config)
        for index in (range(5) if args.outer_index is None else [args.outer_index]):
            split = _read_json(args.split_dir/f'outer_{index}.json')
            manifest = build_derived(split, split['profiles'][args.mode],
                raw_dir=Path(snapshot['output_dir']), snapshot=snapshot,
                temporal_config=policy, output_dir=args.output_root, mode=args.mode)
            directory = Path(manifest['manifest_path']).parent
            frames = [pd.read_parquet(v['path']) for v in manifest['videos'] if v['rows']]
            if frames:
                temporal = pd.concat(frames, ignore_index=True)
            else:
                temporal = pd.DataFrame(columns=['video_id','subject_id','label_id','split',
                    'timestamp_ms','segment_id','values','validity','event_summaries','profile_hash'])
            windows = build_window_index(temporal, split=split, config=config)
            coverage = windows.attrs['coverage']
            path = directory/'window_index.parquet'
            windows.to_parquet(path, index=False)
            support = {role:{str(label):int((windows['split'].eq(role)&windows.label_id.eq(label)).sum())
                             for label in (0,1,2)} for role in ('train','validation','test')}
            reasons = [role+'_missing_class_'+label for role in ('train','validation')
                       for label,count in support[role].items() if not count]
            if not sum(support['test'].values()):
                reasons.append('no_accepted_test_windows')
            dataset = {**manifest, 'index_path':str(path), 'index_sha256':digest_file(path)[0],
                       'raw_dir':str(Path(snapshot['output_dir']).resolve()),
                       'source_root':str(Path(snapshot['source_root']).resolve()),
                       'snapshot_path':str((frozen/'snapshot.json').resolve()),
                       'coverage':coverage, 'class_support':support,
                       'eligibility':dict(status='blocked' if reasons else 'eligible', reasons=reasons),
                       'profile_policy_hash':split['qc_policy']['policy_hash'],
                       'temporal_hash':hash_payload(manifest['identity']['temporal_config']),
                       'split_path':str((args.split_dir/f'outer_{index}.json').resolve())}
            dataset['dataset_manifest_hash'] = hash_payload(dataset)
            dataset_path = directory/'dataset_manifest.json'
            atomic_json(dataset_path, dataset)
            report['artifacts'].append(dict(outer_index=index, path=str(dataset_path),
                derived_hash=manifest['derived_hash'], class_support=support,
                eligibility=dataset['eligibility'], coverage=coverage))
            atomic_json(args.run_dir/(args.mode+'.json'), report)
            print(f'{args.mode} outer{index}: windows={len(windows)} status={dataset["eligibility"]["status"]}', flush=True)
        report['status'] = 'complete'
    except Exception as exc:
        report['error'] = str(exc)
        print(f'Derivation failed: {exc}', flush=True)
    atomic_json(args.run_dir/(args.mode+'.json'), report)
    return int(report['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
