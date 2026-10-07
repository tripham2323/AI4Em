"""Build all five restricted official split slots from audited frozen raw pairs; never extracts."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.contracts import FeatureSample
from src.datasets.acquisition import digest_file
from src.datasets.splits import build_splits, hash_payload, strict_json_value
from src.preprocessing.audit import audit_snapshot
from src.preprocessing.builder import FeatureDatasetBuilder, _read_json


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(strict_json_value(value),indent=2,sort_keys=True,allow_nan=False)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix=path.name+'.',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/phase9'))
    parser.add_argument('--snapshot-run',type=Path,default=Path('runs/phase8'))
    parser.add_argument('--config',type=Path,default=Path('configs/training.yaml'))
    parser.add_argument('--split-dir',type=Path,default=Path('data/splits'))
    args=parser.parse_args(argv)
    run_dir,snapshot_run,config_path,split_dir=map(_resolve,
        (args.run_dir,args.snapshot_run,args.config,args.split_dir))
    report={'schema_version':'restricted_split_build_v1','status':'failed',
            'eligibility_scope':'preliminary profile support only; no window/class claims',
            'artifacts':[]}
    try:
        config_sha=digest_file(config_path)[0]
        with config_path.open(encoding='utf-8') as handle:
            config=yaml.safe_load(handle)
        if not isinstance(config,dict) or config.get('schema_version')!='facial_features_v1':
            raise ValueError('Invalid training feature schema/config')
        frozen=snapshot_run/'snapshot'
        snapshot=_read_json(frozen/'snapshot.json')
        raw_dir=Path(snapshot['output_dir']).resolve()
        if (split_dir==raw_dir or split_dir.is_relative_to(raw_dir) or
                split_dir==snapshot_run or split_dir.is_relative_to(snapshot_run) or
                run_dir==raw_dir or run_dir.is_relative_to(raw_dir) or
                run_dir==snapshot_run or run_dir.is_relative_to(snapshot_run)):
            print('Split/report destinations must not overwrite frozen/raw artifacts')
            return 1
        audit=audit_snapshot(frozen/'preprocessing.yaml',frozen/'manifest.parquet',raw_dir,
                             snapshot,_read_json(snapshot_run/'report.json'))
        _atomic_json(run_dir/'input_audit.json',audit)
        if audit['status']!='complete':
            raise ValueError('Frozen input audit failed; inspect input_audit.json; no splits published')
        manifest=pd.read_parquet(frozen/'manifest.parquet')
        raw={}; metadata={}
        sample_fields={f.name for f in fields(FeatureSample)}
        for source in snapshot['sources']:
            video=source['video_id']
            parquet,marker=FeatureDatasetBuilder._paths(raw_dir,video)
            meta=_read_json(marker)
            table=pd.read_parquet(parquet)
            rows=[]
            for record in table.to_dict('records'):
                record['source_id']=video
                rows.append(FeatureSample(**{key:(float('nan') if record[key] is None else record[key])
                                             for key in sample_fields}))
            raw[video]=rows
            metadata[video]=meta
        epsilon=snapshot['signature']['effective_config'].get('epsilon',1e-6)
        artifacts=build_splits(manifest,raw,metadata,snapshot_sha256=snapshot['snapshot_sha256'],
                               epsilon=epsilon)
        if digest_file(config_path)[0]!=config_sha:
            raise ValueError('Training config changed during split build')
        # Each split is atomically replaced; completed report is the publication receipt for all five.
        for split in artifacts:
            split['training_config_sha256']=config_sha
            split['training_config']=config
            split['split_hash']=hash_payload({k:v for k,v in split.items() if k!='split_hash'})
            destination=split_dir/f"outer_{split['outer_index']}.json"
            _atomic_json(destination,split)
            report['artifacts'].append(dict(path=str(destination),test_fold=split['test_fold'],
                outer_index=split['outer_index'],split_hash=split['split_hash'],
                qc_policy_hash=split['qc_policy']['policy_hash'],protocols=split['protocols'],
                membership_subject_counts={r:len(m['subject_ids']) for r,m in split['roles'].items()}))
        report.update(status='completed',snapshot_sha256=snapshot['snapshot_sha256'],
                      source_count=len(manifest),subject_count=int(manifest['subject_id'].nunique()),
                      training_config_sha256=config_sha)
    except Exception as exc:
        report['error']=str(exc)
    try:
        _atomic_json(run_dir/'report.json',report)
    except Exception as exc:
        print(f'Failed to publish split build receipt: {exc}')
        return 1
    print(json.dumps(strict_json_value(report),indent=2,sort_keys=True,allow_nan=False))
    return int(report['status']!='completed')


if __name__=='__main__':
    raise SystemExit(main())
