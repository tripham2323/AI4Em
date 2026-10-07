"""Subject-disjoint official restricted splits with train-only QC and full rejection receipts."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import numpy as np
import pandas as pd
from src.contracts import CalibrationProfile
from src.calibration.profile import BASELINE_FIELDS, SCHEMA_VERSION, estimate_profile, fit_qc_policy


def strict_json_value(value):
    """Represent missing numbers as JSON null, never JavaScript NaN/Infinity."""
    if isinstance(value,dict):
        return {str(k):strict_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [strict_json_value(v) for v in value]
    if isinstance(value,np.ndarray):
        return strict_json_value(value.tolist())
    if isinstance(value,np.generic):
        return strict_json_value(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def hash_payload(value) -> str:
    return hashlib.sha256(json.dumps(strict_json_value(value),sort_keys=True,
                                   separators=(',',':'),allow_nan=False).encode()).hexdigest()


def profile_from_dict(record: dict) -> CalibrationProfile:
    fields=dict(record)
    for name in BASELINE_FIELDS:
        if fields.get(name) is None:
            fields[name]=float('nan')
    fields['image_size']=tuple(fields['image_size'])
    return CalibrationProfile(**fields)


def make_outer_split(manifest: pd.DataFrame, test_fold: int) -> dict:
    if isinstance(test_fold,bool) or test_fold not in range(1,6):
        raise ValueError('test_fold must be an official fold 1..5')
    required={'subject_id','video_id','fold_id','fold_source','label_id'}
    if manifest.empty or not required<=set(manifest):
        raise ValueError('Empty manifest or missing split identity columns')
    if manifest[['subject_id','video_id','fold_id','fold_source']].isna().any().any():
        raise ValueError('Unknown official fold or missing source identity')
    if (manifest['video_id'].duplicated().any() or
        not manifest['fold_id'].map(lambda v:not isinstance(v,bool) and v in range(1,6)).all() or
        not manifest['fold_source'].eq('official_archive_index').all() or
        manifest.groupby('subject_id')['fold_id'].nunique().ne(1).any()):
        raise ValueError('Duplicate source or inconsistent/unofficial subject fold')
    validation=test_fold%5+1
    train=[f for f in range(1,6) if f not in (test_fold,validation)]
    roles={}
    for role,folds in (('train',train),('validation',[validation]),('test',[test_fold])):
        members=manifest[manifest['fold_id'].isin(folds)]
        roles[role]=dict(subject_ids=sorted(members['subject_id'].unique().tolist()),
                         video_ids=sorted(members['video_id'].tolist()))
    return dict(test_fold=int(test_fold),outer_index=int(test_fold)-1,validation_fold=validation,
                train_folds=train,roles=roles,sources=strict_json_value(manifest.to_dict('records')))


def _provenance(metadata):
    return dict(schema_version=metadata.get('schema_version'),
                image_size=metadata.get('image_size'),artifact_hashes=metadata.get('artifact_hashes'),
                camera_metadata=metadata.get('camera_metadata'))


def _failed(mode,reason,metadata=None):
    metadata=metadata or {}
    return CalibrationProfile(mode,False,*([float('nan')]*6),
        metadata.get('schema_version',SCHEMA_VERSION),
        metadata.get('artifact_hashes',{}).get('asset_sha256',
            metadata.get('artifact_hashes',{}).get('landmarker','')),
        tuple(metadata.get('image_size',[0,0])),
        dict(reasons=[reason],prefix_start_ms=0,prefix_end_ms=None,valid_duration_ms=0))


def build_splits(manifest: pd.DataFrame, samples_by_video: dict, metadata_by_video: dict,
                 *, snapshot_sha256: str, epsilon: float=1e-6) -> list[dict]:
    """All five slots. Profile support is preliminary, not class/window eligibility."""
    artifacts=[]
    for fold in range(1,6):
        split=make_outer_split(manifest,fold)
        train_ids=set(split['roles']['train']['video_ids'])
        train_alert=sorted(row['video_id'] for row in split['sources']
                           if row['video_id'] in train_ids and row['label_id']==0)
        training=[s for video in train_alert for s in samples_by_video.get(video,[])]
        qc=fit_qc_policy(training,epsilon=epsilon)
        qc['train_subject_ids']=split['roles']['train']['subject_ids']
        qc['train_alert_video_ids']=train_alert
        qc['policy_hash']=hash_payload(qc)
        exemplar=next((v for v in train_alert if v in metadata_by_video),None)
        if exemplar is None:
            population=_failed('P0','missing_train_alert_metadata')
        else:
            meta=metadata_by_video[exemplar]
            population=estimate_profile(training,'P0',image_size=tuple(meta['image_size']),
                asset_sha256=meta['artifact_hashes'].get('asset_sha256',meta['artifact_hashes'].get('landmarker','')),
                qc_policy=qc)
            population=replace(population,quality_stats={**population.quality_stats,
                'binding_source_id':exemplar,'provenance':_provenance(meta),
                'qc_policy_hash':qc['policy_hash']})
        profiles={'P0':{},'P1':{}}
        reserved={'P0':{},'P1':{}}
        for subject in sorted(manifest['subject_id'].unique()):
            profiles['P0'][subject]=asdict(population)
            members=[row for row in split['sources'] if row['subject_id']==subject]
            alerts=sorted(row['video_id'] for row in members if row['label_id']==0)
            if not alerts:
                personal=_failed('P1','missing_alert')
            else:
                # Select one deterministic Alert source; never select another label or fit on future clips.
                video=alerts[0]
                meta=metadata_by_video.get(video)
                if meta is None:
                    personal=_failed('P1','missing_alert_metadata')
                    reserved['P1'][video]=dict(start_ms=0,end_ms=60000)
                else:
                    personal=estimate_profile(samples_by_video.get(video,[]),'P1',
                        image_size=tuple(meta['image_size']),
                        asset_sha256=meta['artifact_hashes'].get('asset_sha256',meta['artifact_hashes'].get('landmarker','')),
                        qc_policy=qc)
                    reserved['P1'][video]=dict(start_ms=0,end_ms=personal.quality_stats['prefix_end_ms'])
                    reasons=list(personal.quality_stats['reasons'])
                    if (meta.get('schema_version')!=SCHEMA_VERSION or
                            any(v['video_id'] not in metadata_by_video or
                                _provenance(metadata_by_video[v['video_id']])!=_provenance(meta)
                                for v in members)):
                        reasons.append('provenance_mismatch')
                    personal=replace(personal,valid=personal.valid and not reasons,
                        quality_stats={**personal.quality_stats,'reasons':sorted(set(reasons)),
                            'alert_video_id':video,'provenance':_provenance(meta),
                            'qc_policy_hash':qc['policy_hash']})
            profiles['P1'][subject]=asdict(personal)
        protocols={}
        for mode in ('P0','P1'):
            accepted={}; rejected={}; reasons=[]
            for role,members in split['roles'].items():
                accepted[role]=[s for s in members['subject_ids'] if profiles[mode][s]['valid']]
                rejected[role]={s:profiles[mode][s]['quality_stats']['reasons']
                                for s in members['subject_ids'] if not profiles[mode][s]['valid']}
                if not accepted[role]:
                    reasons.append('no_accepted_'+role+'_profiles')
            protocols[mode]=dict(status='blocked' if reasons else 'preliminary_profile_support',
                reasons=reasons,accepted_subject_ids=accepted,rejected_subjects=rejected,
                eligibility_scope='profile_only; window/class support not yet assessed',
                profile_hash=hash_payload(profiles[mode]))
        split.update(snapshot_sha256=snapshot_sha256,profiles=profiles,reserved_ranges=reserved,
                     qc_policy=qc,protocols=protocols)
        split['split_hash']=hash_payload(split)
        artifacts.append(strict_json_value(split))
    return artifacts
