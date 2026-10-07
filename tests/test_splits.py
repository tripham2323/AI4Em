import copy
import json
import math

import pandas as pd
import numpy as np
import pytest

from src.datasets.splits import build_splits, make_outer_split, profile_from_dict, strict_json_value
from test_profile import samples


def fixture():
    rows, raw, metadata = [], {}, {}
    for fold in range(1,6):
        subject = f'{fold:02}'
        video = subject+'_0'
        rows.append(dict(subject_id=subject,video_id=video,label_id=0,fold_id=fold,
                         fold_source='official_archive_index'))
        raw[video] = samples(video)
        metadata[video] = dict(image_size=[640,480], schema_version='facial_features_v1',
                               artifact_hashes={'landmarker':'a'*64},camera_metadata={'camera':'same'})
    return pd.DataFrame(rows), raw, metadata


def test_arrow_array_manifest_metadata_survives_split_publication():
    frame, _, _ = fixture()
    frame['sample_timestamps_ms'] = [np.array([0, 1000], dtype=np.int64) for _ in range(len(frame))]
    published = json.loads(json.dumps(make_outer_split(frame, 1), allow_nan=False))
    assert published['sources'][0]['sample_timestamps_ms'] == [0, 1000]


def test_all_five_subject_and_video_roles_are_disjoint():
    frame,_,_ = fixture()
    for k in range(1,6):
        split = make_outer_split(frame,k)
        assert split['outer_index'] == k-1
        assert split['validation_fold'] == k%5+1
        roles = list(split['roles'].values())
        for i,a in enumerate(roles):
            for b in roles[i+1:]:
                assert not set(a['subject_ids']) & set(b['subject_ids'])
                assert not set(a['video_ids']) & set(b['video_ids'])
        assert len(split['sources']) == 5


@pytest.mark.parametrize('mutation', ['unknown','unofficial','inconsistent'])
def test_bad_fold_provenance_is_rejected(mutation):
    frame,_,_ = fixture()
    if mutation == 'unknown': frame.loc[0,'fold_id'] = None
    if mutation == 'unofficial': frame.loc[0,'fold_source'] = 'guess'
    if mutation == 'inconsistent':
        extra=frame.iloc[0].copy(); extra['video_id']='01_5'; extra['fold_id']=2
        frame=pd.concat([frame,pd.DataFrame([extra])])
    with pytest.raises(ValueError): make_outer_split(frame,1)


def test_test_changes_cannot_change_training_qc_or_population_fit():
    frame,raw,meta=fixture()
    before=build_splits(frame,raw,meta,snapshot_sha256='snapshot')[0]
    raw['01_0']=samples('01_0',left=.9,right=.8)
    after=build_splits(frame,raw,meta,snapshot_sha256='snapshot')[0]
    assert before['qc_policy'] == after['qc_policy']
    assert before['profiles']['P0'] == after['profiles']['P0']


def test_missing_alert_and_invalid_subjects_remain_full_membership_and_blocked():
    frame,raw,meta=fixture()
    frame.loc[4,'label_id']=1
    raw['03_0']=[]
    splits=build_splits(frame,raw,meta,snapshot_sha256='snapshot')
    for split in splits:
        assert len(split['sources']) == 5
        assert not split['profiles']['P1']['05']['valid']
        assert 'missing_alert' in split['profiles']['P1']['05']['quality_stats']['reasons']
        assert '03_0' in split['reserved_ranges']['P1']
    assert splits[3]['protocols']['P1']['status'] == 'blocked'
    assert splits[4]['protocols']['P1']['status'] == 'blocked'


def test_provenance_mismatch_rejects_personal_profile():
    frame,raw,meta=fixture()
    extra=dict(frame.iloc[0]); extra.update(video_id='01_5',label_id=1)
    frame=pd.concat([frame,pd.DataFrame([extra])])
    raw['01_5']=samples('01_5'); meta['01_5']=copy.deepcopy(meta['01_0'])
    meta['01_5']['image_size']=[1280,720]
    split=build_splits(frame,raw,meta,snapshot_sha256='snapshot')[0]
    assert 'provenance_mismatch' in split['profiles']['P1']['01']['quality_stats']['reasons']


def test_failed_profile_serializes_strict_null_and_restores_nan():
    frame,raw,meta=fixture(); raw['01_0']=[]
    split=build_splits(frame,raw,meta,snapshot_sha256='snapshot')[0]
    encoded=json.dumps(strict_json_value(split),allow_nan=False)
    record=json.loads(encoded)['profiles']['P1']['01']
    assert record['ear_left_baseline'] is None
    assert math.isnan(profile_from_dict(record).ear_left_baseline)
