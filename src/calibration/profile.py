"""Pure train-QC fitting, bounded Alert-prefix estimation and AND-only transforms."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from src.contracts import CalibrationProfile, FeatureSample

SCHEMA_VERSION = 'facial_features_v1'
BASELINE_FIELDS = ('ear_left_baseline', 'ear_right_baseline', 'mar_baseline',
                   'pitch_baseline', 'yaw_baseline', 'roll_baseline')


def _usable(s):
    return (s.face_detected and s.left_eye_valid and s.right_eye_valid and
            s.mouth_valid and s.pose_valid and all(math.isfinite(v) for v in
            (s.ear_left,s.ear_right,s.mar,s.pitch,s.yaw,s.roll)))


def _candidates(samples, field, epsilon):
    values = sorted(getattr(s,field) for s in samples
                    if _usable(s) and getattr(s,field) > epsilon)
    return np.asarray(values[len(values)//2:], dtype=float)


def fit_qc_policy(samples: list[FeatureSample], *, epsilon: float = 1e-6) -> dict:
    """Fit per-source top-half positive candidates, exclusively supplied train Alert rows."""
    groups = defaultdict(list)
    for s in samples:
        groups[s.source_id].append(s)
    result = dict(version='train_alert_qc_v1', epsilon=epsilon, max_mad_ratio=.15,
                  quantile_method='linear', quantiles=[.01,.99],
                  source_ids=sorted(groups), sample_count=len(samples),
                  valid_sample_count=sum(_usable(s) for s in samples),
                  max_sample_age_ms=100, min_valid_duration_ms=20000,
                  initial_prefix_ms=30000, maximum_prefix_ms=60000,
                  max_pitch_delta_deg=25., max_yaw_delta_deg=35.)
    reasons = []
    for field in ('ear_left','ear_right'):
        candidates = [_candidates(rows,field,epsilon) for rows in groups.values()]
        values = np.concatenate(candidates) if candidates else np.asarray([])
        result[field+'_candidate_count'] = int(values.size)
        result[field+'_bounds'] = (np.quantile(values,[.01,.99],method='linear').tolist()
                                  if values.size else [None,None])
        if not values.size:
            reasons.append('no_'+field+'_candidates')
    result.update(valid=not reasons, reasons=reasons)
    return result


def _duration(rows, end, cap):
    return sum(min(b.timestamp_ms-a.timestamp_ms,cap,end-a.timestamp_ms)
               for a,b in zip(rows,rows[1:])
               if a.source_id == b.source_id and _usable(a) and _usable(b)
               and 0 <= a.timestamp_ms < end and b.timestamp_ms > a.timestamp_ms)


def _mouth_opening(rows, baseline, end, cap):
    active = False
    duration = 0
    for a,b in zip(rows,rows[1:]):
        if (not _usable(a) or not _usable(b) or a.source_id != b.source_id
                or b.timestamp_ms-a.timestamp_ms > cap):
            active=False; duration=0
            continue
        if a.timestamp_ms >= end:
            break
        delta=a.mar-baseline
        if not active and delta >= .35:
            active=True; duration=0
        elif active and delta <= .25:
            active=False; duration=0
        if active:
            duration += min(b.timestamp_ms-a.timestamp_ms,end-a.timestamp_ms)
            if duration >= 2000:
                return True
    return False


def estimate_profile(samples: list[FeatureSample], mode: str, *,
                     image_size: tuple[int,int], asset_sha256: str,
                     qc_policy: dict) -> CalibrationProfile:
    if mode not in ('P0','P1'):
        raise ValueError('Calibration mode must be P0 or P1')
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError('Calibration requires explicit positive source image size')
    rows=sorted(samples,key=lambda s:(s.source_id,s.timestamp_ms))
    if mode == 'P1' and len({s.source_id for s in rows}) > 1:
        raise ValueError('P1 estimation requires one Alert source')
    epsilon=qc_policy.get('epsilon',1e-6)
    ends=(range(qc_policy.get('initial_prefix_ms',30000),
                qc_policy.get('maximum_prefix_ms',60000)+1,1000) if mode=='P1' else [None])
    for end in ends:
        selected=[s for s in rows if end is None or 0<=s.timestamp_ms<end]
        usable=[s for s in selected if _usable(s)]
        reasons=list(qc_policy.get('reasons',[])) if not qc_policy.get('valid',False) else []
        stats=dict(prefix_start_ms=0,prefix_end_ms=end,source_ids=sorted({s.source_id for s in rows}),
                   sample_count=len(selected),valid_sample_count=len(usable))
        duration=_duration(selected,end if end is not None else float('inf'),qc_policy.get('max_sample_age_ms',100))
        stats['valid_duration_ms']=duration
        if mode=='P1' and duration<qc_policy.get('min_valid_duration_ms',20000):
            reasons.append('insufficient_valid_duration')
        baselines=[]
        for field in ('ear_left','ear_right'):
            if mode == 'P0':
                groups=defaultdict(list)
                for s in usable:
                    groups[s.source_id].append(s)
                parts=[_candidates(group,field,epsilon) for group in groups.values()]
                candidates=np.concatenate(parts) if parts else np.asarray([])
            else:
                candidates=_candidates(usable,field,epsilon)
            value=float(np.median(candidates)) if candidates.size else float('nan')
            ratio=float(np.median(np.abs(candidates-value))/value) if candidates.size else float('nan')
            stats[field+'_candidate_count']=int(candidates.size)
            stats[field+'_mad_ratio']=ratio
            bounds=qc_policy.get(field+'_bounds',[None,None])
            if not math.isfinite(value) or value<=epsilon:
                reasons.append('invalid_'+field+'_baseline')
            elif bounds[0] is None or not max(epsilon,bounds[0])<=value<=bounds[1]:
                reasons.append(field+'_outside_train_qc')
            if not math.isfinite(ratio) or ratio>qc_policy.get('max_mad_ratio',.15):
                reasons.append('unstable_'+field)
            baselines.append(value)
        for field in ('mar','pitch','yaw','roll'):
            baselines.append(float(np.median([getattr(s,field) for s in usable])) if usable else float('nan'))
        if not usable:
            reasons.append('no_valid_samples')
        if usable and _mouth_opening(selected,baselines[2],end if end is not None else float('inf'),qc_policy.get('max_sample_age_ms',100)):
            reasons.append('persistent_mouth_opening')
        stats.update(reasons=sorted(set(reasons)), max_pitch_delta_deg=qc_policy.get('max_pitch_delta_deg',25.),
                     max_yaw_delta_deg=qc_policy.get('max_yaw_delta_deg',35.))
        result=CalibrationProfile(mode,not reasons,*baselines,SCHEMA_VERSION,asset_sha256,tuple(image_size),stats)
        if result.valid:
            return result
    return result


def transform_sample(sample: FeatureSample, profile: CalibrationProfile) -> dict[str,float|bool]:
    """P0 valid transforms deliberately retain calibration_valid=False."""
    valid=profile.valid and profile.schema_version==SCHEMA_VERSION
    face=sample.face_detected and valid
    left=(face and sample.left_eye_valid and math.isfinite(sample.ear_left)
          and math.isfinite(profile.ear_left_baseline) and profile.ear_left_baseline>0)
    right=(face and sample.right_eye_valid and math.isfinite(sample.ear_right)
           and math.isfinite(profile.ear_right_baseline) and profile.ear_right_baseline>0)
    mouth=face and sample.mouth_valid and math.isfinite(sample.mar) and math.isfinite(profile.mar_baseline)
    deltas=[(v-b+180)%360-180 for v,b in zip(
        (sample.pitch,sample.yaw,sample.roll),(profile.pitch_baseline,profile.yaw_baseline,profile.roll_baseline))]
    pose=(face and sample.pose_valid and all(math.isfinite(v) for v in deltas)
          and abs(deltas[0])<=profile.quality_stats.get('max_pitch_delta_deg',25.)
          and abs(deltas[1])<=profile.quality_stats.get('max_yaw_delta_deg',35.))
    nan=float('nan')
    return dict(ear_left_norm=sample.ear_left/profile.ear_left_baseline if left else nan,
                ear_right_norm=sample.ear_right/profile.ear_right_baseline if right else nan,
                mar_delta=sample.mar-profile.mar_baseline if mouth else nan,
                pitch_delta=deltas[0] if pose else nan,yaw_delta=deltas[1] if pose else nan,
                roll_delta=deltas[2] if pose else nan,left_eye_valid=bool(left),right_eye_valid=bool(right),
                mouth_valid=bool(mouth),pose_valid=bool(pose),calibration_valid=bool(valid and profile.mode=='P1'))
