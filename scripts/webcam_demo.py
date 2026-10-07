"""Temporary native rule demo; diagnostic only, no driving, audio or smoothing.

Replay uses the exact frozen raw producer and its source timestamps. Camera P1
requires safe Alert confirmation and a bounded, train-QC checked personal prefix.
No images are written unless --save-overlay and the applicable permission allow it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2

from scripts.preview_landmarks import draw_overlay
from src.calibration.profile import SCHEMA_VERSION, estimate_profile, transform_sample
from src.config import load_config
from src.contracts import CalibrationProfile, EyeFeatures, MouthFeatures, PoseFeatures
from src.datasets.acquisition import digest_file
from src.datasets.splits import hash_payload, profile_from_dict, strict_json_value
from src.features.pipeline import FeaturePipeline
from src.features.temporal import (FEATURE_NAMES, TemporalFeatureExtractor,
                                   resolve_temporal_config, temporal_config_hash)
from src.models.rules import RuleBasedClassifier
from src.preprocessing.builder import FeatureDatasetBuilder
from src.preprocessing.snapshot import extraction_program_hashes
from src.preprocessing.video_reader import VideoReader

WINDOW_TITLE = 'Rule demo - NOT for driving (q/Esc: stop)'
SAFETY_NOTICE = ('Diagnostic experiment only, NOT a driving-safety device. Stay seated safely; '
                 'never run this while driving. P1 confirmation means you are currently Alert. '
                 'Failed calibration stays UNRELIABLE: restart explicitly to retry; no P0 fallback.')


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    with path.open(encoding='utf-8') as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON mapping: {path}')
    return value


def _encode_mapping(value):
    if isinstance(value, Mapping):
        return strict_json_value(dict(value))
    raise TypeError(f'Unsupported receipt value: {type(value).__name__}')


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(strict_json_value(value), handle, indent=2, allow_nan=False, default=_encode_mapping)
        handle.write('\n')


def append_json(handle, value: dict) -> None:
    handle.write(json.dumps(strict_json_value(value), allow_nan=False, separators=(',', ':'), default=_encode_mapping) + '\n')


def bind_profile(profile: CalibrationProfile, provenance: dict) -> CalibrationProfile:
    """P1 keeps exact binding; P0 contributes only train-population numerics."""
    if (profile.schema_version != SCHEMA_VERSION or provenance['schema_version'] != SCHEMA_VERSION
            or profile.asset_sha256 != provenance['artifact_hashes']['asset_sha256']):
        raise ValueError('Profile schema/asset provenance mismatch')
    if profile.mode == 'P1':
        if (list(profile.image_size) != provenance['image_size']
                or strict_json_value(profile.quality_stats.get('provenance')) != strict_json_value(provenance)):
            raise ValueError('Personal profile camera/assets/resolution provenance mismatch')
        return profile
    if profile.mode != 'P0':
        raise ValueError('Unknown profile mode')
    return replace(profile, image_size=tuple(provenance['image_size']),
                   quality_stats={**profile.quality_stats, 'population_numeric_only': True,
                                  'application_provenance': provenance})


def replay_prefix_end(profile: CalibrationProfile, video_id: str) -> int:
    if profile.mode != 'P1' or profile.quality_stats.get('alert_video_id') != video_id:
        return 0
    end = profile.quality_stats.get('prefix_end_ms')
    if not isinstance(end, int) or not 30000 <= end <= 60000:
        raise ValueError('Personal replay requires the entire bounded used Alert prefix')
    return end


def attempt_live_profile(samples: list, elapsed_ms: int, *, image_size: tuple[int, int],
                         asset_sha256: str, qc_policy: dict) -> CalibrationProfile | None:
    """Only completed source-relative seconds may be considered by the estimator."""
    initial = qc_policy.get('initial_prefix_ms', 30000)
    maximum = qc_policy.get('maximum_prefix_ms', 60000)
    if initial != 30000 or maximum != 60000 or qc_policy.get('min_valid_duration_ms', 20000) != 20000:
        raise ValueError('Demo P1 requires the approved 30–60s / 20 valid seconds policy')
    completed = min(maximum, (elapsed_ms // 1000) * 1000)
    if completed < initial:
        return None
    observed_policy = {**qc_policy, 'maximum_prefix_ms': completed}
    return estimate_profile(samples, 'P1', image_size=image_size,
                            asset_sha256=asset_sha256, qc_policy=observed_policy)


def load_inputs(args) -> tuple[dict, dict, dict, dict | None, Path | None, dict]:
    directory = args.snapshot_run / 'snapshot'
    snapshot = read_json(directory / 'snapshot.json')
    if hash_payload({k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}) != snapshot['snapshot_sha256']:
        raise ValueError('Frozen snapshot hash mismatch')
    config_path = directory / 'preprocessing.yaml'
    if digest_file(config_path)[0] != snapshot['config_file_sha256']:
        raise ValueError('Frozen preprocessing config file drift')
    config = load_config(config_path)
    signature = FeatureDatasetBuilder(config, constant_fps_verified=snapshot['signature']['constant_fps_verified'])._signature()
    if signature != snapshot['signature']:
        raise ValueError('Frozen raw producer config/assets/dependencies drift')
    if extraction_program_hashes() != snapshot['extraction_program_sha256']:
        raise ValueError('Frozen raw producer program drift')
    split = read_json(args.profiles)
    if hash_payload({k: v for k, v in split.items() if k != 'split_hash'}) != split.get('split_hash'):
        raise ValueError('Split/profile map hash mismatch')
    if split.get('snapshot_sha256') != snapshot['snapshot_sha256']:
        raise ValueError('Split/profile map belongs to another snapshot')
    qc = split['qc_policy']
    if hash_payload({k: v for k, v in qc.items() if k != 'policy_hash'}) != qc.get('policy_hash'):
        raise ValueError('Train QC policy hash mismatch')
    if qc.get('train_subject_ids') != split['roles']['train']['subject_ids']:
        raise ValueError('QC policy does not match the split training subjects')
    with args.temporal_config.open(encoding='utf-8') as handle:
        import yaml
        temporal = resolve_temporal_config(yaml.safe_load(handle))
    row = metadata = path = None
    if args.source == 'video':
        matches = [r for r in snapshot['sources'] if r['video_id'] == args.video_id]
        if len(matches) != 1:
            raise ValueError('Video ID is not unique in the frozen snapshot')
        row = matches[0]
        split_rows = [r for r in split['sources'] if r['video_id'] == args.video_id]
        if len(split_rows) != 1 or split_rows[0] != row:
            raise ValueError('Video subject/source identity differs from the frozen split')
        root = Path(snapshot['source_root']).resolve()
        path = (root / row['relative_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Video source escapes frozen source root')
        if digest_file(path)[0] != row['sha256'] or path.stat().st_size != row['size_bytes']:
            raise ValueError('Video source hash/size differs from freeze')
        metadata = read_json(Path(snapshot['output_dir']) / f'{args.video_id}.metadata.json')
        if (metadata['source_sha256'] != row['sha256'] or metadata['manifest_row'] != row
                or metadata['extraction_fingerprint'] != signature['extraction_fingerprint']
                or not metadata['complete_source_validation']):
            raise ValueError('Raw metadata is not a complete matching frozen video')
    return snapshot, split, config, row, path, {'temporal': temporal, 'metadata': metadata}


def select_profile(split: dict, mode: str, row: dict | None) -> CalibrationProfile | None:
    if row is None and mode == 'P1':
        return None  # Camera personal calibration must come from this session, not a recorded person.
    subject = row['subject_id'] if row else split['roles']['train']['subject_ids'][0]
    profile = profile_from_dict(split['profiles'][mode][subject])
    if profile.mode != mode:
        raise ValueError('Profile mode mismatch')
    if profile.quality_stats.get('qc_policy_hash') != split['qc_policy']['policy_hash']:
        raise ValueError('Profile does not bind the provided train QC policy')
    if mode == 'P0':
        binding = profile.quality_stats.get('binding_source_id')
        if binding not in split['qc_policy']['train_alert_video_ids']:
            raise ValueError('Population profile is not bound to train Alert data')
    else:
        alert = profile.quality_stats.get('alert_video_id')
        sources = [r for r in split['sources'] if r['video_id'] == alert]
        if len(sources) != 1 or sources[0]['subject_id'] != subject or sources[0]['label_id'] != 0:
            raise ValueError('Personal profile Alert source belongs to another subject/label')
    return profile


def render_overlay(packet, sample, pipeline, calibration: str, latest, predictions: dict,
                   profile: CalibrationProfile | None):
    quality = pipeline.last_quality
    overlay = draw_overlay(packet.image_bgr, pipeline.last_landmarks,
                           eyes=EyeFeatures(sample.ear_left, sample.ear_right, sample.ear_mean,
                                            sample.left_eye_valid, sample.right_eye_valid),
                           mouth=MouthFeatures(sample.mar, sample.mouth_valid),
                           pose=PoseFeatures(sample.pitch, sample.yaw, sample.roll, sample.pose_valid,
                                             sample.reprojection_error_norm),
                           pose_approximate=pipeline.pose_estimator.camera_metadata(pipeline.last_landmarks.image_size)['approximate'],
                           quality_reasons=quality.reasons)
    rows = [f'P1/P0 calibration: {calibration} | NOT FOR DRIVING']
    normalized = transform_sample(sample, profile) if profile is not None else {}
    current_usable = (all(normalized.get(name, False) for name in
                          ('left_eye_valid', 'right_eye_valid', 'mouth_valid', 'pose_valid'))
                      and calibration in ('P0_POPULATION_NOT_PERSONAL', 'P1_FROZEN'))
    for mode in ('ear_only', 'temporal'):
        pred = predictions.get(mode)
        state = (pred.class_id.name if pred is not None and pred.valid and current_usable
                 else 'NO_FACE' if not sample.face_detected else 'UNRELIABLE/WARMUP')
        reason = pred.reason.split(";")[0] if pred is not None and current_usable else 'current profile/channel invalid'
        rows.append(f'{mode}: {state} | {reason}')
    if latest is not None:
        summary = latest.event_summaries
        rows.append(f'history {summary["history_s"]:.1f}s | coverage {summary["coverage"]:.1%} | ready {summary["perclos_ready"]}')
        rows.append(f'closure {summary["closure_elapsed_s"]:.2f}s | mouth candidate {summary["yawn_elapsed_s"]:.2f}s')
        rows.append(f'grid timestamp {latest.timestamp_ms} ms (causal, not future)')
    scale = max(1., max(packet.image_bgr.shape[:2]) / 640)
    bottom = overlay.shape[0] - round(12 * scale)
    for index, text in enumerate(reversed(rows)):
        cv2.putText(overlay, text, (round(8 * scale), bottom - round(index * 22 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4 * scale, (0, 255, 255),
                    max(1, round(scale)), cv2.LINE_AA)
    return overlay


def run_demo(args: argparse.Namespace) -> dict:
    """Run one producer session, recording strict-JSON receipts and resource closure."""
    args.log_dir.mkdir(parents=True, exist_ok=True)
    outputs = ['summary.json', 'config.json', 'profile.json', 'raw_samples.jsonl', 'observations.jsonl']
    if args.save_overlay:
        outputs.append('overlay.png')
    if any((args.log_dir / name).exists() for name in outputs):
        raise ValueError('Log directory already contains demo artifacts; choose a new directory')
    report = dict(schema_version='native_rule_demo_v1', source=args.source,
                  video_id=args.video_id if args.source == 'video' else None,
                  camera_index=args.camera_index if args.source == 'webcam' else None, source_id=None,
                  mode=args.mode, status='RUNNING', stop_reason=None, error=None,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  safety_notice=SAFETY_NOTICE, physical_acceptance='NOT_RUN',
                  mirrored=False, complete_source_validation=False, saved_overlays=[],
                  scheduled_samples=0, prefix_excluded_samples=0, temporal_samples=0,
                  first_timestamp_ms=None, last_timestamp_ms=None, first_frame_index=None,
                  last_frame_index=None, calibration_state='NOT_STARTED', profile_hash=None,
                  closure_errors=[])
    reader = VideoReader()
    pipeline = frames = raw_handle = observation_handle = None
    profile = latest = extractor = None
    predictions = {}
    rule_counts = {m: Counter() for m in ('ear_only', 'temporal')}
    abstentions = {m: Counter() for m in rule_counts}
    events = Counter()
    native_abstentions = Counter()
    window_created = False
    started = time.perf_counter()
    try:
        print(SAFETY_NOTICE, flush=True)
        if args.source == 'webcam' and args.mode == 'P1' and not args.confirm_alert:
            raise ValueError('Camera P1 requires --confirm-alert after reading the safe seated Alert notice')
        if args.source == 'webcam' and args.mode == 'P1':
            print('Confirmed safe seated Alert collection: remain naturally Alert for 30–60 seconds. '
                  'Stop with q/Esc; retry only by restarting this command.', flush=True)
        snapshot, split, config, row, source, inputs = load_inputs(args)
        if args.save_overlay and ((row is not None and row.get('image_publishable') is not True)
                                  or (row is None and not args.consent_camera_images)):
            raise ValueError('Overlay saving requires a publishable video or explicit --consent-camera-images')
        temporal_config = inputs['temporal']
        rules = {m: RuleBasedClassifier(temporal_config, mode=m) for m in rule_counts}
        report.update(snapshot_sha256=snapshot['snapshot_sha256'], split_hash=split['split_hash'],
                      temporal_hash=temporal_config_hash(temporal_config),
                      qc_policy_hash=split['qc_policy']['policy_hash'],
                      raw_extraction_fingerprint=snapshot['signature']['extraction_fingerprint'])
        write_json(args.log_dir / 'config.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   preprocessing=snapshot['signature']['effective_config'], temporal=temporal_config,
                   feature_names=FEATURE_NAMES, identities={k: report[k] for k in
                       ('snapshot_sha256', 'split_hash', 'temporal_hash', 'qc_policy_hash', 'raw_extraction_fingerprint')}))
        profile = select_profile(split, args.mode, row)
        prefix_end = replay_prefix_end(profile, args.video_id) if profile is not None and row is not None else 0
        report['excluded_prefix_end_ms'] = prefix_end
        pipeline = FeaturePipeline(config)
        frames = (reader.iter_frames(source, config['landmark_target_fps'],
                                     constant_fps_verified=snapshot['signature']['constant_fps_verified'])
                  if row else reader.iter_camera(args.camera_index, config['landmark_target_fps'],
                                                  width=args.camera_width, height=args.camera_height))
        raw_handle = (args.log_dir / 'raw_samples.jsonl').open('x', encoding='utf-8')
        observation_handle = (args.log_dir / 'observations.jsonl').open('x', encoding='utf-8')
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            window_created = True
        source_provenance = None
        collection = []
        attempt_second = -1
        collection_done = False
        first_wall = None
        overlay_saved = False
        for packet in frames:
            now = time.perf_counter()
            if report['first_timestamp_ms'] is None:
                report['first_timestamp_ms'] = packet.timestamp_ms
                report['first_frame_index'] = packet.frame_index
                first_wall = now
            elapsed_ms = packet.timestamp_ms - report['first_timestamp_ms']
            if args.seconds is not None and ((now - first_wall if row is None else elapsed_ms / 1000) > args.seconds):
                report.update(status='STOPPED', stop_reason='seconds_bound')
                report['bound_unprocessed_packet'] = dict(timestamp_ms=packet.timestamp_ms,
                                                          frame_index=packet.frame_index)
                break
            sample = pipeline.process(packet)  # Exactly once; geometry and both rules reuse this result.
            if report['source_id'] is None:
                report['source_id'] = sample.source_id
            report['scheduled_samples'] += 1
            report['last_timestamp_ms'] = sample.timestamp_ms
            report['last_frame_index'] = sample.frame_index
            size = tuple(pipeline.last_landmarks.image_size)
            provenance = dict(schema_version=SCHEMA_VERSION, image_size=list(size),
                              artifact_hashes=snapshot['signature']['artifact_hashes'],
                              camera_metadata=pipeline.pose_estimator.camera_metadata(size))
            if source_provenance is None:
                source_provenance = provenance
                report['source_provenance'] = provenance
                if row and (list(size) != inputs['metadata']['image_size']
                            or provenance['camera_metadata'] != inputs['metadata']['camera_metadata']):
                    raise ValueError('Actual decoded resolution/camera differs from frozen raw metadata')
                if profile is not None:
                    profile = bind_profile(profile, provenance)
                    extractor = TemporalFeatureExtractor(profile, temporal_config)
                    extractor.reset()  # Prefix is never passed through the temporal engine.
                    report['profile_hash'] = hash_payload(asdict(profile))
                    write_json(args.log_dir / 'profile.json', dict(profile=asdict(profile), profile_hash=report['profile_hash'],
                               origin='offline_personal' if args.mode == 'P1' else 'train_population_numeric_only'))
                if not args.headless:
                    width, height = size
                    scale = min(640 / width, 640 / height)
                    cv2.resizeWindow(WINDOW_TITLE, round(width * scale), round(height * scale))
            elif source_provenance != provenance:
                raise ValueError('Source camera/resolution/assets changed after session freeze; restart required')
            excluded = bool(row and sample.timestamp_ms < prefix_end)
            if excluded:
                report['prefix_excluded_samples'] += 1
            if row is None and args.mode == 'P1' and not collection_done:
                if elapsed_ms <= 60000:
                    collection.append(replace(sample, timestamp_ms=elapsed_ms))
                second = elapsed_ms // 1000
                if second >= 30 and second != attempt_second:
                    attempt_second = second
                    candidate = attempt_live_profile(collection, elapsed_ms, image_size=size,
                                                     asset_sha256=provenance['artifact_hashes']['asset_sha256'],
                                                     qc_policy=split['qc_policy'])
                    profile = candidate
                    append_json(observation_handle, dict(kind='calibration_attempt', timestamp_ms=sample.timestamp_ms,
                                elapsed_ms=elapsed_ms, profile=asdict(candidate), profile_hash=hash_payload(asdict(candidate))))
                    if candidate.valid or elapsed_ms >= 60000:
                        collection_done = True
                        profile = replace(candidate, quality_stats={**candidate.quality_stats,
                                          'provenance': provenance, 'qc_policy_hash': split['qc_policy']['policy_hash'],
                                          'session_source_id': sample.source_id,
                                          'session_prefix_origin_ms': report['first_timestamp_ms'],
                                          'confirmed_alert': True, 'frozen_for_session': True})
                        report['profile_hash'] = hash_payload(asdict(profile))
                        write_json(args.log_dir / 'profile.json', dict(profile=asdict(profile), profile_hash=report['profile_hash'], origin='live_confirmed_alert'))
                        if profile.valid:
                            extractor = TemporalFeatureExtractor(profile, temporal_config)
                            extractor.reset()
                        else:
                            print('UNRELIABLE: calibration failed. Stop and restart explicitly to retry.', flush=True)
                        collection.clear()
                excluded = not collection_done or (profile is not None and elapsed_ms < profile.quality_stats['prefix_end_ms'])
                if excluded:
                    report['prefix_excluded_samples'] += 1
            if row is None and args.mode == 'P1' and not collection_done:
                calibration = 'CALIBRATING'
            elif profile is None or not profile.valid:
                calibration = 'UNRELIABLE_RESTART_TO_RETRY'
            elif excluded:
                calibration = 'OFFLINE_USED_PREFIX_EXCLUDED'
            else:
                calibration = 'P0_POPULATION_NOT_PERSONAL' if args.mode == 'P0' else 'P1_FROZEN'
            report['calibration_state'] = calibration
            append_json(raw_handle, dict(kind='raw_sample', sample=asdict(sample),
                        quality_reasons=pipeline.last_quality.reasons,
                        quality_metrics=pipeline.last_quality.metrics, calibration_state=calibration,
                        prefix_excluded=excluded, profile_hash=report['profile_hash']))
            if extractor is not None and not excluded:
                for temporal_sample in extractor.update(sample):
                    latest = temporal_sample
                    report['temporal_samples'] += 1
                    predictions = {mode: rule.predict(latest) for mode, rule in rules.items()}
                    for mode, prediction in predictions.items():
                        rule_counts[mode][prediction.class_id.name if prediction.valid else 'ABSTAIN'] += 1
                        if not prediction.valid:
                            abstentions[mode][prediction.reason] += 1
                    for event in latest.event_summaries['events']:
                        classification = event['classification'] or ('censored' if event['censored'] else 'noncandidate')
                        events[event['kind'] + ':' + classification] += 1
                    append_json(observation_handle, dict(kind='temporal_rules', native_frame_index=sample.frame_index,
                                native_timestamp_ms=sample.timestamp_ms, temporal=asdict(latest),
                                predictions={m: asdict(p) for m, p in predictions.items()}, profile_hash=report['profile_hash']))
            else:
                native_abstentions[calibration] += 1
                append_json(observation_handle, dict(kind='native_abstention', timestamp_ms=sample.timestamp_ms,
                            frame_index=sample.frame_index, reason=calibration, rule_modes=list(rules)))
            if not args.headless or (args.save_overlay and not overlay_saved):
                overlay = render_overlay(packet, sample, pipeline, calibration, latest, predictions, profile)
                if args.save_overlay and not overlay_saved and sample.face_detected:
                    destination = args.log_dir / 'overlay.png'
                    if destination.exists() or not cv2.imwrite(str(destination), overlay):
                        raise RuntimeError(f'Cannot safely save overlay: {destination}')
                    report['saved_overlays'].append(str(destination))
                    overlay_saved = True
                if not args.headless:
                    cv2.imshow(WINDOW_TITLE, overlay)
                    if cv2.waitKey(1) & 0xff in (27, ord('q')) or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        report.update(status='STOPPED', stop_reason='user_window_stop')
                        break
            if args.seconds is not None and ((time.perf_counter() - first_wall if row is None else elapsed_ms / 1000) >= args.seconds):
                report.update(status='STOPPED', stop_reason='seconds_bound')
                break
        else:
            report.update(status='EOF', stop_reason='source_eof', complete_source_validation=args.source == 'video')
    except KeyboardInterrupt:
        report.update(status='STOPPED', stop_reason='keyboard_interrupt')
    except Exception as exc:
        report.update(status='ERROR', error=f'{type(exc).__name__}: {exc}', complete_source_validation=False)
    finally:
        for name, resource, close in (('iterator', frames, lambda r: r.close()),
                                      ('reader', reader, lambda r: r.close()),
                                      ('pipeline', pipeline, lambda r: r.close()),
                                      ('raw_log', raw_handle, lambda r: r.close()),
                                      ('observation_log', observation_handle, lambda r: r.close())):
            if resource is not None:
                try:
                    close(resource)
                except Exception as exc:
                    report['closure_errors'].append(f'{name}: {type(exc).__name__}: {exc}')
        if window_created:
            try:
                cv2.destroyWindow(WINDOW_TITLE)
            except Exception as exc:
                report['closure_errors'].append(f'window: {type(exc).__name__}: {exc}')
        if report['closure_errors']:
            report.update(status='ERROR', complete_source_validation=False)
        report['reader'] = reader.stats
        report['pipeline'] = pipeline.stats if pipeline is not None else None
        report['window_closed'] = window_created and not any(e.startswith('window:') for e in report['closure_errors'])
        report['rule_counts'] = {m: dict(c) for m, c in rule_counts.items()}
        report['abstentions'] = {m: dict(c) for m, c in abstentions.items()}
        report['event_counts'] = dict(events)
        report['native_abstentions'] = dict(native_abstentions)
        report['last_temporal_timestamp_ms'] = latest.timestamp_ms if latest is not None else None
        report['last_event_state'] = latest.event_summaries if latest is not None else None
        report['wall_seconds'] = time.perf_counter() - started
        if profile is not None and not (args.log_dir / 'profile.json').exists():
            report['profile_hash'] = hash_payload(asdict(profile))
            write_json(args.log_dir / 'profile.json', dict(profile=asdict(profile), profile_hash=report['profile_hash'], origin='incomplete_live_collection_not_applied'))
        if report['calibration_state'] == 'CALIBRATING':
            report['calibration_state'] = 'UNRELIABLE_INCOMPLETE_COLLECTION_RESTART_TO_RETRY'
        write_json(args.log_dir / 'summary.json', report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, epilog=SAFETY_NOTICE)
    result.add_argument('--source', choices=['video', 'webcam'], required=True)
    result.add_argument('--video-id', default='04_0')
    result.add_argument('--snapshot-run', type=Path, default=Path('runs/phase8'))
    result.add_argument('--profiles', type=Path, required=True, help='data/splits/outer_i.json with train QC and P0/P1 profiles')
    result.add_argument('--mode', choices=['P0', 'P1'], required=True)
    result.add_argument('--temporal-config', type=Path, default=Path('configs/temporal.yaml'))
    result.add_argument('--log-dir', type=Path, required=True)
    result.add_argument('--seconds', type=float, help='Optional source-relative video / wall-time camera bound')
    result.add_argument('--headless', action='store_true')
    result.add_argument('--camera-index', type=int, default=0)
    result.add_argument('--camera-width', type=int, default=640)
    result.add_argument('--camera-height', type=int, default=480)
    result.add_argument('--confirm-alert', action='store_true', help='Confirm safe seated, currently Alert P1 camera collection; never driving')
    result.add_argument('--save-overlay', action='store_true', help='Opt in to one permitted diagnostic overlay, never default imagery')
    result.add_argument('--consent-camera-images', action='store_true', help='Explicit consent for camera overlay storage')
    return result


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        cli.error('--seconds must be finite and positive')
    if args.camera_index < 0 or min(args.camera_width, args.camera_height) <= 0:
        cli.error('Camera index must be nonnegative and dimensions positive')
    try:
        report = run_demo(args)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', flush=True)
        return 1
    print(json.dumps(strict_json_value(report), indent=2, allow_nan=False), flush=True)
    return 1 if report['status'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
