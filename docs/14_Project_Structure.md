# 14 — Cấu trúc project và config

Cây dưới mô tả đích cuối cùng. Phase0–13 đã có CV/raw snapshot/audit, pure calibration, causal temporal/rules, shared derived/index/scaler, RF và LSTM trainer/bundle. P0outer0 đã train/cold-load; P1 cả5slot data-blocked, không có checkpoint. Realtime manager/buffer/smoother/alerts/Qt UI và aggregate evaluation còn là thiết kế. Shared raw schema/config giữ nguyên;9zero-feature-coverage videos không bị xóa/nới quality. Xem roadmap/README cho metrics có điều kiện và hardware acceptance pending.

```text
project/
  README.md                # Lệnh chạy và trạng thái implementation
  docs/                    # README thiết kế, 01–20, Dataset_Access.md, CHANGELOG.md
  data/raw/                 # Không commit
  data/processed/           # Parquet per video, không commit
  data/acquisition/        # Inventory, subset plan, permission map, download report
  data/splits/              # subject IDs, protocol, hash
  notebooks/               # Exploration, không chứa logic production
  src/
    contracts.py           # Shared data records + enums
    config.py              # Safe YAML, validation, paths project-relative
    preprocessing/video_reader.py, builder.py
    features/landmarks.py, eye.py, mouth.py, head_pose.py, pipeline.py, temporal.py
    datasets/acquisition.py, manifest.py, splits.py, derived.py, normalization.py, sequence.py, summaries.py
    models/baseline.py, lstm.py, bundle.py
    training/trainer.py, experiment.py
    evaluation/evaluator.py, replay.py
    calibration/profile.py, manager.py
    realtime/buffer.py, smoother.py, detector.py, camera_worker.py
    alerts/manager.py
    ui/main_window.py
  scripts/
    check_environment.py, acquire_manifest.py, explore_dataset.py
    preview_landmarks.py, preprocess.py, process_snapshot.py, audit_features.py, build_splits.py, build_derived.py
    train_baseline.py, train_lstm.py, evaluate.py, run_ablation.py
    webcam_demo.py, benchmark.py
  tests/                   # Synthetic deterministic fixtures
  models/assets/           # .task, canonical OBJ + URL/hash/license
  models/checkpoints/      # Không commit weights lớn
  configs/preprocessing.yaml, temporal.yaml, training.yaml, realtime.yaml
  runs/                    # Config/metrics/log từng experiment
  requirements.txt, requirements-dev.txt, requirements-lock.txt
  .gitignore, main.py      # main.py thuộc phase UI sau, chưa có
```

## Storage schema
Parquet per video, dtype `float32` cho feature, bool cho validity, int64 timestamp, int32 frame_index, int8 label_id. NaN/null là missing; timestamp luôn có. Dùng PyArrow engine; không CSV làm source of truth vì dtype/null và kích thước. CSV chỉ export vài hàng cho team xem.

Raw row: `dataset_name, subject_id, video_id, frame_index, timestamp_ms, ear_left, ear_right, ear_mean, mar, pitch, yaw, roll, face_detected, left_eye_valid, right_eye_valid, mouth_valid, pose_valid, reprojection_error_norm, label_id, label_source`.

Metadata companion `video_id.metadata.json`: schema/producer version, manifest row/membership provenance, source SHA/size, asset/canonical/frozen-quality hashes, effective config/camera K/distortion/approximate flag, dependencies/CFR policy, extraction fingerprint, actual reader/model/stage stats, rows/time/validity/reasons và Parquet SHA256. Nhãn chỉ join ở builder; `source_id` nằm trong sample/metadata, không thêm cột storage trùng video_id. Unknown measurements thành Arrow null, không stringNaN hoặc false0. Future YawDD unknown label_id vẫn nullable, chưa có adapter hoặc download.

### Publication và resume — Phase7
Một writer sequential, tối đa1024 scalar rows/batch. Chỉ publish sau reader EOF/released, writer/native close thành công và source SHA/size vẫn khớp. Stage unique cùng filesystem; sync Parquet bằng writable handle (`r+b` trên Windows), sync metadata, replace Parquet trước rồi replace metadata **cuối cùng như commit marker**. Hai replaces không phải filesystem transaction. Interruption giữa chúng làm pair thiếu/mismatched và không cache được.

Resume/consumers phải check cả pair, complete-source flag + EOF/release, signature/source/manifest/artifact/dependency/CFR agreement, actual Parquet SHA, exact Arrow schema và footer row_count. File existence/config hash riêng không đủ. Stale old bytes được giữ nếu recompute fail nhưng report `output_current=false`; không công nhận current success. Không nhiều builder/process ghi đồng thời cùng output directory. CLI bắt buộc explicit repeated video IDs hoặc `--all-working-snapshot`; Phase7 chỉ QC04_0/04_5/04_10, không sửa manifest/full-acquisition gate.

### Frozen working snapshot và audit — Phase8
`scripts/process_snapshot.py` freeze manifest/YAML byte-for-byte và resolved builder signature, original source rows/paths/hashes, extraction-program hashes, dependencies và acquisition missing IDs dưới `<run-dir>/snapshot/`. `snapshot_sha256` nhận diện cả snapshot; input/code/asset drift phải dùng run-dir mới, không silently rewrite freeze. Runner gọi builder sequential với một selected ID nhưng **full frozen manifest**, checkpoint report sau từng member; pending không là complete. Completed/cached results phải có frozen fingerprint và current commit pair.

`scripts/audit_features.py` reopen từng pair và stream toàn rows để kiểm exact schema, source/label identity, time/index ordering qua batch boundaries, first source frame0, finite/null/masks/mean/angle bounds, counts và release. Coverage theo video/subject/class là sum(valid rows)/sum(audited emitted rows), không mean(video ratios) hoặc duration-weighted coverage. Failed members được giữ với explicit reason/support; zero denominator là null. Các quality reasons chồng lấp.

`data/processed/extraction_status.parquet` là status manifest riêng, chứa frozen source fields/status cùng extraction/audit status và snapshot/source/fingerprint provenance. Giữ nguyên bytes của `manifest.parquet`: `status=ok/error` chứng minh acquisition/source verification, không chứng minh feature quality. Raw feature schema và metadata-last publication không đổi; sidecar không phải calibrated sequences hoặc train/validation/test split.

### Derived/model storage — Phase9–13
`data/splits/outer_0..4.json` giữ roles, QC, profiles, reserved ranges và hashes. `data/processed/derived_<hash>/` giữ per-video float32[16]/bool[16] temporal Parquet cùng event receipts, shared accepted index/coverage và final dataset manifest. Source schema_version giữ `facial_features_v1`; ordered features và derived identity phân biệt layout mới. Row range `[start_row,end_row)` chứa100ticks; protocol mode/outer nằm ở manifest; nominal10s duration khác9.9s tick span.

Scaler first10 channels fit valid unique accepted-train timesteps, không overlap-weighted; zero valid observations chặn fit, constant channel scale1; binary channels không scale. LSTM bundle là **directory** `weights.pt`, `metadata.json`, `scaler.json`, `integrity.json` trong mode-specific run, không alias `.pt` cho P1 thiếu dữ liệu. RF dùng trusted local joblib và summary schema. Checkpoints/profiles/cache/receipts giữ local, không commit; metadata/six-hash/mode mismatch bị reject.



## Config — một nơi cho mỗi giá trị
YAML đọc bằng `safe_load`; validate ngay startup: FPS dương, window*FPS nguyên, stride≤window, threshold range đúng, paths tồn tại; schema/checkpoint mismatch báo lỗi. Runtime paths resolve từ project root hoặc config root thống nhất, không phụ thuộc shell cwd.

Ví dụ đoạn config; bản chạy nằm trong `configs/`. Phase7 đã thêm `camera_model` và optical `quality` thực đo/freeze; accepted evidence là `configs/quality_policy_v1.json`, nested path/SHA được kiểm tra startup. Không universal blur defaults hoặc null placeholder. Current defaults approximate camera; calibrated block cần reference_size/matrix/distortion hữu hạn và đúng shape.
```yaml
# preprocessing.yaml
landmark_target_fps: 20
sequence_fps: 10
num_faces: 1
landmarker_mode: VIDEO
min_face_detection_confidence: 0.5
min_face_presence_confidence: 0.5
min_tracking_confidence: 0.5
asset_path: models/assets/face_landmarker.task
canonical_model_path: models/assets/canonical_face_model.obj
max_sample_age_ms: 100
statistics_window_s: 60
```

Parameter còn lại phải tập trung:
- **preprocessing:** frozen raw/output/manifest paths, camera K/distortion, blur/brightness/absolute pose gates, landmark IDs/EAR epsilon. Legacy event settings trong frozen YAML không đổi để giữ Phase8 provenance; temporal engine nhận duy nhất resolved policy từ `configs/temporal.yaml`.
- **temporal:** native eye/mouth event thresholds/durations, PERCLOS proxy/history/coverage, hold/reset timing và diagnostic rule gates; resolved policy hash được bind vào derived/model metadata.
- **training:** splits path/outer fold; calibration_mode P0/P1; window=10; stride=1; missing_ratio=0.20; max_gap_s=1; ordered features; seed=42; RF/LSTM parameters; device; batch/epochs/loss/optimizer/early stopping; experiment path.
- **realtime:** camera index/resolution=640×480; model path; checkpoint mode; prediction interval=1; stale_ms=500; smoothing samples=3; calibration_seconds=30, min_valid_seconds=20, timeout_seconds=60; warning enter/exit thresholds và durations ở [12](12_Realtime_Inference.md); audio path/cooldown=15; mute; UI refresh=10 Hz; log retention.

Không duplicate sequence FPS/window trong realtime YAML: realtime đọc từ model bundle; chỉ override khi tương thích và được kiểm tra, không tự đổi 100 bước thành 50. Resolved config snapshot lưu mỗi run.

## Git và logging
`main` luôn chạy được; feature branch ngắn `feature/ear`, `feature/head-pose`, `feature/lstm`, `feature/realtime`; PR nhỏ theo phase, merge sau tests/smoke. `develop` chỉ thêm nếu team thật sự cần tích hợp nhiều người, không bắt buộc GitFlow.

Commit docs/config/code/test; không raw data, video mặt, weights lớn, credential, calibration profile cá nhân. Log JSONL sự kiện/error và CSV/Parquet predictions; rotation theo size/session, không log ảnh mặc định. Notebook gọi src, không giữ một bản thuật toán riêng. Pin environment sau Phase 0, cập nhật changelog theo milestone.

## Lệnh Phase 0–2 hiện tại
Chạy `.venv/Scripts/python.exe -m scripts.check_environment`, `-m scripts.acquire_manifest`, `-m scripts.explore_dataset` từ project root theo [README](../README.md). Smoke đầy đủ cần `--sample-video`; `--assets-only` chỉ tải asset, không chứng minh environment. `manifest.parquet` có thêm frame_count/part_id/fold_source/status/error; mọi video lỗi vẫn được giữ để audit.
