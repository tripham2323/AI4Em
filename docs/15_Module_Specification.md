# 15 — Module contracts

Contract này là chuẩn cho coding agent. Shared records trong `src/contracts.py` giữ nguyên. Phase3–8 có CV/raw pipeline/snapshot/audit; Phase9–13 đã có pure profiles, causal temporal, shared windows, RF, sequence scaler và LSTM trainer/bundle. P0 đã train/reload thật; mọi slot P1 bị data gate chặn, không có P1 checkpoint. Realtime manager/buffer/smoother/alerts/Qt UI và Phase14 aggregate evaluation vẫn là thiết kế. Thời gian ms, duration giây, góc độ, velocity độ/giây; probability/coverage ở0–1, **EAR/MAR không có upper bound1**.

## Shared records — `src/contracts.py`
| Type | Trường tối thiểu |
|---|---|
| FramePacket | `image_bgr:uint8[H,W,3], timestamp_ms:int, frame_index:int, source_id:str` |
| LandmarkResult | `points:float32[N,3] hoặc null` tọa độ normalized, `timestamp_ms`, `face_detected:bool`, `image_size:(W,H)`; không bịa visibility/confidence |
| EyeFeatures | `ear_left/right/mean:float hoặc NaN`, `left_eye_valid/right_eye_valid:bool` |
| MouthFeatures | `mar:float hoặc NaN`, `mouth_valid:bool` |
| PoseFeatures | `pitch/yaw/roll:float hoặc NaN`, `pose_valid:bool`, `reprojection_error_norm:float` |
| FeatureSample | timestamp, eye/mouth/pose fields, `face_detected`, source/frame IDs; **không label trong realtime record** |
| CalibrationProfile | baselines từng mắt/MAR/pose, valid, schema/asset/resolution, quality stats; baseline population có mode P0 |
| TemporalSample | timestamp, 16 giá trị theo [07](07_Temporal_Modeling.md), validity trước impute, `segment_id`, event summaries |
| SequenceWindow | `x:float32[100,16]`, timestamps start/end, quality, feature_names/schema; training wrapper thêm label_id/subject/video metadata ngoài X |
| Prediction | `timestamp_ms` của cuối window, `probabilities:float32[3]`, class_id argmax, `valid`, reason, model_id |
| DetectionResult | raw/smoothed Prediction hoặc null, system_status, quality, calibration status |
| AlertDecision | level 0/1/2, strong bool, audio command hoặc null, message, timestamp; không đổi class của Prediction |
| UiSnapshot | preview sở hữu bytes, FeatureSample/temporal display, DetectionResult, AlertDecision, FPS/latency/staleness |

`SystemStatus` theo [02](02_Problem_Definition.md); idle có thể hiển thị trước Start nhưng không là label AI. Metadata storage có nhãn ở manifest, không truyền nhãn vào FeatureExtractor.

### Nguồn timestamp
VideoReader ưu tiên thời gian decode `CAP_PROP_POS_MSEC`, kiểm tra tăng nghiêm ngặt và đối chiếu duration. Nếu backend trả timestamp lỗi, chỉ fallback `frame_index / fps` khi metadata/khảo sát xác nhận nguồn constant-FPS và fps dương; ghi rõ phương pháp vào metadata. Với variable-FPS không có timestamp tin cậy, dừng preprocessing video đó và ghi reason, không âm thầm giả 30 FPS. Webcam ghi monotonic timestamp lúc capture, không lúc inference hoàn tất; chênh lệch dùng để đo tuổi frame.

Implementation Phase 3: webcam dùng **`time.perf_counter_ns()`**, clock monotonic `QueryPerformanceCounter` trên Python 3.12 Windows; timestamp ghi ngay sau capture, trước inference. `monotonic_ns()` ở môi trường này dùng GetTickCount64/15.625 ms và đã gây timestamp trùng trong smoke. Consumer đo tuổi frame phải dùng **cùng clock perf_counter**, không trộn epoch giữa clocks. Stats ghi implementation/resolution. File dùng thời gian decode và chỉ fallback khi caller truyền `constant_fps_verified=True` sau verification độc lập.

Sampling chọn frame đầu tiên mỗi bucket `floor((source_ms-origin_ms)*target_fps/1000)`, tối đa một frame/bucket và target≤20; không duplicate nếu nguồn chậm, không bịa một chuỗi timestamp đều. Khoảng cách giữa hai frame có thể jitter theo cadence nguồn; causal 10 Hz tick/sample-age thuộc Phase 7/temporal, không phải reader này. Prefix stop là `STOPPED`, full EOF mới là `EOF`; disconnect/decode lỗi là `ERROR`, có release.

## CV và offline
| Module / file | Purpose, Input → Output; Interface | Dependencies và quy tắc |
|---|---|---|
| VideoReader / preprocessing/video_reader.py | File/camera → iterator FramePacket; `VideoReader()`, `iter_frames(path,target_fps,*,constant_fps_verified=False)`, `iter_camera(camera_index,target_fps,*,width,height)`, `close()`, `stats` | OpenCV; decode tuần tự, chọn theo source timestamp, không seek mỗi frame; lỗi duration/FPS phải ghi; camera failure không là EOF |
| FaceLandmarkDetector / features/landmarks.py | FramePacket → LandmarkResult; `FaceLandmarkDetector(config)`, `detect(packet)`, `close()`, `stats` | MediaPipe Tasks VIDEO, asset; BGR→RGB, timestamp/source-bound session; tạo detector mới mỗi video; no-face/malformed points trả null + false, không confidence/visibility giả |
| EyeFeatureExtractor / features/eye.py | LandmarkResult → EyeFeatures; `EyeFeatureExtractor(epsilon:float)`, `extract(result)` | Stateless NumPy, x*W/y*H trước khoảng cách, z ignored; canonical `LEFT_EYE`/`RIGHT_EYE` theo [06]; từng mắt độc lập, mean chỉ khi cả hai valid |
| MouthFeatureExtractor / features/mouth.py | LandmarkResult → MouthFeatures; `MouthFeatureExtractor(epsilon:float)`, `extract(result)` | Stateless NumPy, inner-lip `MOUTH_HORIZONTAL`/`MOUTH_VERTICAL`; sum3 dọc/(3*ngang), không clamp và không kết luận ngáp từ một frame |
| HeadPoseEstimator / features/head_pose.py | `HeadPoseEstimator(canonical_path,camera_model,*,max_reprojection_error_norm,epsilon)`, `estimate(result) -> PoseFeatures`, `camera_metadata(image_size)` | Stateless ITERATIVE solvePnP; canonical D=diag(1,−1,−1), R=Rz(roll) Ry(−yaw_display) Rx(pitch); pitch+ xuống, yaw+ phải ảnh, roll+ CW unmirrored. K approximate công khai hoặc calibrated resize; selected XY, hull/depth/Euler/RMS gates; invalid giữ NaN/false và RMS hữu hạn khi đo được |
| QualityGate / features/quality.py | `QualityGate(config["quality"])`, `evaluate(packet,result,eyes,mouth,pose) -> QualityResult`; `measure_quality(image_bgr,result)` | Immutable flags/reasons/metrics snapshot; fixed256 ROI, independent widths; optical failure mask all, oblique/invalid pose mask eyes+mouth, manual occlusion eyes-only; no classifier/temporal state |
| FeaturePipeline / features/pipeline.py | `FeaturePipeline(config)`, `process(packet) -> FeatureSample`, `reset()`, `close()`; read-only last_landmarks/last_quality/stats/pose_estimator | Frozen config validated before native allocation; one detector/session. Strict source/time/index; clear diagnostics before processing/errors/reset/close. Reset closes/recreates native and clears counters; close terminal/idempotent. Raw unnormalized/no labels; per-stage timings distinct from detect |
| FeatureDatasetBuilder / preprocessing/builder.py | `FeatureDatasetBuilder(config,*,constant_fps_verified=False)`, `build(manifest,output_dir,*,video_ids=None) -> dict` | Working validator accepts partial subject coverage, acquisition validator stays strict. Streaming1024 scalar rows, fullEOF and release, source SHA before/after. Metadata-last marker, Parquet SHA/schema/footer/fingerprint required for resume; per-source failure continues, nonzero CLI |
| Snapshot runner / preprocessing/snapshot.py | `freeze_snapshot(config_path,manifest_path,run_dir,output_dir) -> dict`, `run_snapshot(config_path,manifest_path,run_dir,output_dir) -> dict` | Byte-copy freeze + self-hash/source/program/signature provenance; drift refuse, sequential unchanged builder against full frozen manifest, checkpoint every pending/completed/cached/failed member; no downloads/source-status mutation |
| Feature audit / preprocessing/audit.py | `audit_snapshot(config_path,manifest_path,output_dir,snapshot,extraction_report) -> dict` | Read-only pair validation and streaming row/identity/null/time/count/release audit; frame-weighted video/subject/class coverage, explicit failed support and overlapping reason counts; CLI publishes separate extraction-status manifest |
| Splits / datasets/splits.py | `make_outer_split(manifest,test_fold)`, `build_splits(...)` → five restricted official outer slots | Subject-disjoint train/validation/test; train-only QC/P0 numerics; retain invalid profiles and complete source membership |
| Profiles / calibration/profile.py | `fit_qc_policy(...)`, `estimate_profile(...)`, `transform_sample(sample,profile)` | P1 Alert prefix30–60s/20s valid, independent eyes, train QC bounds/stability, strict camera/schema/assets; no silent P0 fallback |
| TemporalFeatureExtractor / features/temporal.py | `TemporalFeatureExtractor(profile,config)`, `update(sample) -> list[TemporalSample]`, `reset()` | Native event transitions; causal10Hz latest-not-future hold≤100ms; gap>1s/source switch resets; exact16 names/masks and clipped60s duration/event receipts |
| Derived cache / datasets/derived.py | `build_derived(split,profiles,*,raw_dir,snapshot,temporal_config,output_dir,mode)`, `read_temporal(path)` | Per-video typed16-channel Parquet; source/raw-commit/snapshot/split/profile/config/code identities; changed identity gets a new directory |
| Window index / datasets/sequence.py | `build_window_index(temporal,*,split,config)` → accepted index with coverage | 100 ticks spanning9.9s, nominal10s/stride1; exclusive end_row; missing>20%, gap/current masks/prefix gates; mode/outer identity bound by dataset manifest |
| SequenceDataset / datasets/sequence.py | `SequenceDataset(index,*,derived_manifest,scaler,feature_names)`, `__getitem__` → `(x,label_id,metadata)` | Immutable per-video cache, returned float32[100,16] copy; integer label and IDs outside X; exact scaler/manifest protocol binding |
| Normalization / datasets/normalization.py | `fit_scaler(train_unique_timesteps,feature_names)`, `transform_values(values,validity,scaler)` | Unique accepted-train(video,segment,time), valid-only first10 channels; zero-observation channel blocks; invalid→0 after scale; six flags unscaled |

### Eye/mouth implementation semantics — Phase4/5
- Epsilon lấy từ resolved preprocessing config (hiện1e-6), hữu hạn/dương/non-boolean; áp cho denominator pixel đầy đủ:2*eye_width hoặc3*mouth_width. Denominator≤epsilon invalid, không dùng epsilon để bịa một ratio.
- Missing face/points trả NaN + false masks. Khi face/points present, sai floating ndarray[478,3] hoặc sai positive-integer tuple(W,H) là ValueError, không pad/truncate. Chỉ selected x/y phải finite và trong[0,1]; selected z hoặc unselected points không ảnh hưởng ratio.
- Closed vertical span0 với horizontal width hợp lệ trả valid0; invalid không đổi thành0. Invalid một mắt giữ mắt kia, nhưng EAR mean NaN. Không mutate points, không đổi timestamp, không thêm confidence/visibility.
- Geometry validity không là quality. Phase6/7 mask raw eye/mouth measurements theo pose/ROI/optics, giữ independent masks và finite RMS diagnostic khi có; không clamp ratio để che profile-view artifacts.
- Preview dùng một FeaturePipeline cho headless/GUI; derive gated display records từ sample, không extraction/lifecycle path thứ hai. Summaries valid-only/null, `inference_ms` detect-only; source cleanup failures được báo, các closers còn lại và source tiếp theo vẫn chạy. Mặc định không lưu ảnh.


## Model và evaluation
| Module / file | Purpose, Input → Output; Interface | Dependencies |
|---|---|---|
| RuleBasedClassifier / models/rules.py | `RuleBasedClassifier(config,mode='temporal')`, `predict(TemporalSample)` | Mode ear_only/temporal; missing/startup abstains; valid outputs diagnostic one-hot[3], not calibrated probabilities |
| BaselineClassifier / models/baseline.py | `fit(X,y,sample_weight=None)`, `predict_proba(X) -> [N,3]`, `save(path)`, `load(path,trusted=True)` | Train median + all-column missing flags; all-missing median0; RF300/depth12/leaf5/balanced/seed42, bounded workers; trusted local joblib only |
| LSTMClassifier / models/lstm.py | `forward(x:[B,100,16]) -> logits[B,3]` | Unidirectional1-layer hidden64, internal dropout0; last hidden→32/ReLU/head dropout0.3→3; reset state every window |
| ModelTrainer / training/trainer.py | model, train_loader, val_loader, config → best state/history/validation predictions; `fit()` | Real Adam/backprop/clip1; val Macro F1/patience8, max50; require three train/val classes; never receives test loader |
| ModelEvaluator / evaluation/evaluator.py | `evaluate(labels,probabilities,*,metadata,coverage)` → training metrics dict | Class order0/1/2, confusion/support/coverage; full-three-class macro null nếu thiếu truth class, supported macro separately named |
| ModelBundle / models/bundle.py | `save(path,*,model,scaler,metadata)`, `load(path,trusted=True)`, `validate_schema(names,hashes)`, `predict_proba(scaled_x)` | Directory weights.pt/metadata.json/scaler.json/integrity.json; CPU weights_only state_dict load; strict16 names, mode and six hashes; no fallback |
| Evaluation functions / evaluation/evaluator.py | module `evaluate(...,subject_ids,video_ids,min_video_coverage)` và `aggregate_folds()` → nested window/video/subject/bootstrap reports | Macro trên supported classes, explicit classes_present/full_class_coverage; khác contract training ModelEvaluator; không claim five-fold khi chỉ có outer0 |
| Comparison cohorts / datasets/cohorts.py | `select_p0_cohort`, `select_p1_comparison_cohort`, `keep_after_calibration_prefix` | Intersection subject sets, explicit dropped/P1 coverage; matched prefix mask loại start < end; độc lập summaries để giữ derived producer fingerprint |

## Calibration, realtime và UI

CalibrationManager, PredictionBuffer, PredictionSmoother và DrowsinessDetector dưới đây đã triển khai và có regression tests. Native `scripts.webcam_demo` vẫn là diagnostic profile/temporal/rules riêng; alerts, camera worker và Qt lifecycle chưa triển khai.

| Module / file | Purpose, Input → Output; Interface | Dependencies và failure |
|---|---|---|
| CalibrationManager / calibration/manager.py | valid raw samples → profile/status; `start(timestamp_ms)`, `update(sample)`, `finish()`, `reset()` | Profile estimators ở calibration/profile.py, policy [11]; timeout fail; không fine-tune |
| PredictionBuffer / realtime/buffer.py | TemporalSample → SequenceWindow hoặc null; `append(sample)`, `get_window()`, `reset()` | deque, scaler/bundle; đủ 100 bước và quality gate; tên buffer giữ **feature**, không giữ probability |
| PredictionSmoother / realtime/smoother.py | valid Prediction → smooth Prediction hoặc null; `update(prediction)`, `reset()` | deque 3 probabilities, expire theo timestamp, chưa đủ mẫu trả null |
| DrowsinessDetector / realtime/detector.py | FramePacket → DetectionResult; `process(packet)`, `reset_session()`, `close()` | FeaturePipeline, calibration, temporal, buffer, model bundle, smoother; status trước class, prediction cadence |
| AlertManager / alerts/manager.py | DetectionResult + timestamp → AlertDecision; `update(result, timestamp_ms)`, `set_muted(bool)`, `reset()` | State machine [12]; unknown không kéo timer, audio chỉ command |
| CameraWorker / realtime/camera_worker.py | camera/config → signal UiSnapshot/status; `start()`, `request_stop()`, `request_calibration()` | QtCore, capture latest slot, detector, alerts; không mutate widget, stop thread-safe |
| MainWindow / ui/main_window.py | UiSnapshot/button actions → UI và worker commands; `render(snapshot)`, `start_session()`, `stop_session()` | PySide6 Widgets/Multimedia; owns audio/UI, clean thread shutdown |

## Quy tắc lỗi interface
Bad config/path/schema là exception startup có thông điệp; sample chất lượng kém là output invalid có reason, không exception mỗi frame. EOF video là iterator kết thúc; camera read error là status ERROR với release. Probability invalid NaN, tổng không gần 1 hoặc sai shape → reject, log, không phát warning từ output đó.
