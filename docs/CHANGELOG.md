# Changelog

## 2026-10-07 — Phase 20/21 Desktop UI và Startup

- Cài đặt `MainWindow` bằng PySide6 hiển thị video, system status, mode P1/P0, EAR/MAR/PERCLOS/Pose, và smoothed probabilities.
- Render UI ở tốc độ tối đa 10Hz qua cấu trúc `UiSnapshot` truyền từ thread backend; các thông số thiếu được hiển thị là `N/A` thay vì fake metrics.
- Âm thanh cảnh báo `QSoundEffect` được tích hợp trên main thread theo `AlertDecision.audio_command`, bao gồm nút Mute.
- Triển khai `main.py` nhận args `--config` đọc `configs/realtime.yaml`, validate assets, models và mode lúc cold-start.
- Cung cấp flag `--fixture` cho mục đích test UI layout, render, timer, và smoke test widget mà không cần `CameraWorker` (Phase 19 từ Team 3 chưa hoàn thành).
- Chuẩn hóa contract nhận/gửi tín hiệu luồng an toàn qua PySide6 Signals. Đã cập nhật README hướng dẫn chạy desktop UI.

## 2026-10-07 — Người 2 Phase21 replay/parity core

- Thêm chronological `replay_session`: dùng timestamp nguồn làm current time, giữ mọi scheduled record kể cả NO_FACE/unavailable và từ chối nối source/timestamp/frame order sai.
- Mỗi replay record giữ tám giá trị `FeatureSample` (giá trị không hữu hạn thành null) và năm validity mask; trace phải khớp source/frame/timestamp của packet hiện tại.
- Thêm `compare_replays` với tolerance khai báo trước; so feature/validity, identity/status/class/raw+smooth probability và ghi rõ missing/unexpected timestamp thay vì loại khỏi report.
- Detector nhận optional `current_time_ms` chỉ cho replay, live mặc định vẫn dùng perf_counter clock. Native parity với Phase9/10/13 bundle thật còn là integration acceptance chưa thể chạy trong snapshot hiện tại.

## 2026-10-07 — Người 2 Phase18 prediction smoothing core

- Thêm `PredictionSmoother` mean-N probability vectors (mặc định3), chưa đủ mẫu trả null, gap `>=2s`/reset xóa lịch sử và không majority-vote class.
- Reject prediction invalid/NaN/sai shape/sai tổng/model-mix; detector giữ raw ngay và chỉ xuất smooth sau ba prediction hợp lệ.
- Thêm behavioral regressions cho averaging/expiry/invalid/model identity và detector integration; classifier metrics vẫn dùng raw, realtime policy mới dùng smooth.

## 2026-10-07 — Người 2 Phase17 realtime buffer/detector core

- Thêm `PredictionBuffer`: cửa sổ fixed-length theo ordered features, reset theo segment/gap, reject current-invalid và missing ratio, chỉ xuất `SequenceWindow` hữu hạn sau transformer/scaler được inject.
- Thêm `DrowsinessDetector`: orchestration raw pipeline→calibration→temporal→buffer→model, cadence1s, stale/no-face/current-invalid status, P0/P1 mode/schema/order checks, reset/close và probability validation.
- Thêm13 behavioral regressions cho100-step window, missing20%, gap/segment, cadence, stale, no-face reset, P1 startup và invalid model output. Core chưa được gọi native-complete cho đến khi Phase9/10/12/13 cung cấp estimator/temporal/scaler/bundle thật.

## 2026-10-07 — Người 2 Phase16 calibration lifecycle core

- Thêm `CalibrationManager` cho P0/P1 với state `IDLE/COLLECTING/COMPLETE/FAILED`, wall-clock 30s, tối thiểu 20s valid, timeout60s, retry/reset và profile freeze.
- Phase9 profile estimator được inject thay vì viết lại trong realtime; manager từ chối profile sai mode/schema/asset/resolution và không P1→P0 fallback.
- Thêm9 behavioral regressions cho completion, timeout, retry, hai baseline mắt khác nhau, profile mismatch, P0 explicit và timestamp strict. Đây là core đã implement; native webcam/profile-estimator integration còn phụ thuộc Phase9/13.

## 2026-10-07 — Team handoff và GitHub publication policy

- Thêm bốn tài liệu `Team_01`–`Team_04`: evaluation, realtime core, alerts/camera worker và desktop UI/startup; có ownership, prerequisites, tài liệu nguồn và checklist nghiệm thu bằng evidence thật.
- `.gitignore` loại toàn bộ `data/`, `runs/`, môi trường local, dataset media/archives, generated model weights và secret/key files khỏi các lần add mới. Dataset và run artifacts vẫn giữ local; ignore không tự xóa file đã được track trong lịch sử.

## 2026-10-07 — Phase6 signed pose và Phase7 raw pipeline/builder

- Thêm stateless six-point `HeadPoseEstimator`, proper canonical D=diag(1,−1,−1), signed display Euler, calibrated/approximate K resize và hull/depth/singular/RMS guards. Không offset180 hoặc fake neutral. Sửa convention docs06/15, giữ shared records.
- Thêm fixed256 `measure_quality`/immutable `QualityGate`, measurement CLI và frozen `configs/quality_policy_v1.json`:12 visually reviewed clean samples giữ12/12,36 dark/overexposure/Gaussian controls reject36/36. Actual thresholds/report SHA pin trong preprocessing YAML; no profile→startup fail, không class-driven tuning/baseline0.
- `FeaturePipeline` sở hữu một native session; raw gated samples có strict identity/time/index, independent masks/mean, NaN/false, reset/terminal close và stage diagnostics. Preview cutover hoàn toàn, actual windows normal/manual-eye/blur/dark/profile đã xem; unknown hiển thị N/A, không Alert hoặc ratio cap.
- Working validation nhận subject51 partial coverage/error rows, giữ acquisition completeness strict. Builder stream1024 rows, join label tại storage, fullEOF/release/source before+after hashes; unique stages và metadata-last marker. Resume check SHA/schema/footer/fingerprint/source/artifacts/dependencies/CFR, không file-existence shortcut hoặc multi-writer guarantee.
- Native CLI `--video-id 04_0 --video-id 04_5 --video-id 04_10`:3 completed EOF outputs12291/12149/12315 rows (36,755 total), actual Arrow/time/null/label/hash round-trip verified; rerun3/3 cached. Reports `runs/phase7/full_three_clips/`, `resume/`; chưa chạy all34. Native15FPS blank15frame EOF giữ15 null rows; same5 saved packets qua independent native sessions tương đương1e-6.
- Reproduced/fixed Windows read-only fsync EBADF (`r+b` writable flush), duplicate JSON-text policy comparison rejecting35 vs35.0, và native-close exception escaping preview source reporting. Covering regressions observed RED/GREEN; final `.venv/Scripts/python.exe -m pytest -q --tb=short`: **394 passed in12.01s**. Numerical review clean; raw/storage review clean sau scoped cleanup re-review.
- Webcam Phase6/45s814 vàPhase7/45s806 emitted đều no-face, monotonic QPC và release; physical signs/occlusion/face transitions/disconnect vẫn chưa quan sát. Actual recorded9000ms profile bị quality gate; old416000ms MAR outlier thành no-face/null trong new sequential run, không suy pose-only cause hoặc mouth-open truth.
- Coverage risk giữ nguyên và báo rõ:04_0 left-eye2767/12291, mouth2873/12291 valid dưới raw absolute pose policy; không claim đủ cho training/universal quality. Không tải thêm11 missing, mutate original manifest, split/calibrate/train, fake P1 baseline hoặc silentP0 fallback.

## Quyết định phạm vi — dùng dataset hiện tại

- User chốt tiếp tục với **34 video/12subjects** hiện có; không đợi hoặc tự tải thêm11 file. Acquisition plan45/missing ledger và full-source gate được giữ, không đổi partial thành complete.
- Roadmap8 xử lý working snapshot sau QC7; roadmap9/training/evaluation dùng restricted official membership và cardinality thực tế, không ép9/3/3 hoặc36/12/12. Freeze paths/hashes và version snapshot khi thêm nguồn.
- Giữ subject51/Low Vigilance trong provenance/coverage. Thiếu Alert khiến P1 abstain và fold5 P1 validation/test blocked/undefined; không fake baseline hoặc silent P0 fallback. Không đổi calibration config/checkpoint mode.
- Đây là đổi phạm vi tài liệu cho các phase tiếp theo, không training/splits implementation mới và không claim full benchmark. Phase3/hardware verification giữ kết quả đã báo.


## 2026-10-06 — Phase4 EAR và Phase5 MAR

- Thêm stateless `EyeFeatureExtractor(epsilon)`/`MouthFeatureExtractor(epsilon)` và79 deterministic regressions; dùng đúng anatomical/inner-lip topology, pixelXY, complete-denominator epsilon, independent eye validity và mean, NaN/false vs valid0. Shared contracts/schema/config không đổi; không clamp ratios.
- Preview gọi geometry cả headless/no-face, dùng canonical topology thay duplicate eye IDs; thêm EAR/MAR/N/A và valid-only finite/null summaries. Detect latency giữ nguyên; sửa text scaling sau actual640px window cho thấy chữ HD quá nhỏ, đã xem landscape/portrait after-fix. Mặc định không lưu ảnh.
- Red-before: hai targeted suites fail đúng missing modules. Green:41 EAR/38 MAR; final `.venv/Scripts/python.exe -m pytest -q --tb=short` **178 passed in6.96s**. LSP feature/preview diagnostics không có issues; read-only geometry/preview reviews đều pass, không actionable findings.
- Ba clip subject04 prefix12s:723 face/geometry-valid frames, release; recorded eyes mở0ms EAR mean0.329221 → nhắm6280ms0.022919. Sequential20FPS mouth QC đến585400ms/11709 frames: MAR0.005320 →0.081579, raw lips khép/hé mở đã xem. Evidence ở `runs/phase4/` và `runs/phase5/mouth_sequence/`.
- Native blank/full15FPS synthetic EOF:15 no-face, không upsample, NaN/masks→N/A/null, release. Webcam45s:1326 decoded/821 emitted/505 dropped, tất cả821 no-face và null aggregates; no webcam images.
- Discovery5FPS decode đủ3 subject04 clips để tìm transition; high MAR khi quay đầu không là mouth-open/quality proof. Nói chuyện có xác nhận, human webcam transitions và physical camera disconnect vẫn chưa quan sát; không claim nghiệm thu hardware đầy đủ.
- Dataset working34/12 và acquisition45/missing11/provenance giữ nguyên; không download thêm, không pose/calibration/temporal classification/feature builder/training/realtime UI.


## 2026-10-06 — Phase 3

- Thêm sequential `VideoReader`, MediaPipe Tasks VIDEO `FaceLandmarkDetector`, preview chưa mirror và regression tests; giữ shared contracts. Source timestamps strict, target-grid≤20, không upsample, fallback CFR chỉ khi caller đã kiểm chứng; EOF/STOPPED/ERROR và release phân biệt.
- Runtime ba prefix 12s, subject04/cả ba class: 723 face frames, zero invalid landmarks; overlay topology/aspect đúng đã xem trên actual preview HWND. Model no-face/blank và full synthetic EOF đã chạy. Không decode toàn dataset.
- Camera0 sau sửa: 218 decoded/131 emitted, tất cả no-face; capture/model release. Camera99 unavailable: ERROR/exit1/release. Chưa physical disconnect hoặc human face→no-face transition, không claim hardware checklist hoàn chỉnh.
- Sửa clock Python3.12 Windows từ coarse GetTickCount64/monotonic_ns sang high-resolution monotonic perf_counter_ns/QPC. Regression fail-before/pass-after và webcam runtime xác nhận. Contract15 yêu cầu future age/staleness dùng cùng clock.
- Sửa native model initialization failure giữ source error report; regression fail-before/pass-after. Preview resize giữ source aspect ratio.
- Final recheck: **99 passed in 8.89s**; actual video prefix3s:61 face frames, no errors, capture/model release, exit0 (`runs/phase3/final_check/preview_report.json`). Read-only temporal/detector reviews đã hoàn tất; actionable constructor-report finding đã sửa.
- [Phase3_Report.md](Phase3_Report.md) ghi evidence và readiness: đủ bắt đầu Phase4–6 trên clip nhỏ; dataset34/45 và pipeline/training/UI tương lai vẫn chưa nghiệm thu. Đề xuất thảo luận restricted15-subject protocol thay cho full60 targets còn trong roadmap, không silently migrate.


## 2026-10-06 — Phase 0–2

### Tài liệu
- Chuyển 21 tài liệu kỹ thuật Markdown gốc vào `docs/`; root README là lối vào và hướng dẫn chạy thực tế.
- Cập nhật roadmap, cấu trúc project, shared contracts và môi trường đã kiểm chứng. Nguồn, quyền ảnh, selection policy, checksum và giới hạn dữ liệu ghi trong [Dataset_Access.md](Dataset_Access.md).

### Phase 0 — đạt
- Python 3.12 x64 trong `.venv`; requirements và lockfile, config YAML có validation và path resolution độc lập working directory; shared dataclasses/enums.
- Smoke MediaPipe Tasks VIDEO trên hai frame video thật: một mặt/frame, 478 landmarks/mặt. Asset chính thức có URL/SHA256 metadata.
- PyTorch CPU: forward LSTM `[1,100,16] → [1,3]`; cửa sổ Qt thật và screenshot chỉ chứa text; `pip check` không có broken requirements.
- Bằng chứng: `runs/phase0/environment_report.json`, resolved configs và `qt_smoke.png`. Network smoke dùng trọng số ngẫu nhiên, **không phải classifier đã train**. CUDA chưa cài/chưa smoke.

### Phase 1 — code đã chạy, dữ liệu chưa đủ
- Inventory ZIP64 chính thức: 10 archives, 182 physical files / 180 logical recordings; giữ multipart IDs và official fold provenance.
- Kế hoạch 15 complete subjects, ba người/fold, 45 file; chọn dung lượng nhỏ nhất, không phải full/random benchmark. Chỉ stream exact compressed-member Range; từ chối HTTP 200, không tải full ZIP. CRC32/size trước publish, SHA256 và receipts sau xác minh.
- Sửa an toàn path/temporary/receipt để không ghi đè file ngoài raw root qua symlink/hardlink; dọn partial thuộc plan trước disk preflight. Streaming disk reserve là best-effort trước concurrent/external writes.
- **34/45 video verified, 12/15 subjects, 8,206,607,880 bytes (~7.64 GiB)**. 11 file còn thiếu bị Drive quota; cả route chính thức/form và các Range thử nghiệm chưa hoàn tất chúng. Không thay subject/mirror để che phần thiếu. Xóa scaffolding thử nghiệm và inventory prototype lỗi thời.

### Phase 2 — exploration thật, full acceptance bị chặn
- Manifest Parquet giữ leading-zero IDs, nhãn nội bộ nullable, lỗi, hash, official fold source và `split="unassigned"`. Kiểm tra exact plan identity và receipt provenance, không chỉ so số lượng.
- Trên 34 video: SHA256 và decode đầu/giữa/cuối hợp lệ; class counts 11/12/11, subject counts theo fold 3/3/3/2/1, khoảng 5.796 giờ. Duration/FPS là metadata container, chưa khảo sát VFR toàn clip.
- CLI **exit 1 đúng** vì thiếu dữ liệu, có missing IDs/download errors trong `runs/phase2/exploration_report.json`; không claim Phase 1–2 đã đạt.
- Notebook chạy thành công cả ba code cells bằng kernel `.venv` thật, in 34/45 và 12/15. Kernel TCP có cảnh báo không mã hóa; không expose lên mạng công cộng. Bằng chứng: `runs/phase2/notebook_smoke.json`.
- Storage check: không có video ngoài plan, full ZIP hoặc partial; còn 51,347,931,136 bytes trống tại lúc kiểm tra. Bằng chứng: `runs/phase2/storage_check.json`.
- Sửa environment smoke dùng report directory ngoài project; cửa sổ Qt và report đã chạy thành công với directory đó.

### Verification và phạm vi
- Regression suite bao phủ config boundaries, HTTP Range/CRC/SHA, path/partial safety, metadata/labels và plan/source mismatch. Lệnh `.venv/Scripts/python.exe -m pytest -q --tb=short`: **36 passed in 5.72s**.
- Không tải/request NTHU hoặc YawDD; không training, accuracy, checkpoint giả hay runtime Phase 3 trở đi.
