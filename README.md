# Driver Drowsiness Detection

Tài liệu thiết kế đã được chuyển vào **[docs/README.md](docs/README.md)**. Roadmap triển khai: [docs/16_Development_Roadmap.md](docs/16_Development_Roadmap.md).

## Phase 0–13
**Trạng thái nghiệm thu:** Phase 0 đạt. Code Phase 1–2 đã có nhưng dataset mới **34/45 video, 12/15 subjects, 8,206,607,880 bytes (~7.64 GiB)**. Drive chặn 11 file bằng “Quota exceeded”; **Phase 1–2 chưa đạt full acceptance**. Exploration chạy trên dữ liệu thật và exit 1 đúng để chặn báo thành công sai. Xem [Dataset Access](docs/Dataset_Access.md) và [changelog](docs/CHANGELOG.md).

- Python **3.12 x64**, môi trường riêng `.venv/`.
- `configs/`: cấu hình preprocessing/training/realtime; `src/contracts.py`: dữ liệu dùng chung.
- `scripts/check_environment.py`: tải asset chính thức và kiểm tra MediaPipe, PyTorch, cửa sổ Qt thật.
- `scripts/acquire_manifest.py`: chỉ tải thành viên ZIP đã chọn bằng HTTP Range, không tải toàn archive.
- `scripts/explore_dataset.py`: kiểm tra video thật, checksum, nhãn/fold; xuất manifest Parquet và report.

Kế hoạch UTA subset: **15 người × 3 trạng thái = 45 video**, ba người từ mỗi official fold; **chưa tải đủ**. Chọn các subject đầy đủ có tổng dung lượng thấp nhất để tiết kiệm ổ đĩa. Đây là **development subset thiên lệch theo dung lượng**, không đại diện full benchmark 60 người. Dữ liệu đã tải nằm tại `data/raw/uta_rldd/`, ngay trong project; không commit vào Git.

**Phạm vi làm việc được user chốt:** tiếp tục development với **34 video hiện có / 12 subjects**, không đợi đủ45, không tự tải thêm. Kế hoạch acquisition45 và ledger11 missing được giữ để truy nguyên, không sửa thành “đã đủ”. Phase8 xử lý snapshot34 sau khi Phase7 qua QC; Phase9 chia theo subject/official fold với số lượng thực tế, không cố định9/3/3 hoặc36/12/12. Subject51 thiếu Alert nên P1 phải abstain, không lấy Low Vigilance làm baseline hoặc tự đổi sang P0.

Phase3–8 đã có CV/raw pipeline, signed pose và frozen quality. **Phase8 xử lý/audit34/34 videos, 379,361 rows; resume34/34cached**, historical441tests. Có9 videos0eye/mouth/pose-valid, subjects18/45 hoàn toàn0coverage dưới policy hiện tại: **extraction complete không là training-ready**. Giữ nguyên snapshot/raw config và tất cả rejected sources; không tải thêm dữ liệu.

## Phase9–13 — kết quả đã chạy

Đã có five restricted official splits, train-only QC/P0 và pure P1 profiles, causal native events/10Hz temporal, shared window index, RF, unique-train scaler và LSTM trainer/trusted bundle. **Full regression604passed27.11s.** P0outer0 dùng3863train/2083validation/3861test windows; test accepted3861/5361scheduled. Metrics chỉ có điều kiện trên windows đạt gate của development subset, không phải full UTA benchmark.

| P0 outer0, seed42 | Validation Macro F1 | Test Macro F1 | Selection |
|---|---:|---:|---|
| RF | 0.173248 | 0.238839 | Fixed config, no tuning |
| LSTM | 0.276877 | 0.230164 | Best epoch28; early stop after36epochs |

Independent fresh-process reload đã đạt: RF full3861test probabilities atol1e-12; LSTM real64×100×16 batch atol1e-6. LSTM không thắng RF trên test; cả hai hiện có chất lượng thấp, **không dùng cho an toàn lái xe**.

**P1 cả5outer slots bị data gate chặn** do profile QC/camera provenance và thiếu accepted validation/test support. Actual RF/LSTM P1outer0 CLIs trả blocked/nonzero, không tạo checkpoint và không fallback P0. P0outer3 thiếu validation classes; outer4 chỉ có Low Vigilance ở test. Full-fold aggregation/bootstrap/realtime manager/audio/Qt UI thuộc Phase14 trở đi, chưa triển khai.

### Lệnh offline — PowerShell tại project root
```powershell
.\.venv\Scripts\python.exe -m scripts.build_splits --snapshot-run runs/phase8 --split-dir data/splits --run-dir runs/phase9
.\.venv\Scripts\python.exe -m scripts.build_derived --mode P0 --split-dir data/splits --snapshot-run runs/phase8 --output-root data/processed --run-dir runs/phase11/derived
.\.venv\Scripts\python.exe -m scripts.build_derived --mode P1 --split-dir data/splits --snapshot-run runs/phase8 --output-root data/processed --run-dir runs/phase11/derived
$p0 = ((Get-Content runs/phase11/derived/P0.json -Raw | ConvertFrom-Json).artifacts | Where-Object outer_index -eq 0).path
.\.venv\Scripts\python.exe -m scripts.train_baseline --derived-manifest $p0 --outer-index 0 --mode P0 --config configs/training.yaml --run-dir runs/phase11/rf_p0_outer0_repeat
.\.venv\Scripts\python.exe -m scripts.train_lstm --derived-manifest $p0 --outer-index 0 --mode P0 --config configs/training.yaml --run-dir runs/phase13/lstm_p0_outer0_repeat --cpu-threads 4
```
Training run-dir phải mới hoặc rỗng; không ghi đè frozen run. Dataset/derived manifests ghi absolute local paths và SHA; clone mới cần dữ liệu hợp pháp và snapshot local tương ứng, không tải từ GitHub.

### Webcam — cần người kiểm tra trực tiếp
```powershell
.\.venv\Scripts\python.exe -m scripts.webcam_demo --source webcam --camera-index 0 --snapshot-run runs/phase8 --profiles data/splits/outer_0.json --mode P1 --confirm-alert --seconds 120 --log-dir runs/phase10/manual_p1_check
```
Chỉ xác nhận khi đang tỉnh táo, ngồi an toàn; **không chạy khi lái xe**. 30–60s đầu nhìn thẳng/mở mắt tự nhiên/ánh sáng ổn định; sau calibration thử blink/closure/mouth-open/turn/cover-face. Failed calibration giữ UNRELIABLE, phải restart để retry. Mất/invalid dữ liệu không thành Drowsy. Camera access smoke không là physical acceptance: lượt đầu60faces nhưng brightness QC reject; lượt cuối56no-face, cả hai rules abstain và release. Manual scenarios/disconnect vẫn NOT_RUN. Không lưu ảnh mặc định; dùng log-dir mới cho mỗi lượt.


## Cài lại môi trường — PowerShell
```powershell
py -3.12 -m venv .venv
uv pip install --python .venv/Scripts/python.exe --no-cache torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe --no-cache -r requirements-lock.txt
```

Nếu chưa có `uv`, dùng `.venv/Scripts/python.exe -m pip install --no-cache-dir ...` với cùng package/index. `requirements-lock.txt` chỉ được tạo sau khi full environment smoke đạt. Bản CPU được chọn cho Phase 0 để giảm dung lượng; máy có GPU không có nghĩa CPU wheel chạy CUDA. Khi đến training, chọn CUDA wheel theo [PyTorch installer chính thức](https://pytorch.org/get-started/locally/) và kiểm tra lại `torch.cuda.is_available()`; không cần cài CUDA Toolkit cho Phase 0.

### CUDA chỉ khi đến training — chưa cài/chưa smoke
Official [CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/) có `torch-2.14.1+cu130-cp312-cp312-win_amd64.whl`. Máy hiện có RTX 4050 Laptop 6 GB, driver 610.62; cần kiểm tra lại driver và dung lượng trước cutover. Lệnh tham khảo, **không chạy ở Phase 0**:
```powershell
uv pip install --python .venv/Scripts/python.exe --no-cache --reinstall-package torch torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.rand(1, device='cuda').item())"
```
Sau khi GPU smoke đạt, cập nhật training `device`, environment report và lockfile riêng; lockfile CPU hiện tại không được dùng để ép CUDA wheel trở lại CPU.

## Chạy công cụ
Chạy tại project root, không dùng Python global/Conda mặc định:

```powershell
.venv/Scripts/python.exe -m scripts.check_environment --download-assets --assets-only
.venv/Scripts/python.exe -u -m scripts.acquire_manifest --workers 3
.venv/Scripts/python.exe -m scripts.explore_dataset
.venv/Scripts/python.exe -m scripts.check_environment --sample-video data/raw/uta_rldd/04/0.mp4
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.preview_landmarks --video data/raw/uta_rldd/04/0.mp4 --seconds 12
.venv/Scripts/python.exe -m scripts.preview_landmarks --camera 0 --seconds 45
```

### Raw feature extraction — Phase6/7
```powershell
.venv/Scripts/python.exe -m scripts.preprocess --config configs/preprocessing.yaml --manifest data/processed/manifest.parquet --video-id 04_0 --video-id 04_5 --video-id 04_10 --output-dir runs/phase7/raw_features_smoke --report-dir runs/phase7/full_three_clips
.venv/Scripts/python.exe -m scripts.measure_quality --video data/raw/uta_rldd/04/0.mp4 --video data/raw/uta_rldd/04/5.mp4 --video data/raw/uta_rldd/04/10.mp4 --start-ms 0 --end-ms 12000 --annotations runs/phase7/quality_annotations.json --report-dir runs/phase7/quality_measurements
```
`preprocess` bắt buộc chọn repeated `--video-id` hoặc explicit `--all-working-snapshot`; không mặc định chạy34 video. `--all-working-snapshot` chưa được thực thi trong Phase7. Output mặc định `data/processed/raw_features/`, report mặc định `runs/phase7/preprocess/`; không sửa manifest hoặc tự tải thêm. Không hỗ trợ nhiều writer đồng thời vào cùng output directory.

Quality profile nằm trong `configs/quality_policy_v1.json`, SHA được pin trong YAML:12 frame xem trực tiếp, giữ12/12 clean và reject36/36 degraded controls. Đây là development envelope subject04, **không universal calibration**. Phase6/7 dùng raw absolute yaw35°/pitch25°, không giả baseline cá nhân0. Full `04_0` chỉ còn2767/12291 left-eye và2873/12291 mouth valid; pose/camera gate gây coverage bias, phải xem coverage theo source/lớp trước Phase8/training, không nới gate theo nhãn lớp để che missing.

Muốn bật `quality.eyes_occluded`, phát hành bản frozen report/config tương ứng (cùng optical evidence, đổi explicit operator mode và SHA); startup kiểm tra policy/hash, không cho override âm thầm. Mode chỉ disable eyes, không tự phát hiện kính râm. Camera mặc định approximate; calibrated mode cần `reference_size: [W,H]`, finite3×3 `matrix`, OpenCV distortion vector. K resize theo kích thước frame thực.

Kiểm tra đuôi/tên video trong `data/acquisition/uta_subset_plan.json` nếu chạy smoke với video khác. Downloader kiểm tra file đã có bằng size/CRC, không tải lại video hợp lệ; partial thuộc kế hoạch được dọn trước preflight và video chưa hoàn tất phải tải lại từ đầu. Không thêm NTHU/YawDD và không tải các video ngoài kế hoạch 45 file.

Tạm thời không cần chạy lại acquisition để tiếp tục development. Nếu sau này mở rộng dữ liệu và quota nguồn được gỡ, chạy acquisition rồi exploration tuần tự. `--inventory data/acquisition/uta_source_inventory.json` giữ archive IDs nhưng vẫn re-fetch indexes; không bypass quota. File verified không tải lại; không đổi subject/classes để che phần thiếu.

Notebook: mở `notebooks/01_dataset_exploration.ipynb` bằng VS Code và chọn kernel `.venv` (Python 3.12); `ipykernel` đã nằm trong dev dependencies. Notebook in số file/subject thật, không coi plan 45 là kết quả tải.

Preview dùng MediaPipe Tasks VIDEO, RGB và session mới mỗi video. `--headless` không mở cửa sổ; `--save-overlay` là opt-in lưu ảnh mặt, phải kiểm tra quyền công bố. Mặc định không lưu ảnh. Không bật `--constant-fps-verified` chỉ vì FPS metadata dương: flag này là xác nhận của caller sau khảo sát constant-FPS độc lập. `--seconds` chỉ kiểm tra prefix; chạy file headless không pace như thời gian thực, FPS report là throughput xử lý, không FPS camera.

Preview gọi `FeaturePipeline.process` cho cả headless/GUI; hiển thị gated EAR/MAR/pose hoặc `N/A`, approximate/calibrated mode và categorical quality reasons. Face present không đồng nghĩa feature usable; oblique pose mask cả eyes/mouth, optical failure mask mọi channel; valid closed vertical span vẫn là0. JSON feature summaries dùng valid-only min/max/mean hoặc null; `pipeline` chứa lifecycle/counters/stage timings, `last_quality` có metrics/reasons. `inference_ms` vẫn detect-only, không gộp geometry/render. Không suy blink/yawn/Drowsy từ một raw measurement.

### Working snapshot và independent audit — Phase8
```powershell
.venv/Scripts/python.exe -m scripts.process_snapshot --run-dir runs/phase8 --config configs/preprocessing.yaml --manifest data/processed/manifest.parquet --output-dir data/processed/raw_features
.venv/Scripts/python.exe -m scripts.audit_features --run-dir runs/phase8 --output-dir data/processed/raw_features --status-manifest data/processed/extraction_status.parquet
```
Chạy hai lệnh **tuần tự**, không có writer khác trong output directory. Runner freeze byte-copy manifest/YAML cùng resolved signature, source/program hashes và acquisition missing IDs dưới `<run-dir>/snapshot/`; checkpoint `report.json` sau từng video, giữ pending/completed/cached/failed. Chạy lại cùng run-dir phải khớp freeze; đổi dữ liệu/config/assets/code thì tạo snapshot/run-dir mới, không ghi đè provenance cũ. Resume vẫn dùng complete-pair SHA/schema/fingerprint checks của builder, không skip theo file existence.

Audit không trích xuất lại hoặc sửa feature pairs: reopen toàn bộ rows, kiểm time/index/labels/null/masks/counts và release/provenance, xuất `audit.json`/`coverage.csv`. Coverage frame-weighted theo video/subject/class, không lấy trung bình tỷ lệ từng video; failed members giữ riêng support, zero-row denominator là null. Quality reason counts có thể chồng lấp, không phải tổng rejected frames.

`data/processed/manifest.parquet` giữ nguyên acquisition/source `status=ok/error`. `extraction_status.parquet` là sidecar có source fields/status, extraction/audit status và snapshot/hash provenance; không dùng việc extraction complete để sửa weak labels, folds hay quyền ảnh. Working denominator34 khác acquisition45; thiếu11 nguồn vẫn giữ trong freeze/report. Feature cache này chưa phải calibrated sequences hoặc training-ready classifier data.

## Output
- `models/assets/`: Face Landmarker `.task`, canonical OBJ và metadata URL/SHA256.
- `data/acquisition/`: remote inventory, kế hoạch subset, permissions và download receipts/report.
- `data/processed/manifest.parquet`: dữ liệu video, nhãn nội bộ `0/1/2`, fold provenance, checksum và lỗi.
- `runs/phase0/`: báo cáo môi trường, resolved config và ảnh cửa sổ Qt thử nghiệm.
- `runs/phase2/`: exploration report, manifest CSV và biểu đồ không có ảnh mặt.
- `runs/phase3/`: reports clip/camera/no-face, throughput/drop và optional overlay; screenshot bằng chứng chỉ lấy subject 04 có quyền công bố.
- `runs/phase4/`: EAR three-clip reports, eye-state candidates và actual window capture.
- `runs/phase5/`: bằng chứng lịch sử EAR/MAR, native blank, webcam và cửa sổ; không bị ghi đè bởi Phase6/7.
- `configs/quality_policy_v1.json`: frozen optical evidence và policy, không chứa ảnh mặt.
- `runs/phase6/`: numerical/native pose proof, three-clip prefix, actual windows và webcam.
- `runs/phase7/`: measured quality, actual gated windows/controls, native blank/parity, selected-three EOF Parquet+metadata và resume. Default preview report: `runs/phase7/preview/preview_report.json`.
- `data/processed/raw_features/`:34 verified EOF Parquet+metadata pairs,379,361 rows.
- `data/processed/extraction_status.parquet`: source fields/status giữ nguyên, extraction/audit statuses và frozen snapshot provenance.
- `runs/phase8/`: frozen snapshot, `extraction_report.json`, current `report.json`, `audit.json`, `coverage.csv`, `resume_report.json`, `acceptance_summary.json` và permitted quality spotchecks; interrupted staging đã dọn.
- `data/splits/outer_0..4.json`: full subject/video roles, frozen P0/P1 profiles, train QC và reserved prefixes.
- `data/processed/derived_<hash>/`: typed per-video temporal cache, `window_index.parquet`, `dataset_manifest.json`; exact protocol/provenance and rejection support.
- `runs/phase10/`: clear/rejected native rule replay, camera access và causal/cache parity receipts; physical acceptance riêng.
- `runs/phase11/rf_p0_outer0_seed42_verified/`: trusted joblib, summaries/schema, frozen config/split/selection, predictions/metrics và cold reload proof.
- `runs/phase12/p0_outer0_batch_parity.json`:49400unique train steps, independent scaler/tensor arithmetic và three representative real batches.
- `runs/phase13/lstm_p0_outer0_seed42/`: best bundle, scaler,36epoch history, validation/test metrics và cold reload proof; P1 blocked receipts tách riêng.
- `notebooks/01_dataset_exploration.ipynb`: đọc manifest để team xem thống kê.

Video, ảnh thử nghiệm, environment, runtime logs, profiles và model weights được `.gitignore`, không push lên GitHub. `image_publishable` chỉ là quyền công bố ảnh, không phải quyền tái phân phối dataset. Nhãn UTA áp cho cả video, không chính xác từng frame. Offline Phase9–13 P0 đã kiểm chứng; P1 data-blocked và human webcam acceptance vẫn riêng. Không cần thao tác webcam để chạy offline; UI/audio/realtime cuối cùng chưa triển khai.
