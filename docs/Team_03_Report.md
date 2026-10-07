# Báo cáo Team 3 — Alerts và CameraWorker

**Ngày cập nhật:** 2026-10-07  
**Trạng thái:** Phần code độc lập đã triển khai; chưa nghiệm thu tích hợp/hardware.  
**Phạm vi:** Phase19 (alert policy) và phần CameraWorker của Phase20 theo [kế hoạch Team 3](Team_03_Alerts_UI.md).

## Tóm tắt

Team 3 đã triển khai state machine cảnh báo không chặn, CameraWorker Qt dùng latest-frame slot, asset âm báo tổng hợp có thông tin nguồn/license, và kiểm thử hồi quy cho alerts/worker/capture. Kiểm thử scoped gần nhất đạt **58 passed**; `pip check` báo không có dependency hỏng.

Kết quả này **không phải nghiệm thu Phase19/Phase20/Phase21**. Chưa có detector/calibration realtime của Team 2 nối vào worker; chưa có UI/audio playback của Team 4 nối vào ứng dụng; camera và loa thật chưa được smoke test. Snapshot hiện chưa có FeatureSample và temporal feature metrics từ detector.

## File theo từng thay đổi

| File | Thay đổi | Ghi chú bàn giao |
|---|---|---|
| `src/alerts/__init__.py` | Tạo package `alerts`. | Không có side effect khi import. |
| `src/alerts/manager.py` | Tạo `AlertManager`: ngưỡng Level 1/2, dwell/exit hysteresis, ưu tiên Drowsy, strong escalation, cooldown, mute, stale/unknown và no-face technical message. `update()` chỉ trả `AlertDecision`/audio command, không phát âm thanh hoặc sleep. | Dùng đúng `DetectionResult`/`Prediction`; timestamp phải cùng monotonic epoch với prediction. Audio command hiện là `drowsy` và `strong`. |
| `tests/test_alerts.py` | Thêm fake-clock regression cho entry/exit, oscillation, unknown/stale, escalation, cooldown, mute, no-face, validation/reset và các mốc ngay trước/đúng/sau ngưỡng được kiểm tra. | Không dùng camera/model thật. Lệnh trong phần Kiểm chứng. |
| `src/realtime/__init__.py` | Tạo package `realtime`. |  |
| `src/realtime/camera_worker.py` | Tạo QObject worker: detector được tiêm qua factory; capture producer dùng slot capacity 1; frame mới thay frame chưa xử lý; xử lý detector ngoài UI thread; chuyển preview BGR→RGB thành bytes sở hữu; phát `UiSnapshot` tối đa 10 Hz; chuyển trạng thái và alert; stop/mute thread-safe; cleanup capture/detector khi kết thúc/lỗi. | Consumer tạo `QThread`, move worker vào thread và nối `QThread.started` tới `start()`. Worker không thao tác widget hoặc phát audio. Calibration yêu cầu handler được tiêm. |
| `src/preprocessing/video_reader.py` | Mở rộng `iter_camera(..., stop_event=None)`: vòng đọc capture dừng theo Event nếu được yêu cầu; đường đọc hiện có giữ nguyên khi không truyền Event. | Capture vẫn do `VideoReader` tạo và release; worker không tạo implementation camera thứ hai. |
| `tests/test_camera_worker.py` | Thêm kiểm thử slot latest-frame/drop count, snapshot bytes/size, lifecycle reset/close, cleanup khi inference lỗi, calibration adapter, stop idempotent trước Start và validation cấu hình/rates. | Detector và reader trong các test là fake. Không chứng minh camera vật lý. |
| `scripts/generate_warning_asset.py` | Thêm script Python standard library để tái tạo âm báo hai nốt, không dùng sample bên thứ ba. | Chạy từ root: `.\.venv\Scripts\python.exe scripts/generate_warning_asset.py`. Script tái tạo asset và metadata/hash. |
| `models/assets/warning.wav` | Thêm âm báo mono 16-bit PCM, 24 kHz, dài 0.44 giây. | Âm báo tổng hợp, không phải giọng nói. Team 4 cần chọn cách phát cho command `drowsy`/`strong` và nghe thử loa thật. |
| `models/assets/warning.wav.json` | Ghi nguồn tổng hợp, license CC0-1.0, thông số và SHA-256 `02059f6ef5d096bd897081e8154b604a87c0d6472f89d0578d2032c27995e336`. | Sidecar provenance đi cùng asset khi sao chép/đóng gói. |
| `docs/Team_03_Handoff.md` | Ghi API Team 2/4 cần nối, command test và giới hạn nghiệm thu hiện tại. | Tài liệu phối hợp ngắn. |
| `Team_03_handoff.zip` | Gói snapshot 23 file để chuyển thủ công, gồm source, test, asset, contract/config tham chiếu và tài liệu. | File ZIP bị `.gitignore` loại khỏi commit; nếu dùng Git, đưa các file nguồn vào branch bằng danh sách staging tường minh, không commit archive. |

### Files được tham chiếu nhưng không thay đổi

- `src/contracts.py`: giữ nguyên records/enums hiện hữu.
- `configs/realtime.yaml`: thresholds/durations cần cho `AlertManager` đã có sẵn; Team 3 không sửa file config dùng chung.
- Các tài liệu kiến trúc/module/team khác: dùng làm chuẩn tích hợp, không sửa hợp đồng chung trong phần này.

## Hợp đồng tích hợp CameraWorker

### Yêu cầu từ Team 2

`detector_factory` truyền vào `CameraWorker` phải tạo đối tượng có lifecycle đã thống nhất:

1. `reset_session()` — worker gọi một lần đầu session.
2. `process(packet: FramePacket) -> DetectionResult` — gọi cho từng frame được xử lý.
3. `close()` — luôn gọi trong cleanup, kể cả khi inference lỗi.

Timestamp camera và timestamp trong prediction phải dùng cùng monotonic epoch (`perf_counter`). Calibration hiện được chuyển bằng callback tùy chọn `calibration_handler(detector)`; callback chạy trong worker thread. Team 2 cần chỉ định entry point calibration thật. Detector cũng cần cung cấp cách đọc FeatureSample/temporal display cho snapshot, hoặc hai team phải thống nhất adapter read-only tương đương; không tạo pipeline feature thứ hai và không tự sửa `src/contracts.py`.

### Yêu cầu từ Team 4

1. Tạo `CameraWorker`, đặt vào `QThread`, nối `QThread.started` đến `worker.start()`.
2. Nhận `snapshot_ready(UiSnapshot)` và `status_changed(SystemStatus, str)`; preview là RGB bytes đã copy và có `preview_size`.
3. Gọi `request_stop()` để dừng; nối `finished` để thread owner gọi quit/wait theo lifecycle của UI. Không dùng `terminate()`.
4. Gọi `set_muted(bool)` và `request_calibration()`; truyền callback calibration đã thống nhất với Team 2 khi khởi tạo worker.
5. Phát `AlertDecision.audio_command` và load `models/assets/warning.wav` trên Qt main thread. Worker chỉ phát command, không gọi Qt audio.

## Cách chuyển và phối hợp

Workspace hiện tại không có Git metadata, nên snapshot ZIP có thể được gửi qua Teams/Drive. Nếu nhóm đã có remote/shared Git repo, hãy nhập các thay đổi vào project chung và review diff trước khi tạo PR.

- **Gửi Team 2:** báo API detector/lifecycle ở trên; đề nghị cung cấp calibration entry point và API read-only cho FeatureSample/temporal metrics. Nhờ xác nhận epoch timestamp và trạng thái/error semantics.
- **Gửi Team 4:** đề nghị review QThread/signal/lifecycle wiring; nối mute/stop/calibration; xác nhận nơi load WAV và mapping hai audio command. Sau đó chạy test loa thật cùng Team 3.
- **Khi giải nén ZIP:** giữ cấu trúc thư mục. `src/contracts.py`, `configs/realtime.yaml` và `src/preprocessing/video_reader.py` là file dùng chung; so sánh thay đổi trước khi thay thế bản đang có.
- **Không đánh dấu PASS từ test fake:** chỉ ghi smoke thật sau khi có detector, UI, camera/loa và lưu evidence có timestamp/provenance theo Team 3 checklist.

## Kiểm chứng đã chạy

Từ root project, suite scoped cuối cùng:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_camera_worker.py tests/test_alerts.py tests/test_video_reader.py -q --tb=short
```

Kết quả: **58 passed**. Môi trường:

```powershell
.venv\Scripts\python.exe -m pip check
```

Kết quả: **No broken requirements found.** Các test worker dùng fake reader/detector; test này xác nhận logic phần mềm, không xác nhận camera, UI thread native, âm lượng/chất lượng loa hay giải phóng thiết bị vật lý.

## Còn chờ nghiệm thu/phối hợp

- Nối detector realtime, calibration và FeatureSample/temporal display của Team 2.
- Nối UI, signal/thread shutdown và audio playback trên Qt main thread của Team 4.
- Camera smoke: Start/Stop 5 lần, rút camera, đóng app trong lúc xử lý; ghi drop/cadence/latency và cleanup evidence.
- Nghe `warning.wav` thật cùng Team 4; xác nhận command/episode/escalation và mute.
- Lưu run evidence vào `runs/<id>/` sau smoke; hiện chưa có hardware run evidence.

Vì vậy trạng thái nên báo là **IMPLEMENTED — CHƯA NGHIỆM THU**, không phải DONE.
