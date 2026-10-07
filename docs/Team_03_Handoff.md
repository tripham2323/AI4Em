# Team 3 handoff — Alerts and CameraWorker

## Included implementation

- `src/alerts/manager.py`: warning policy; returns `AlertDecision.audio_command` (`drowsy`, `strong`) without playing audio.
- `src/realtime/camera_worker.py`: Qt worker, capacity-one latest-frame slot, owned RGB preview bytes, status/snapshot signals, thread-safe stop/mute, and calibration adapter hook.
- `src/preprocessing/video_reader.py`: shared camera iterator accepts an optional stop event; capture remains owned/released by the reader thread.
- `models/assets/warning.wav` and `.wav.json`: original generated two-tone chime, CC0-1.0 provenance and SHA-256; rebuild with `scripts/generate_warning_asset.py`.
- Regression tests: `tests/test_alerts.py`, `tests/test_camera_worker.py`, and `tests/test_video_reader.py`.

## Team 2: detector handoff needed

`CameraWorker` receives a `detector_factory` and expects the documented lifecycle:

- `reset_session()` once when a worker session starts.
- `process(FramePacket) -> DetectionResult` for each consumed frame.
- `close()` during cleanup, including inference errors.

The detector timestamps and camera timestamps must share the `perf_counter` monotonic epoch. Calibration is passed through the worker's optional `calibration_handler(detector)` callback; agree on the concrete Team 2 calibration entry point before connecting it. Current snapshots leave `feature_sample=None` and `temporal_display` without feature metrics because the detector contract does not yet expose those values. Agree on a read-only source for them; do not create a second feature pipeline or change `src/contracts.py` without the shared owner decision.

## Team 4: UI/audio handoff

Move the worker to a `QThread`, connect `QThread.started` to `worker.start`, consume `snapshot_ready(UiSnapshot)` and `status_changed(SystemStatus, str)`, and connect worker completion to thread shutdown. Send Stop through `request_stop()`, mute through `set_muted(bool)`, and calibration through `request_calibration()`. Supply the agreed calibration handler when constructing the worker. Execute `AlertDecision.audio_command` and load/play `models/assets/warning.wav` on the Qt main thread. No widget or audio call belongs in the worker.

## Verification and remaining acceptance

Run the scoped suite from the project root:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_camera_worker.py tests/test_alerts.py tests/test_video_reader.py -q --tb=short
```

Latest local result: **58 passed**; `pip check` reports no broken requirements. These are code/fake-adapter checks. Real-camera Start/Stop, disconnect/close, feature snapshot integration, and speaker playback still need Team 2/4 and hardware acceptance; do not report those as PASS.
