"""Qt worker that consumes the shared camera reader through a latest-frame slot.

The detector is injected by Team 2 and must implement the published
``process(packet)``, ``reset_session()``, and ``close()`` lifecycle. This module
owns capture orchestration and alert decisions; it never touches widgets or
plays audio.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from numbers import Integral, Real
from threading import Condition, Event, Lock, Thread
from time import perf_counter, perf_counter_ns
from typing import Any

import cv2
from PySide6.QtCore import QObject, Signal

from src.alerts.manager import AlertManager
from src.contracts import AlertDecision, DetectionResult, SystemStatus, UiSnapshot
from src.preprocessing.video_reader import VideoReader


class _LatestFrameSlot:
    """A capacity-one handoff: a newly captured frame replaces an old one."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._packet = None
        self.dropped = 0

    def put(self, packet: Any) -> None:
        with self._condition:
            if self._packet is not None:
                self.dropped += 1
            self._packet = packet
            self._condition.notify()

    def get(self, timeout: float) -> Any | None:
        with self._condition:
            if self._packet is None:
                self._condition.wait(timeout)
            packet, self._packet = self._packet, None
            return packet

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class CameraWorker(QObject):
    """Capture and process camera frames off the Qt main thread.

    Move this QObject to a QThread and connect ``QThread.started`` to
    ``start``. ``request_stop`` and ``set_muted`` are safe to call from the UI
    thread while ``start`` is running.
    """

    snapshot_ready = Signal(object)
    status_changed = Signal(object, str)
    finished = Signal()

    def __init__(
        self,
        camera_index: int,
        detector_factory: Callable[[], Any],
        *,
        config: Mapping[str, Any] | None = None,
        reader_factory: Callable[[], VideoReader] = VideoReader,
        alert_manager: AlertManager | None = None,
        calibration_handler: Callable[[Any], None] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(camera_index, bool) or not isinstance(camera_index, int) or camera_index < 0:
            raise ValueError("camera_index must be a nonnegative integer")
        if not callable(detector_factory) or not callable(reader_factory):
            raise TypeError("detector_factory and reader_factory must be callable")
        values = dict(config or {})
        for key, value in (("camera_width", values.get("camera_width", 640)),
                           ("camera_height", values.get("camera_height", 480))):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        capture_fps = values.get("capture_fps", 20)
        if (isinstance(capture_fps, bool) or not isinstance(capture_fps, Real)
                or not math.isfinite(capture_fps) or not 0 < capture_fps <= 20):
            raise ValueError("capture_fps must be finite and in (0, 20]")
        refresh_hz = values.get("ui_refresh_hz", 10)
        if (isinstance(refresh_hz, bool) or not isinstance(refresh_hz, Real)
                or not math.isfinite(refresh_hz) or not 0 < refresh_hz <= 10):
            raise ValueError("ui_refresh_hz must be finite and in (0, 10]")
        self._camera_index = camera_index
        self._detector_factory = detector_factory
        self._reader_factory = reader_factory
        self._alert_manager = alert_manager or AlertManager(values)
        self._calibration_handler = calibration_handler
        self._width = values.get("camera_width", 640)
        self._height = values.get("camera_height", 480)
        self._capture_fps = float(capture_fps)
        self._publish_interval_s = 1.0 / float(refresh_hz)
        self._stop_event = Event()
        self._command_lock = Lock()
        self._calibration_pending = False
        self._slot = _LatestFrameSlot()
        self._capture_done = Event()
        self._capture_error: str | None = None
        self._reader: VideoReader | None = None
        self._capture_thread: Thread | None = None
        self._started = False

    def set_muted(self, muted: bool) -> None:
        """Thread-safe mute command; warning state remains owned by worker."""
        self._alert_manager.set_muted(muted)

    def request_calibration(self) -> None:
        """Queue calibration for the detector's integration adapter."""
        with self._command_lock:
            self._calibration_pending = True
        self._slot.wake()

    def request_stop(self) -> None:
        """Request interruption without relying on Qt's blocked event loop."""
        self._stop_event.set()
        self._slot.wake()

    def _capture(self) -> None:
        assert self._reader is not None
        try:
            for packet in self._reader.iter_camera(
                self._camera_index,
                self._capture_fps,
                width=self._width,
                height=self._height,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    break
                self._slot.put(packet)
        except Exception as exc:
            self._capture_error = str(exc)
        finally:
            self._capture_done.set()
            self._slot.wake()

    def _take_calibration_request(self, detector: Any) -> None:
        with self._command_lock:
            requested, self._calibration_pending = self._calibration_pending, False
        if not requested:
            return
        if self._calibration_handler is None:
            self.status_changed.emit(
                SystemStatus.UNRELIABLE,
                "Calibration chưa được nối: cần handler từ detector/realtime core của Team 2.",
            )
            return
        self._calibration_handler(detector)

    @staticmethod
    def _preview_rgb(packet: Any) -> tuple[bytes, tuple[int, int]]:
        rgb = cv2.cvtColor(packet.image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        return rgb.tobytes(), (width, height)

    def _make_snapshot(
        self, packet: Any, detection: DetectionResult, alert: AlertDecision,
        fps: float, started: float,
    ) -> UiSnapshot:
        now_ms = perf_counter_ns() // 1_000_000
        preview, size = self._preview_rgb(packet)
        prediction = detection.smoothed_prediction
        prediction_age = None if prediction is None else max(0, now_ms - prediction.timestamp_ms)
        return UiSnapshot(
            preview_rgb=preview,
            preview_size=size,
            feature_sample=None,
            temporal_display={
                "capture_timestamp_ms": packet.timestamp_ms,
                "capture_age_ms": max(0, now_ms - packet.timestamp_ms),
                "worker_dropped_frames": self._slot.dropped,
            },
            detection=detection,
            alert=alert,
            fps=fps,
            latency_ms=(perf_counter() - started) * 1000,
            prediction_age_ms=prediction_age,
        )

    def start(self) -> None:
        """Run the session; connect QThread.started to this blocking slot."""
        if self._started:
            self.status_changed.emit(SystemStatus.ERROR, "CameraWorker session already started")
            return
        self._started = True
        if self._stop_event.is_set():
            self.finished.emit()
            return
        detector = None
        last_status: tuple[SystemStatus, str] | None = None
        last_publish = 0.0
        processed = 0
        fps_started = perf_counter()
        fps = 0.0
        try:
            detector = self._detector_factory()
            detector.reset_session()
            self._alert_manager.reset()
            self._reader = self._reader_factory()
            self._capture_thread = Thread(target=self._capture, name="camera-capture", daemon=False)
            self._capture_thread.start()
            while not self._stop_event.is_set():
                self._take_calibration_request(detector)
                packet = self._slot.get(timeout=0.05)
                if packet is None:
                    if self._capture_done.is_set():
                        if self._capture_error:
                            self.status_changed.emit(SystemStatus.ERROR, self._capture_error)
                        break
                    continue

                started = perf_counter()
                detection = detector.process(packet)
                if not isinstance(detection, DetectionResult):
                    raise TypeError("detector.process(packet) must return DetectionResult")
                decision_time = max(perf_counter_ns() // 1_000_000, packet.timestamp_ms)
                alert = self._alert_manager.update(detection, decision_time)
                processed += 1
                elapsed = perf_counter() - fps_started
                if elapsed >= 1.0:
                    fps = processed / elapsed
                    processed = 0
                    fps_started = perf_counter()
                now = perf_counter()
                if now - last_publish >= self._publish_interval_s:
                    self.snapshot_ready.emit(self._make_snapshot(packet, detection, alert, fps, started))
                    last_publish = now

                current_status = (detection.system_status, detection.calibration_status)
                if current_status != last_status:
                    self.status_changed.emit(*current_status)
                    last_status = current_status
        except Exception as exc:
            self.status_changed.emit(SystemStatus.ERROR, str(exc))
        finally:
            self.request_stop()
            if self._capture_thread is not None:
                self._capture_thread.join()
            if detector is not None:
                try:
                    detector.close()
                except Exception as exc:
                    self.status_changed.emit(SystemStatus.ERROR, f"Detector close failed: {exc}")
            self._alert_manager.reset()
            self.finished.emit()
