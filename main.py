"""Desktop entry point for fixture or production realtime inference."""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

from src.contracts import (
    AlertDecision,
    DetectionResult,
    DriverState,
    FeatureSample,
    Prediction,
    SystemStatus,
    UiSnapshot,
)
from src.realtime.camera_worker import CameraWorker
from src.realtime.factory import (
    build_detector,
    load_runtime_config,
    restart_calibration,
    validate_runtime_artifacts,
)
from src.ui.main_window import MainWindow


class RuntimeController(QObject):
    """Own one worker thread per Start/Stop session and allow clean restart."""

    def __init__(self, config: dict[str, Any], window: MainWindow) -> None:
        super().__init__(window)
        self._config = config
        self._window = window
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None

    @Slot()
    def start(self) -> None:
        if self._thread is not None:
            self._window.render_status(SystemStatus.WARMING_UP, "Realtime session is already active")
            return
        thread = QThread(self)
        worker = CameraWorker(
            int(self._config["camera_index"]),
            lambda: build_detector(self._config),
            config=self._config,
            calibration_handler=restart_calibration,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.start)
        worker.snapshot_ready.connect(self._window.render)
        worker.status_changed.connect(self._window.render_status)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot()
    def stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    @Slot()
    def calibrate(self) -> None:
        if self._worker is None:
            self._window.render_status(SystemStatus.UNRELIABLE, "Start the camera before calibration")
            return
        self._worker.request_calibration()

    @Slot(bool)
    def set_muted(self, muted: bool) -> None:
        if self._worker is not None:
            self._worker.set_muted(muted)

    @Slot()
    def _thread_finished(self) -> None:
        thread = self._thread
        self._worker = None
        self._thread = None
        if thread is not None:
            thread.deleteLater()
        self._window.worker_finished()

    def shutdown(self, timeout_ms: int = 10_000) -> bool:
        thread = self._thread
        if thread is None:
            return True
        self.stop()
        return bool(thread.wait(timeout_ms))


def _install_fixture(window: MainWindow, config: dict[str, Any]) -> QTimer:
    timer = QTimer(window)

    def emit_fixture() -> None:
        feature = FeatureSample(
            timestamp_ms=1000,
            frame_index=30,
            source_id="fixture",
            ear_left=0.25,
            ear_right=0.26,
            ear_mean=0.255,
            mar=0.1,
            pitch=5.0,
            yaw=-2.0,
            roll=1.5,
            face_detected=True,
            left_eye_valid=True,
            right_eye_valid=True,
            mouth_valid=True,
            pose_valid=True,
            reprojection_error_norm=0.01,
        )
        prediction = Prediction(
            timestamp_ms=1000,
            probabilities=np.array([0.9, 0.08, 0.02], dtype=np.float32),
            class_id=DriverState.ALERT,
            valid=True,
            reason="",
            model_id="fixture_model",
        )
        window.render(
            UiSnapshot(
                preview_rgb=None,
                preview_size=(640, 480),
                feature_sample=feature,
                temporal_display={"perclos": 0.15, "coverage": 0.95},
                detection=DetectionResult(
                    raw_prediction=prediction,
                    smoothed_prediction=prediction,
                    system_status=SystemStatus.READY,
                    quality={"calibration_progress": 1.0, "calibration_msg": "Fixture ready"},
                    calibration_status=config["calibration_mode"],
                ),
                alert=AlertDecision(level=0, strong=False, audio_command=None, message="", timestamp_ms=1000),
                fps=30.0,
                latency_ms=45.0,
                prediction_age_ms=10.0,
            )
        )

    timer.timeout.connect(emit_fixture)
    window.request_start.connect(lambda: timer.start(round(1000 / config.get("ui_refresh_hz", 10))))
    window.request_stop.connect(timer.stop)
    return timer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection UI")
    parser.add_argument("--config", default="configs/realtime.yaml", help="Path to realtime YAML")
    parser.add_argument("--fixture", action="store_true", help="Run the UI without camera/model inference")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        config = load_runtime_config(args.config)
        validate_runtime_artifacts(config, require_model=not args.fixture)
    except Exception as exc:  # startup boundary: show one actionable error
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{exc}")
        return 1

    window = MainWindow(config)
    controller: RuntimeController | None = None
    fixture_timer: QTimer | None = None
    if args.fixture:
        fixture_timer = _install_fixture(window, config)
    else:
        controller = RuntimeController(config, window)
        window.request_start.connect(controller.start)
        window.request_stop.connect(controller.stop)
        window.request_calibrate.connect(controller.calibrate)
        window.request_mute.connect(controller.set_muted)

    window.show()
    exit_code = app.exec()
    if controller is not None and not controller.shutdown():
        return 2
    # Keep the timer referenced for the full event-loop lifetime.
    _ = fixture_timer
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
