from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from src.contracts import DetectionResult, FramePacket, SystemStatus
from src.realtime.camera_worker import CameraWorker, _LatestFrameSlot


def test_latest_frame_slot_replaces_pending_packet_and_counts_drop():
    slot = _LatestFrameSlot()
    first = object()
    second = object()

    slot.put(first)
    slot.put(second)

    assert slot.get(timeout=0) is second
    assert slot.dropped == 1


def test_worker_uses_injected_detector_and_emits_owned_preview_and_finished():
    app = QCoreApplication.instance() or QCoreApplication([])
    packet = FramePacket(np.zeros((3, 4, 3), dtype=np.uint8), 1_000, 0, "camera:0")

    class Reader:
        stats = {"status": "STOPPED", "capture_released": True}

        def iter_camera(self, camera_index, target_fps, *, width, height, stop_event):
            yield packet

    class Detector:
        closed = False

        def reset_session(self):
            pass

        def process(self, received):
            assert received is packet
            return DetectionResult(None, None, SystemStatus.NO_FACE, {}, "warming")

        def close(self):
            self.closed = True

    detector = Detector()
    worker = CameraWorker(0, lambda: detector, reader_factory=Reader)
    snapshots = []
    statuses = []
    finished = []
    worker.snapshot_ready.connect(snapshots.append)
    worker.status_changed.connect(lambda *args: statuses.append(args))
    worker.finished.connect(lambda: finished.append(True))

    worker.start()

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.preview_size == (4, 3)
    assert snapshot.preview_rgb == bytes(4 * 3 * 3)
    assert snapshot.detection.system_status == SystemStatus.NO_FACE
    assert snapshot.temporal_display["capture_timestamp_ms"] == packet.timestamp_ms
    assert detector.closed
    assert finished == [True]
    assert statuses == [(SystemStatus.NO_FACE, "warming")]
    app.processEvents()


def test_worker_releases_reader_and_detector_when_inference_raises():
    app = QCoreApplication.instance() or QCoreApplication([])
    packet = FramePacket(np.zeros((2, 2, 3), dtype=np.uint8), 2_000, 0, "camera:0")

    class Reader:
        released = False

        def iter_camera(self, camera_index, target_fps, *, width, height, stop_event):
            try:
                yield packet
                while not stop_event.is_set():
                    stop_event.wait(0.001)
            finally:
                self.released = True

    class Detector:
        closed = False

        def reset_session(self):
            pass

        def process(self, received):
            raise RuntimeError("inference failed")

        def close(self):
            self.closed = True

    reader, detector = Reader(), Detector()
    worker = CameraWorker(0, lambda: detector, reader_factory=lambda: reader)
    errors, finished = [], []
    worker.status_changed.connect(lambda *args: errors.append(args))
    worker.finished.connect(lambda: finished.append(True))

    worker.start()

    assert reader.released
    assert detector.closed
    assert errors == [(SystemStatus.ERROR, "inference failed")]
    assert finished == [True]
    app.processEvents()


def test_queued_calibration_uses_explicit_team2_adapter():
    app = QCoreApplication.instance() or QCoreApplication([])
    packet = FramePacket(np.zeros((2, 2, 3), dtype=np.uint8), 3_000, 0, "camera:0")

    class Reader:
        def iter_camera(self, camera_index, target_fps, *, width, height, stop_event):
            yield packet

    class Detector:
        def reset_session(self):
            pass

        def process(self, received):
            return DetectionResult(None, None, SystemStatus.CALIBRATING, {}, "calibrating")

        def close(self):
            pass

    detector = Detector()
    calibration_calls = []
    worker = CameraWorker(
        0,
        lambda: detector,
        reader_factory=Reader,
        calibration_handler=lambda current: calibration_calls.append(current),
    )
    worker.request_calibration()

    worker.start()

    assert calibration_calls == [detector]
    app.processEvents()


def test_repeated_stop_before_start_is_idempotent_and_does_not_open_camera():
    app = QCoreApplication.instance() or QCoreApplication([])
    factory_calls = []
    worker = CameraWorker(
        0,
        lambda: factory_calls.append("detector"),
        reader_factory=lambda: factory_calls.append("reader"),
    )
    finished = []
    worker.finished.connect(lambda: finished.append(True))

    worker.request_stop()
    worker.request_stop()
    worker.start()

    assert factory_calls == []
    assert finished == [True]
    app.processEvents()


@pytest.mark.parametrize("config", [
    {"camera_width": 0},
    {"camera_height": True},
    {"capture_fps": 21},
    {"ui_refresh_hz": 10.1},
    {"ui_refresh_hz": 0},
])
def test_worker_rejects_invalid_capture_and_render_rates(config):
    with pytest.raises(ValueError):
        CameraWorker(0, lambda: None, config=config)
