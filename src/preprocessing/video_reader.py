"""Own capture lifetimes and sample real source time without duplicating frames."""
from __future__ import annotations

import math
from pathlib import Path
from threading import Event
import time
from collections.abc import Iterator
from typing import Any

import cv2

from src.contracts import FramePacket


class VideoReader:
    """One active source at a time; completed/stopped readers may be reused.

    Exhausting a file iterator validates its entire decode. Explicit close or
    generator close validates only the consumed prefix and records STOPPED.
    A camera read failure is always an error, never normal end-of-file.
    """

    def __init__(self) -> None:
        self._capture: Any = None
        self._token: object | None = None
        self._started = 0.0
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict[str, Any]:
        return {
            "decoded_frames": 0,
            "emitted_frames": 0,
            "dropped_frames": 0,
            "timestamp_method": None,
            "duration_s": None,
            "fps_reported": None,
            "wall_seconds": 0.0,
            "status": "IDLE",
            "error": None,
            "capture_released": False,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot, including elapsed time while running."""
        result = dict(self._stats)
        result["dropped_frames"] = result["decoded_frames"] - result["emitted_frames"]
        if self._token is not None:
            result["wall_seconds"] = time.perf_counter() - self._started
        return result

    @staticmethod
    def _validate_target(target_fps: float) -> None:
        if not math.isfinite(target_fps) or not 0 < target_fps <= 20:
            raise ValueError("target_fps must be finite and in (0, 20]")

    def _begin(self) -> object:
        if self._token is not None:
            raise RuntimeError("VideoReader already has an active source iterator")
        token = object()
        self._token = token
        self._stats = self._new_stats()
        self._stats["status"] = "RUNNING"
        self._started = time.perf_counter()
        return token

    def _finish(self, token: object, status: str, error: str | None = None) -> None:
        if self._token is not token:
            return
        capture = self._capture
        self._capture = None
        self._token = None
        self._stats["status"] = status
        self._stats["error"] = error
        self._stats["wall_seconds"] = time.perf_counter() - self._started
        self._stats["dropped_frames"] = self._stats["decoded_frames"] - self._stats["emitted_frames"]
        if capture is not None:
            capture.release()
            self._stats["capture_released"] = True

    @staticmethod
    def _bucket(timestamp_ms: float, origin_ms: float, target_fps: float) -> int:
        # Protect exact grid boundaries from binary floating-point roundoff;
        # this is not a source jitter allowance or a synthetic timestamp.
        return math.floor((timestamp_ms - origin_ms) * target_fps / 1000 + 1e-9)

    def iter_frames(
        self, path: str | Path, target_fps: float = 20, *, constant_fps_verified: bool = False
    ) -> Iterator[FramePacket]:
        self._validate_target(target_fps)
        token = self._begin()
        try:
            source_id = str(Path(path).resolve())
            capture = cv2.VideoCapture(str(path))
            self._capture = capture
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
                raise ValueError("Video FPS and frame count must be finite and positive")
            duration_s = count / fps
            if not math.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("Video duration must be finite and positive")
            self._stats.update(fps_reported=fps, duration_s=duration_s,
                               timestamp_method="CAP_PROP_POS_MSEC")
            duration_ms = duration_s * 1000
            tolerance_ms = 1000 / fps
            if not math.isfinite(duration_ms) or not math.isfinite(tolerance_ms):
                raise ValueError("Video timing metadata exceeds supported range")
            previous_time = None
            previous_integer = None
            origin = None
            last_bucket = -1
            fallback = False
            index = 0
            while self._token is token:
                ok, image = capture.read()
                if not ok or image is None:
                    if index < count:
                        raise RuntimeError(f"Decode failed at frame {index} before declared end {count:g}")
                    self._finish(token, "EOF")
                    return
                self._stats["decoded_frames"] += 1
                timestamp = index * 1000 / fps if fallback else float(capture.get(cv2.CAP_PROP_POS_MSEC))
                valid = math.isfinite(timestamp) and timestamp >= 0
                integer_ms = round(timestamp) if valid else None
                valid = valid and (previous_time is None or timestamp > previous_time)
                valid = valid and (previous_integer is None or integer_ms > previous_integer)
                if not valid:
                    if not constant_fps_verified or fallback:
                        raise RuntimeError(f"Unreliable source timestamp at frame {index}")
                    fallback = True
                    self._stats["timestamp_method"] = "frame_index/fps"
                    timestamp = index * 1000 / fps
                    if not math.isfinite(timestamp) or timestamp < 0:
                        raise RuntimeError(f"Invalid verified constant-FPS timestamp at frame {index}")
                    integer_ms = round(timestamp)
                    if ((previous_time is not None and timestamp <= previous_time)
                            or (previous_integer is not None and integer_ms <= previous_integer)):
                        raise RuntimeError(f"Constant-FPS fallback cannot preserve timestamp order at frame {index}")
                if origin is None:
                    origin = timestamp
                # An out-of-duration timestamp is not a backend-missing timestamp:
                # explicit constant-FPS permission must not hide this mismatch.
                if timestamp - origin > duration_ms + tolerance_ms:
                    raise RuntimeError(f"Timestamp exceeds declared duration at frame {index}")
                previous_time = timestamp
                previous_integer = integer_ms
                bucket = self._bucket(timestamp, origin, target_fps)
                frame_index = index
                index += 1
                if bucket > last_bucket:
                    last_bucket = bucket
                    self._stats["emitted_frames"] += 1
                    yield FramePacket(image, integer_ms, frame_index, source_id)
        except Exception as exc:
            self._finish(token, "ERROR", str(exc))
            raise
        finally:
            self._finish(token, "STOPPED")

    def iter_camera(
        self, camera_index: int = 0, target_fps: float = 20, *, width: int = 640, height: int = 480,
        stop_event: Event | None = None,
    ) -> Iterator[FramePacket]:
        self._validate_target(target_fps)
        token = self._begin()
        try:
            capture = cv2.VideoCapture(camera_index)
            self._capture = capture
            self._stats["timestamp_method"] = "perf_counter_ns"
            clock = time.get_clock_info("perf_counter")
            self._stats["capture_clock"] = {
                "implementation": clock.implementation,
                "resolution_s": clock.resolution,
                "monotonic": clock.monotonic,
            }
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera: {camera_index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            origin_ns = None
            previous_ns = None
            last_bucket = -1
            index = 0
            while self._token is token and not (stop_event is not None and stop_event.is_set()):
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Camera {camera_index} read failed or disconnected")
                captured_ns = time.perf_counter_ns()
                self._stats["decoded_frames"] += 1
                if previous_ns is not None and captured_ns <= previous_ns:
                    raise RuntimeError("Camera capture clock did not increase")
                previous_ns = captured_ns
                if origin_ns is None:
                    origin_ns = captured_ns
                bucket = self._bucket((captured_ns - origin_ns) / 1_000_000, 0, target_fps)
                frame_index = index
                index += 1
                if bucket > last_bucket:
                    last_bucket = bucket
                    self._stats["emitted_frames"] += 1
                    yield FramePacket(image, captured_ns // 1_000_000, frame_index, f"camera:{camera_index}")
        except Exception as exc:
            self._finish(token, "ERROR", str(exc))
            raise
        finally:
            self._finish(token, "STOPPED")

    def close(self) -> None:
        """Release the active capture without declaring a prefix fully validated."""
        if self._token is not None:
            self._finish(self._token, "STOPPED")

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
