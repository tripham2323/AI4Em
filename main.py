import sys
import argparse
import yaml
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from src.ui.main_window import MainWindow
from src.contracts import SystemStatus

def validate_config(config: dict, skip_model_check=False):
    required_keys = ["calibration_mode", "model_path", "ui_refresh_hz"]
    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing required config key: {k}")
    
    mode = config["calibration_mode"]
    if mode not in ["P0", "P1"]:
        raise ValueError(f"Invalid calibration_mode: {mode}. Must be P0 or P1.")
        
    if not skip_model_check:
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {model_path}. Please check configuration.")

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection UI")
    parser.add_argument("--config", type=str, default="configs/realtime.yaml", help="Path to config file")
    parser.add_argument("--fixture", action="store_true", help="Run with fixture data instead of real camera worker")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        validate_config(config, skip_model_check=args.fixture)
        
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{str(e)}")
        return 1

    window = MainWindow(config)
    
    if args.fixture:
        from PySide6.QtCore import QTimer
        from src.contracts import UiSnapshot, DetectionResult, AlertDecision, FeatureSample, DriverState, Prediction
        import numpy as np
        
        def emit_fixture():
            # Mock fixture snapshot for testing UI layout and updates
            feat = FeatureSample(
                timestamp_ms=1000, frame_index=30, source_id="fixture",
                ear_left=0.25, ear_right=0.26, ear_mean=0.255,
                mar=0.1, pitch=5.0, yaw=-2.0, roll=1.5,
                face_detected=True, left_eye_valid=True, right_eye_valid=True,
                mouth_valid=True, pose_valid=True, reprojection_error_norm=2.5
            )
            
            pred = Prediction(
                timestamp_ms=1000,
                probabilities=np.array([0.9, 0.08, 0.02], dtype=np.float32),
                class_id=DriverState.ALERT,
                valid=True,
                reason="",
                model_id="fixture_model"
            )
            
            snapshot = UiSnapshot(
                preview_rgb=None,
                preview_size=(640, 480),
                feature_sample=feat,
                temporal_display={"perclos": 0.15, "coverage": 0.95},
                detection=DetectionResult(
                    raw_prediction=pred,
                    smoothed_prediction=pred,
                    system_status=SystemStatus.READY,
                    quality={"calibration_progress": 0.8, "calibration_msg": "Calibrating (Fixture)..."},
                    calibration_status=config["calibration_mode"]
                ),
                alert=AlertDecision(level=0, strong=False, audio_command=None, message="", timestamp_ms=1000),
                fps=30.0,
                latency_ms=45.0,
                prediction_age_ms=10.0
            )
            window.render(snapshot)
            
        timer = QTimer(window)
        timer.timeout.connect(emit_fixture)
        
        def on_start():
            timer.start(int(1000 / config.get("ui_refresh_hz", 10)))
            
        window.request_start.connect(on_start)
        window.request_stop.connect(timer.stop)
        
    else:
        try:
            from src.realtime.camera_worker import CameraWorker
            worker = CameraWorker(config)
            
            window.request_start.connect(worker.start)
            window.request_stop.connect(worker.request_stop)
            window.request_calibrate.connect(worker.request_calibration)
            
            # Need to start worker thread safely if implemented as QThread
            # or just connect the signals if worker manages its own thread
            worker.snapshot_ready.connect(window.render)
            
        except ImportError as e:
            QMessageBox.critical(
                None, "Integration Error",
                "CameraWorker from Team 3 (Phase 19) is not yet implemented or cannot be imported.\n"
                f"Details: {e}\n\nUse --fixture to test UI layout."
            )
            return 1
            
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
