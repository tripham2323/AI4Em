from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QGridLayout, QProgressBar,
    QTextEdit, QSizePolicy, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, Slot, Signal, QUrl, QTime, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QSoundEffect

from src.config import PROJECT_ROOT
from src.contracts import UiSnapshot, SystemStatus

# Modern Dark Theme Stylesheet
STYLE_SHEET = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    color: #E0E0E0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 14px;
    background-color: #1E1E1E;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #64B5F6;
}
QPushButton {
    background-color: #2D2D2D;
    color: #FFFFFF;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #404040;
    border: 1px solid #64B5F6;
}
QPushButton:pressed {
    background-color: #64B5F6;
    color: #121212;
}
QLabel {
    background-color: transparent;
}
QProgressBar {
    border: 1px solid #333333;
    border-radius: 4px;
    text-align: center;
    color: white;
    background-color: #2D2D2D;
}
QProgressBar::chunk {
    background-color: #64B5F6;
    border-radius: 3px;
}
QTextEdit {
    background-color: #1A1A1A;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 4px;
}
QSlider::groove:horizontal {
    border: 1px solid #333;
    height: 6px;
    background: #2D2D2D;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #64B5F6;
    border: 1px solid #64B5F6;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #64B5F6;
    border-radius: 4px;
    background-color: #1A1A1A;
}
QCheckBox::indicator:checked {
    background-color: #64B5F6;
}
"""

class MainWindow(QMainWindow):
    # Signals for communicating with the worker/backend
    request_start = Signal()
    request_stop = Signal()
    request_calibrate = Signal()
    request_mute = Signal(bool)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.is_muted = config.get("muted", False)
        
        self.sound_effect = QSoundEffect(self)
        self.sound_effect.setVolume(1.0) # Default max volume
        self.current_audio = None
        
        # Timer setup for Uptime tracking
        self.uptime_seconds = 0
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self.update_uptime)
        
        self.setStyleSheet(STYLE_SHEET)
        self.setup_ui()
        self.update_mute_button()

    def setup_ui(self):
        self.setWindowTitle("AI4Em - Driver Drowsiness Detection System")
        self.resize(1150, 780)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # ==========================================
        # LEFT PANEL (Controls + Video + Logs)
        # ==========================================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # 1. Main Controls
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ START")
        self.btn_start.setStyleSheet("color: #81C784;")
        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.setStyleSheet("color: #E57373;")
        self.btn_calibrate = QPushButton("⚙ CALIBRATE")
        self.btn_mute = QPushButton("🔊 MUTE")
        
        self.btn_start.clicked.connect(self.start_session)
        self.btn_stop.clicked.connect(self.stop_session)
        self.btn_calibrate.clicked.connect(self.request_calibrate.emit)
        self.btn_mute.clicked.connect(self.toggle_mute)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_mute)
        left_panel.addLayout(btn_layout)
        
        # 2. Extra Controls (Volume & Always on Top)
        extra_ctrl_layout = QHBoxLayout()
        
        self.chk_always_top = QCheckBox("📌 Always on Top")
        self.chk_always_top.stateChanged.connect(self.toggle_always_on_top)
        
        lbl_vol = QLabel("🎵 Vol:")
        lbl_vol.setFixedWidth(50)
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(100)
        self.slider_vol.valueChanged.connect(self.change_volume)
        
        extra_ctrl_layout.addWidget(self.chk_always_top)
        extra_ctrl_layout.addStretch()
        extra_ctrl_layout.addWidget(lbl_vol)
        extra_ctrl_layout.addWidget(self.slider_vol)
        left_panel.addLayout(extra_ctrl_layout)
        
        # 3. Camera Preview
        self.preview_label = QLabel("CAMERA OFF")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background-color: #000000; border: 2px solid #333333; border-radius: 8px;")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_panel.addWidget(self.preview_label, stretch=3)
        
        # 4. Event Log
        log_group = QGroupBox("Event Logs")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_panel.addWidget(log_group, stretch=1)
        
        main_layout.addLayout(left_panel, stretch=2)
        
        # ==========================================
        # RIGHT PANEL (Dashboard & Stats)
        # ==========================================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        # 1. System Status
        sys_group = QGroupBox("System Status")
        sys_layout = QGridLayout()
        sys_layout.setVerticalSpacing(8)
        self.lbl_status = QLabel("● IDLE")
        self.lbl_status.setStyleSheet("color: #9E9E9E; font-weight: bold; font-size: 14px;")
        self.lbl_uptime = QLabel("Uptime: 00:00:00")
        self.lbl_uptime.setStyleSheet("color: #64B5F6; font-weight: bold;")
        self.lbl_mode = QLabel("Mode: N/A")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_latency = QLabel("Latency: --")
        
        sys_layout.addWidget(self.lbl_status, 0, 0)
        sys_layout.addWidget(self.lbl_uptime, 0, 1)
        sys_layout.addWidget(self.lbl_mode, 1, 0)
        sys_layout.addWidget(self.lbl_fps, 1, 1)
        sys_layout.addWidget(self.lbl_latency, 2, 0, 1, 2)
        sys_group.setLayout(sys_layout)
        right_panel.addWidget(sys_group)
        
        # 2. Advanced Features
        feat_group = QGroupBox("Biometric Features")
        feat_layout = QGridLayout()
        feat_layout.setVerticalSpacing(10)
        
        self.lbl_ear = QLabel("EAR (L/R/Mean): N/A")
        self.lbl_mar = QLabel("MAR: N/A")
        self.lbl_pose = QLabel("Head Pose: N/A")
        self.lbl_perclos = QLabel("PERCLOS: N/A")
        self.lbl_coverage = QLabel("Coverage: N/A")
        
        feat_layout.addWidget(self.lbl_ear, 0, 0)
        feat_layout.addWidget(self.lbl_mar, 1, 0)
        feat_layout.addWidget(self.lbl_pose, 2, 0)
        feat_layout.addWidget(self.lbl_perclos, 3, 0)
        feat_layout.addWidget(self.lbl_coverage, 4, 0)
        feat_group.setLayout(feat_layout)
        right_panel.addWidget(feat_group)
        
        # 3. Probabilities (Visual Progress Bars)
        prob_group = QGroupBox("AI Prediction")
        prob_layout = QVBoxLayout()
        
        self.lbl_class = QLabel("Current State: N/A")
        self.lbl_class.setStyleSheet("font-size: 15px; font-weight: bold; color: #64B5F6;")
        prob_layout.addWidget(self.lbl_class)
        
        self.bar_alert = self.create_prob_bar("Alert (Normal)", "#81C784")
        self.bar_low_vig = self.create_prob_bar("Low Vigilance", "#FFF176")
        self.bar_drowsy = self.create_prob_bar("Drowsy", "#E57373")
        
        prob_layout.addLayout(self.bar_alert['layout'])
        prob_layout.addLayout(self.bar_low_vig['layout'])
        prob_layout.addLayout(self.bar_drowsy['layout'])
        
        prob_group.setLayout(prob_layout)
        right_panel.addWidget(prob_group)
        
        # 4. Critical Warning Area
        self.lbl_warning = QLabel("SYSTEM READY")
        self.lbl_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setMinimumHeight(60)
        self.lbl_warning.setStyleSheet("""
            background-color: #2D2D2D; 
            color: #9E9E9E; 
            border-radius: 8px; 
            font-size: 16px; 
            font-weight: bold;
        """)
        right_panel.addWidget(self.lbl_warning)
        
        # 5. Calibration Progress
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout()
        self.lbl_cal_info = QLabel("Ready for calibration. Ensure you are looking straight.")
        self.lbl_cal_info.setWordWrap(True)
        self.lbl_cal_info.setStyleSheet("color: #BDBDBD;")
        self.progress_cal = QProgressBar()
        self.progress_cal.setRange(0, 100)
        self.progress_cal.setValue(0)
        cal_layout.addWidget(self.lbl_cal_info)
        cal_layout.addWidget(self.progress_cal)
        cal_group.setLayout(cal_layout)
        right_panel.addWidget(cal_group)
        
        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=1)

    def create_prob_bar(self, label_text, color):
        layout = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(100)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid #333; border-radius: 3px; background-color: #222; text-align: center; color: white; height: 12px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}
        """)
        layout.addWidget(lbl)
        layout.addWidget(bar)
        return {'layout': layout, 'bar': bar}

    def toggle_always_on_top(self, state):
        # We need to preserve current window flags, and just toggle the TopHint
        if self.chk_always_top.isChecked():
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            self.log_event("Enabled 'Always on Top'.")
        else:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
            self.log_event("Disabled 'Always on Top'.")
        self.show() # Requirement in Qt to apply window flag changes

    def change_volume(self, value):
        # Slider is 0-100, QSoundEffect volume is 0.0-1.0
        self.sound_effect.setVolume(value / 100.0)

    def update_uptime(self):
        self.uptime_seconds += 1
        hours, rem = divmod(self.uptime_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        self.lbl_uptime.setText(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def start_session(self):
        self.lbl_status.setText("● STARTING...")
        self.lbl_status.setStyleSheet("color: #FFB74D; font-weight: bold; font-size: 14px;")
        
        # Reset and start timer
        self.uptime_seconds = 0
        self.lbl_uptime.setText("Uptime: 00:00:00")
        self.session_timer.start(1000) # Tick every 1 second
        
        self.log_event("System started.")
        self.request_start.emit()
        
    def stop_session(self):
        self.session_timer.stop()
        self.sound_effect.stop()
        self.lbl_status.setText("● STOPPED")
        self.lbl_status.setStyleSheet("color: #E57373; font-weight: bold; font-size: 14px;")
        self.log_event(f"System stopped. Total Uptime: {self.lbl_uptime.text().split(' ')[1]}")
        self.request_stop.emit()
        
    def toggle_mute(self):
        self.is_muted = not self.is_muted
        self.update_mute_button()
        if self.is_muted:
            self.sound_effect.stop()
            self.log_event("Audio muted by user.")
        else:
            self.log_event("Audio unmuted.")
        self.request_mute.emit(self.is_muted)
        
    def update_mute_button(self):
        if self.is_muted:
            self.btn_mute.setText("🔇 UNMUTE")
            self.btn_mute.setStyleSheet("color: #E57373;")
        else:
            self.btn_mute.setText("🔊 MUTE")
            self.btn_mute.setStyleSheet("")
            
    def log_event(self, message: str):
        timestamp = QTime.currentTime().toString("HH:mm:ss")
        self.log_text.append(f"<span style='color: #64B5F6;'>[{timestamp}]</span> {message}")

    @Slot(object, str)
    def render_status(self, status: SystemStatus, message: str) -> None:
        value = status.value if isinstance(status, SystemStatus) else str(status)
        color = "#81C784" if value == "READY" else "#FFF176" if value in {"WARMING_UP", "CALIBRATING"} else "#E57373"
        self.lbl_status.setText(f"● {value}")
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
        if message:
            self.log_event(message)

    @Slot()
    def worker_finished(self) -> None:
        self.session_timer.stop()
        self.sound_effect.stop()
        self.lbl_status.setText("● STOPPED")
        self.lbl_status.setStyleSheet("color: #E57373; font-weight: bold; font-size: 14px;")
        self.log_event("Realtime worker stopped and released its resources.")
        
    @Slot(UiSnapshot)
    def render(self, snapshot: UiSnapshot):
        # 1. Image Preview
        if snapshot.preview_rgb is not None:
            w, h = snapshot.preview_size
            qimg = QImage(snapshot.preview_rgb, w, h, w * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.preview_label.clear()
            self.preview_label.setText("NO CAMERA FEED")
            
        # 2. System Status
        status_val = snapshot.detection.system_status.value
        color = "#81C784" if status_val == "READY" else "#FFF176" if status_val in ["WARMING_UP", "CALIBRATING"] else "#E57373"
        self.lbl_status.setText(f"● {status_val}")
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
        
        self.lbl_mode.setText(f"Mode: {snapshot.detection.calibration_status}")
        self.lbl_fps.setText(f"FPS: {snapshot.fps:.1f}")
        
        age = snapshot.prediction_age_ms
        self.lbl_latency.setText(f"Age (ms): {age:.0f}" if age is not None else "Age (ms): N/A")
        
        # 3. Biometric Features
        fs = snapshot.feature_sample
        if fs and fs.face_detected:
            left = f"{fs.ear_left:.3f}" if fs.left_eye_valid else "--"
            right = f"{fs.ear_right:.3f}" if fs.right_eye_valid else "--"
            mean = f"{fs.ear_mean:.3f}" if (fs.left_eye_valid and fs.right_eye_valid) else "--"
            self.lbl_ear.setText(f"EAR (L/R/Mean): <b>{left}</b> / <b>{right}</b> / <b>{mean}</b>")
            
            mar = f"{fs.mar:.3f}" if fs.mouth_valid else "--"
            self.lbl_mar.setText(f"MAR: <b>{mar}</b>")
            
            p = f"{fs.pitch:.1f}" if fs.pose_valid else "--"
            y = f"{fs.yaw:.1f}" if fs.pose_valid else "--"
            r_ = f"{fs.roll:.1f}" if fs.pose_valid else "--"
            self.lbl_pose.setText(f"Head Pose: <b>P: {p} | Y: {y} | R: {r_}</b>")
        else:
            self.lbl_ear.setText("EAR (L/R/Mean): N/A")
            self.lbl_mar.setText("MAR: N/A")
            self.lbl_pose.setText("Head Pose: N/A")
            
        perclos = snapshot.temporal_display.get("perclos")
        self.lbl_perclos.setText(f"PERCLOS: <b>{perclos*100:.1f}%</b>" if perclos is not None else "PERCLOS: N/A")
        
        cov = snapshot.temporal_display.get("coverage")
        self.lbl_coverage.setText(f"Coverage: <b>{cov*100:.1f}%</b>" if cov is not None else "Coverage: N/A")
        
        # 4. Predictions & Alerts
        pred = snapshot.detection.smoothed_prediction
        if pred and pred.valid:
            probs = pred.probabilities
            self.bar_alert['bar'].setValue(int(probs[0] * 100))
            self.bar_low_vig['bar'].setValue(int(probs[1] * 100))
            self.bar_drowsy['bar'].setValue(int(probs[2] * 100))
            
            class_name = pred.class_id.name if pred.class_id is not None else 'N/A'
            self.lbl_class.setText(f"Current State: {class_name}")
            
        else:
            self.bar_alert['bar'].setValue(0)
            self.bar_low_vig['bar'].setValue(0)
            self.bar_drowsy['bar'].setValue(0)
            reason = pred.reason if pred else "No Prediction"
            self.lbl_class.setText(f"Current State: N/A ({reason})")
            
        alert = snapshot.alert
        if alert.level > 0:
            warning_text = f"LEVEL {alert.level}: {alert.message.upper()}"
            if self.lbl_warning.text() != warning_text:
                self.log_event(f"<span style='color: #E57373;'>WARNING - {warning_text}</span>")
                
            self.lbl_warning.setText(warning_text)
            if alert.level == 1:
                self.lbl_warning.setStyleSheet("background-color: #F57F17; color: white; border-radius: 8px; font-size: 18px; font-weight: bold;")
            else:
                self.lbl_warning.setStyleSheet("background-color: #D32F2F; color: white; border-radius: 8px; font-size: 20px; font-weight: bold;")
        else:
            self.lbl_warning.setText("SYSTEM NORMAL")
            self.lbl_warning.setStyleSheet("background-color: #1E1E1E; color: #81C784; border-radius: 8px; font-size: 16px; font-weight: bold; border: 1px solid #333;")
            
        # 5. Audio Playback
        if alert.audio_command in {"drowsy", "strong"} and not self.is_muted:
            if self.current_audio != alert.audio_command or not self.sound_effect.isPlaying():
                self.current_audio = alert.audio_command
                url = QUrl.fromLocalFile(str(PROJECT_ROOT / "models" / "assets" / "warning.wav"))
                self.sound_effect.setSource(url)
                self.sound_effect.play()
                self.log_event(f"Playing alert sound: {alert.audio_command}")
                
        # 6. Calibration
        qual = snapshot.detection.quality
        if "calibration_progress" in qual:
            prog = qual["calibration_progress"]
            self.progress_cal.setValue(int(prog * 100))
        if "calibration_msg" in qual:
            self.lbl_cal_info.setText(qual["calibration_msg"])

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.stop_session()
        event.accept()
