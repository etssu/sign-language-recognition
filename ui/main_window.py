import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QListWidget,
    QListWidgetItem, QComboBox,
)

from ui.camera_widget import CameraWidget
from utils.model_loader import load_model
from utils.recognition import Recognition


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Load model
        self.knn_static_model = load_model("knn_static_model")
        self.knn_dynamic_model = load_model("knn_dynamic_model")

        self.recognition = Recognition(self.knn_static_model, self.knn_dynamic_model)
        # History recognition settings
        self.confidence_threshold = 80
        self.required_frames = 10

        self.current_prediction = None
        self.current_confidence = None

        self.last_detected_gesture = None
        self.detected_frames = 0
        self.last_saved_gesture = None

        self.setWindowTitle("Sign Language Recognition")
        self.resize(1200, 800)

        # Load Camera
        self.camera_widget = CameraWidget()
        self.camera_widget.frame_ready.connect(self.process_frame)
        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setObjectName("confidenceLabel")

        # Title
        title = QLabel("Sign Language Recognition")
        title.setObjectName("title")

        # Dropdown list
        combobox = QComboBox(self)
        combobox.addItems(['KNN Static [ODPORUCANE]', 'RF Static', 'SVM Static', 'KNN Dynamic', 'RF Dynamic', 'SVM Dynamic', 'DTW DYNAMIC [ODPORUCANE]'])

        # Camera buttons
        self.start_button = QPushButton("Start Camera")
        self.start_button.setObjectName("cameraButton")
        self.start_button.clicked.connect(self.start_camera)


        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("cameraButton")
        self.stop_button.clicked.connect(self.stop_camera)

        self.stop_button.setEnabled(False)

        camera_buttons = QHBoxLayout()
        camera_buttons.addWidget(self.start_button)
        camera_buttons.addWidget(self.stop_button)

        # Left side: camera and recognition
        left_layout = QVBoxLayout()
        left_layout.addWidget(title)
        left_layout.addWidget(combobox)
        left_layout.addWidget(self.camera_widget, 3)
        left_layout.addWidget(self.confidence_label)
        left_layout.addLayout(camera_buttons)

        # Right side: history panel
        history_panel = QFrame()
        history_panel.setObjectName("historyPanel")

        history_layout = QVBoxLayout(history_panel)

        history_title = QLabel("Recognition History")
        history_title.setObjectName("historyTitle")

        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.setObjectName("clearHistoryButton")
        self.clear_history_button.clicked.connect(self.clear_history)
        self.clear_history_button.setEnabled(False)

        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")

        history_layout.addWidget(history_title)
        history_layout.addWidget(self.clear_history_button)
        history_layout.addWidget(self.history_list)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(separator)
        main_layout.addWidget(history_panel)

        history_panel.setFixedWidth(280)

        self.load_styles()

    def process_frame(self, frame):
        prediction, confidence = self.recognition.process_frame(frame)

        self.camera_widget.current_gesture = prediction

        self.current_prediction = prediction
        self.current_confidence = confidence

        if confidence is not None:
            self.confidence_label.setText(
                f"Confidence: {confidence:.1f}%"
            )
        else:
            self.confidence_label.setText("Confidence: —")

        # No hand detected
        if prediction is None:
            self.last_detected_gesture = None
            self.detected_frames = 0
            self.last_saved_gesture = None
            return

        # Confidence is too low
        if confidence < self.confidence_threshold:
            self.last_detected_gesture = None
            self.detected_frames = 0
            return

        # Count consecutive frames
        if prediction == self.last_detected_gesture:
            self.detected_frames += 1
        else:
            self.last_detected_gesture = prediction
            self.detected_frames = 1

        # Save confirmed gesture
        if (self.detected_frames >= self.required_frames and prediction != self.last_saved_gesture):
            self.add_history_item(prediction, confidence)
            self.last_saved_gesture = prediction

    def add_history_item(self, gesture, confidence):
        time = datetime.now().strftime("%H:%M:%S")

        item = QListWidgetItem(
            f"{gesture}\n{time}                         {confidence:.1f}%"
        )

        self.history_list.addItem(item)
        self.history_list.scrollToBottom()
        self.clear_history_button.setEnabled(True)

    def load_styles(self):
        try:
            styles_path = (
                    Path(__file__).resolve().parent
                    / "styles.qss"
            )

            if styles_path.exists():
                with open(styles_path, encoding="utf-8") as file:
                    self.setStyleSheet(file.read())
        except FileNotFoundError:
            pass

    def start_camera(self):
        self.camera_widget.start_camera()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        self.camera_widget.stop_camera()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.camera_widget.stop_camera()
        self.recognition.close()
        event.accept()

    def clear_history(self):
        self.history_list.clear()
        self.clear_history_button.setEnabled(False)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())
