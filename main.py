import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
)

from ui.camera_widget import CameraWidget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sign Language Recognition")
        self.resize(1000, 800)

        self.camera_widget = CameraWidget()

        title = QLabel("Sign Language Recognition")
        title.setObjectName("title")

        self.gesture_label = QLabel("—")
        self.gesture_label.setObjectName("gesture")

        self.confidence_label = QLabel("Confidence: —")

        self.history_label = QLabel(
            "Recognition history will appear here"
        )
        self.history_label.setWordWrap(True)

        self.start_button = QPushButton("Start Camera")
        self.start_button.setObjectName("cameraButton")
        self.start_button.clicked.connect(
            self.camera_widget.start_camera
        )

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("cameraButton")
        self.stop_button.clicked.connect(
            self.camera_widget.stop_camera
        )

        camera_buttons = QHBoxLayout()
        camera_buttons.addWidget(self.start_button)
        camera_buttons.addWidget(self.stop_button)

        results = QHBoxLayout()

        current_box = QVBoxLayout()
        current_box.addWidget(QLabel("Current Gesture"))
        current_box.addWidget(self.gesture_label)
        current_box.addWidget(self.confidence_label)

        history_box = QVBoxLayout()
        history_box.addWidget(QLabel("History"))
        history_box.addWidget(self.history_label)

        results.addLayout(current_box, 1)
        results.addLayout(history_box, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.camera_widget, 3)
        layout.addLayout(camera_buttons)
        layout.addLayout(results, 1)

        self.load_styles()

    def load_styles(self):
        try:
            with open("ui/styles.qss", encoding="utf-8") as file:
                self.setStyleSheet(file.read())
        except FileNotFoundError:
            pass

    def closeEvent(self, event):
        self.camera_widget.stop_camera()
        event.accept()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())
