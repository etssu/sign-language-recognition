import sys
from datetime import datetime

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

        self.setWindowTitle("Sign Language Recognition")
        self.resize(1200, 800)

        # Load Camera
        self.camera_widget = CameraWidget()
        self.camera_widget.frame_ready.connect(self.process_frame)

        # Title
        title = QLabel("Sign Language Recognition")
        title.setObjectName("title")

        # Dropdown list
        combobox = QComboBox(self)
        combobox.addItems(['KNN Static [ODPORUCANE]', 'RF Static', 'SVM Static', 'KNN Dynamic', 'RF Dynamic', 'SVM Dynamic', 'DTW DYNAMIC [ODPORUCANE]'])

        # Current gesture
        self.gesture_label = QLabel("—")
        self.gesture_label.setObjectName("gesture")

        self.confidence_label = QLabel("Confidence: —")

        # Camera buttons
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

        # Current gesture section
        current_box = QVBoxLayout()
        current_box.addWidget(QLabel("Current Gesture"))
        current_box.addWidget(self.gesture_label)
        current_box.addWidget(self.confidence_label)

        # Left side: camera and recognition
        left_layout = QVBoxLayout()
        left_layout.addWidget(title)
        left_layout.addWidget(combobox)
        left_layout.addWidget(self.camera_widget, 3)
        left_layout.addLayout(current_box)
        left_layout.addLayout(camera_buttons)

        # Right side: history panel
        history_panel = QFrame()
        history_panel.setObjectName("historyPanel")

        history_layout = QVBoxLayout(history_panel)

        history_title = QLabel("Recognition History")
        history_title.setObjectName("historyTitle")

        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")

        history_layout.addWidget(history_title)
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
        prediction = self.recognition.process_frame(frame)

        if prediction is None:
            return

        print("Static prediction:", prediction)

    def add_history_item(self, gesture, confidence):
        time = datetime.now().strftime("%H:%M:%S")

        item = QListWidgetItem(
            f"{gesture}\n{time}                         {confidence:.1f}%"
        )

        self.history_list.insertItem(0, item)

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
