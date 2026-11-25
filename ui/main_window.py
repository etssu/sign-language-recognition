import sys
from datetime import datetime
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtCore import Qt
from ui.settings_page import SettingsPage

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QComboBox,
    QToolButton,
    QScrollArea,
)
from PySide6.QtWidgets import QSizePolicy

from ui.camera_widget import CameraWidget
from utils.model_loader import load_model
from utils.recognition import Recognition


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Pages
        self.pages = QStackedWidget()
        self.settings_page = SettingsPage()

        # Load models
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
        settings_button = QPushButton()
        settings_button.setIcon(QIcon("img/settings.svg"))
        settings_button.setFixedSize(40, 40)
        settings_button.setObjectName("settingsButton")
        settings_button.clicked.connect(self.open_settings)

        header_layout = QHBoxLayout()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(settings_button)


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
        left_layout.addLayout(header_layout)
        left_layout.addWidget(self.camera_widget, 3)
        left_layout.addWidget(self.confidence_label)
        left_layout.addLayout(camera_buttons)

        # Right side: history panel
        self.selected_history_card = None
        history_panel = QFrame()
        history_panel.setObjectName("historyPanel")

        history_layout = QVBoxLayout(history_panel)

        history_title = QLabel("Recognition History")
        history_title.setObjectName("historyTitle")

        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.setObjectName("clearHistoryButton")
        self.clear_history_button.clicked.connect(self.clear_history)
        self.clear_history_button.setEnabled(False)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_layout.setSpacing(10)

        scroll_area.setWidget(self.history_container)

        history_layout.addWidget(history_title)
        history_layout.addWidget(self.clear_history_button)
        history_layout.addWidget(scroll_area)

        history_panel.setFixedWidth(280)
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        # Main layout
        # Main page
        self.main_page = QWidget()

        main_layout = QHBoxLayout(self.main_page)
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(separator)
        main_layout.addWidget(history_panel)

        history_panel.setFixedWidth(280)

        # Pages
        self.pages.addWidget(self.main_page)
        self.pages.addWidget(self.settings_page)
        self.settings_page.back_requested.connect(self.open_main)

        layout = QVBoxLayout(self)
        layout.addWidget(self.pages)

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
        if self.detected_frames >= self.required_frames and prediction != self.last_saved_gesture:
            self.add_history_item(prediction, confidence)
            self.last_saved_gesture = prediction

    def add_history_item(self, gesture, confidence):
        card = QFrame()
        card.setObjectName("historyCard")
        card.setProperty("selected", False)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(5)

        button = QToolButton()
        button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        button.setMinimumHeight(30)
        button.setMaximumHeight(36)
        button.setText(gesture)
        button.setObjectName("historyItemButton")
        button.setCheckable(True)
        button.setArrowType(Qt.ArrowType.RightArrow)
        button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        button.setSizePolicy(
            button.sizePolicy().Policy.Expanding,
            button.sizePolicy().Policy.Fixed
        )

        confidence_label = QLabel(f"{confidence:.1f}%")
        confidence_label.setObjectName("historyConfidence")

        header.addWidget(button, 1)
        header.addWidget(confidence_label)

        # Details
        details = QLabel(
            f"Meaning: Information about {gesture}.\n\n"
            f"Usage: How and when this gesture is used."
        )

        details.setObjectName("historyDetails")
        details.setWordWrap(True)
        details.setAlignment(Qt.AlignmentFlag.AlignTop)
        details.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum
        )
        details.hide()

        def toggle_details(checked):
            if self.selected_history_card is not None:
                old_card = self.selected_history_card

                if old_card != card:
                    old_button = old_card.findChild(
                        QToolButton,
                        "historyItemButton"
                    )

                    old_details = old_card.findChild(
                        QLabel,
                        "historyDetails"
                    )

                    if old_button:
                        old_button.blockSignals(True)
                        old_button.setChecked(False)
                        old_button.setArrowType(Qt.ArrowType.RightArrow)
                        old_button.blockSignals(False)

                    if old_details:
                        old_details.hide()

                    old_card.setProperty("selected", False)
                    old_card.style().unpolish(old_card)
                    old_card.style().polish(old_card)

            if checked:
                self.selected_history_card = card
                card.setProperty("selected", True)
                details.show()
                button.setArrowType(Qt.ArrowType.DownArrow)
            else:
                self.selected_history_card = None
                card.setProperty("selected", False)
                details.hide()
                button.setArrowType(Qt.ArrowType.RightArrow)

            card.style().unpolish(card)
            card.style().polish(card)

        button.toggled.connect(toggle_details)

        card_layout.addLayout(header)
        card_layout.addWidget(details)

        self.history_layout.addWidget(card)
        self.clear_history_button.setEnabled(True)

    def show_gesture_info(self, index):
        if index < 0:
            return

        gesture = self.history_combo.itemData(index)

        self.gesture_name.setText(gesture)
        self.gesture_meaning.setText(
            "Meaning: Information about this gesture."
        )
        self.gesture_usage.setText(
            "Usage: How and when this gesture is used."
        )

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
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

        self.selected_history_card = None
        self.clear_history_button.setEnabled(False)

        self.last_saved_gesture = None
        self.last_detected_gesture = None
        self.detected_frames = 0

    def open_settings(self):
        self.pages.setCurrentWidget(self.settings_page)

    def open_main(self):
        self.pages.setCurrentWidget(self.main_page)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())
