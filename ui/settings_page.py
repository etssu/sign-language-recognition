
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QComboBox,
    QSlider,
    QCheckBox,
)
from PySide6.QtCore import Signal, Qt


class SettingsPage(QWidget):

    back_requested = Signal()

    def __init__(self):
        super().__init__()

        # Header
        title = QLabel("Settings")
        title.setObjectName("title")

        subtitle = QLabel("Customize your experience")
        subtitle.setObjectName("settingsSubtitle")

        back_button = QPushButton("← Back")
        back_button.setObjectName("backButton")
        back_button.clicked.connect(self.back_requested.emit)

        header_layout = QHBoxLayout()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(back_button)

        # Appearance section
        appearance_title = QLabel("Appearance")
        appearance_title.setObjectName("settingsSectionTitle")

        appearance_card = QFrame()
        appearance_card.setObjectName("settingsCard")

        theme_label = QLabel("Theme")
        theme_description = QLabel("Choose application appearance")

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "System"])

        theme_layout = QHBoxLayout()
        theme_layout.addWidget(theme_label)
        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_combo)

        appearance_layout = QVBoxLayout(appearance_card)
        appearance_layout.addLayout(theme_layout)
        appearance_layout.addWidget(theme_description)

        # Recognition section
        recognition_title = QLabel("Recognition")
        recognition_title.setObjectName("settingsSectionTitle")

        recognition_card = QFrame()
        recognition_card.setObjectName("settingsCard")

        confidence_label = QLabel("Confidence threshold")
        confidence_description = QLabel(
            "Minimum confidence required for a prediction"
        )

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(80)

        self.confidence_value = QLabel("80%")

        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_value.setText(f"{value}%")
        )

        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value)

        model_label = QLabel("Recognition model")

        self.model_combo = QComboBox()
        self.model_combo.addItems(["KNN", "SVM", "Random Forest"])

        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addStretch()
        model_layout.addWidget(self.model_combo)

        recognition_layout = QVBoxLayout(recognition_card)
        recognition_layout.addWidget(confidence_label)
        recognition_layout.addWidget(confidence_description)
        recognition_layout.addLayout(confidence_layout)
        recognition_layout.addSpacing(12)
        recognition_layout.addLayout(model_layout)

        # Accessibility section
        accessibility_title = QLabel("Accessibility")
        accessibility_title.setObjectName("settingsSectionTitle")

        accessibility_card = QFrame()
        accessibility_card.setObjectName("settingsCard")

        self.tts_checkbox = QCheckBox("Enable text-to-speech")
        tts_description = QLabel(
            "Read recognized gestures aloud"
        )

        accessibility_layout = QVBoxLayout(accessibility_card)
        accessibility_layout.addWidget(self.tts_checkbox)
        accessibility_layout.addWidget(tts_description)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(12)

        layout.addLayout(header_layout)
        layout.addWidget(subtitle)
        layout.addSpacing(15)

        layout.addWidget(appearance_title)
        layout.addWidget(appearance_card)

        layout.addWidget(recognition_title)
        layout.addWidget(recognition_card)

        layout.addWidget(accessibility_title)
        layout.addWidget(accessibility_card)

        layout.addStretch()
