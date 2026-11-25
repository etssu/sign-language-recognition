from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from PySide6.QtCore import Signal


class SettingsPage(QWidget):

    back_requested = Signal()

    def __init__(self):
        super().__init__()

        title = QLabel("Settings")
        title.setObjectName("title")

        back_button = QPushButton("Back")
        back_button.clicked.connect(self.back_requested.emit)


        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(back_button)