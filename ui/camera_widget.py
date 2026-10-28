import cv2

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout

class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.video_label = QLabel("Нажми «Запустить камеру»")
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)

        self.camera = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if self.camera is not None:
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.video_label.setText("Не удалось открыть камеру")
            self.camera.release()
            self.camera = None
            return

        self.timer.start(30)

    def update_frame(self):
        if self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        image = QImage(
                frame.data,
                width,
                height,
                frame.strides[0],
                QImage.Format.Format_RGB888
            ).copy()

        pixmap = QPixmap.fromImage(image)

        self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    def stop_camera(self):
        self.timer.stop()

        if self.camera is not None:
            self.camera.release()
            self.camera = None
