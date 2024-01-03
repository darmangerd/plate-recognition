from typing import List, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea


class ImageDetail(QWidget):
    def __init__(
        self,
        images: List[Tuple[Tuple[cv2.Mat], str]],
        detected_texts: List[str] = [],
        accuracies: List[float] = [],
        title: str = "Image detail",
        closeEventParent=None,
    ):
        super().__init__()

        self.closeEventParent = closeEventParent

        self.resize(800, 1000)
        self.setWindowTitle(title)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        scroll_area.setWidget(self.scroll_widget)

        # Add images to the layout
        for images, title in images:
            # Add title
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font: bold 12px;")
            self.scroll_layout.addWidget(title_label)

            if len(images) <= 0:
                none_label = QLabel(
                    "Image is not defined, the plate may not have been detected properly"
                )
                none_label.setAlignment(Qt.AlignCenter)
                self.scroll_layout.addWidget(none_label)

            # Add images
            for image in images:
                if image is None:
                    none_label = QLabel(
                        "Image is not defined, the plate may not have been detected properly"
                    )
                    none_label.setAlignment(Qt.AlignCenter)
                    self.scroll_layout.addWidget(none_label)
                    continue

                # image = img.resize_to_match_width(image, self.width() - 100)
                bytes_per_line = image.shape[1] * 3
                pixmap = QPixmap.fromImage(
                    QImage(
                        np.ascontiguousarray(image.data),
                        image.shape[1],
                        image.shape[0],
                        bytes_per_line,
                        QImage.Format_RGB888,
                    )
                )
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setPixmap(pixmap)
                self.scroll_layout.addWidget(label)

        if len(detected_texts) <= 0:
            text = f"No text detected"
            text_label = QLabel(text)
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setStyleSheet("font: bold 14px;")
            self.scroll_layout.addWidget(text_label)

        # Add text at the end
        for i, text in enumerate(detected_texts):
            acc_text = ""
            if i < len(accuracies):
                acc_text = f"(accuracy: {accuracies[i] * 100:.2f}%)"

            text = f"Potential plate N° {i} {acc_text} \n{text}"
            text_label = QLabel(text)
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setStyleSheet("font: bold 14px;")
            self.scroll_layout.addWidget(text_label)

        # Set the main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

    def close(self) -> bool:
        return super().close()

    def closeEvent(self, event) -> None:
        if self.closeEvent is not None:
            self.closeEventParent()
        event.accept()
