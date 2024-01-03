import json
import os
from typing import Any, Tuple

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QDoubleSpinBox,
    QSpinBox,
)


class Settings(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(1000, 1000)

        with open(os.path.join("cache", "settings.json")) as f:
            self.settings = json.load(f)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        scroll_area.setWidget(self.scroll_widget)

        title = "Settings"
        title_label = QLabel(title)
        title_label.setStyleSheet("font: bold 16px;")
        self.scroll_layout.addWidget(title_label)

        # Section height
        self._add_title("Gaussian kernel")
        self._add_sub_title(
            "The GAUSSIAN_KERNEL is the kernel used to blur the image before applying the canny edge detection (The values could throw an error if they aren't odd)."
        )
        self._add_number_input("A", "GAUSSIAN_KERNEL", 0, (0, 50))
        self._add_number_input("B", "GAUSSIAN_KERNEL", 1, (0, 50))

        # Section Plate aspect ratio
        self._add_title("Plate aspect ratio")
        self._add_sub_title(
            "The APECT_RATIO_MIN_MAX is the range of the aspect ratio of the plate: the wider the range is, the more potential plates will be detected."
        )
        self._add_number_input(
            "Aspect ratio min",
            "ASPECT_RATIO_MIN_MAX",
            0,
            (0, 2),
            step=0.1,
            element=QDoubleSpinBox,
        )
        self._add_number_input(
            "Aspect ratio max",
            "ASPECT_RATIO_MIN_MAX",
            1,
            (0, 2),
            step=0.1,
            element=QDoubleSpinBox,
        )

        # Section height
        self._add_title("Plate height")
        self._add_sub_title(
            "The HEIGHT_MIN_MAX is the range of the height of the plate: the wider the range is, the more potential plates will be detected."
        )
        self._add_number_input("Height min", "HEIGHT_MIN_MAX", 0, (0, 500))
        self._add_number_input("Height max", "HEIGHT_MIN_MAX", 1, (0, 500))

        # Section width
        self._add_title("Plate width")
        self._add_sub_title(
            "The WIDTH_MIN_MAX is the range of the width of the plate: the wider the range is, the more potential plates will be detected."
        )
        self._add_number_input("Width min", "WIDTH_MIN_MAX", 0, (0, 500))
        self._add_number_input("Width max", "WIDTH_MIN_MAX", 1, (0, 500))

        # Section min size connected component
        self._add_title("Min size connected component")
        self._add_sub_title(
            "The MIN_SIZE_CONNECTED_COMPONENT is the minimum size of the connected component: the smaller the value is, the more potential plates will be detected."
        )
        self._add_number_input("Min size", "MIN_SIZE_CONNECTED_COMPONENT", 0, (0, 50))

        # Split crop
        self._add_title("Split crop")
        self._add_sub_title("The SPLIT_CROP is the number of pixels to crop on each side of the plate, used to split the plate in 2 parts. (upper, down)")
        self._add_number_input("Split crop upper", "SPLIT_CROP", 0, (0, 100))

        # Section crop
        self._add_title("Crop")
        self._add_sub_title(
            "The CROP is the number of pixels to crop on each side of the plate, on every side. (top, left, bottom, right)"
        )
        self._add_number_input("Crop top", "CROP", 0, (0, 100))
        self._add_number_input("Crop left", "CROP", 1, (0, 100))
        self._add_number_input("Crop bottom", "CROP", 2, (0, 100))
        self._add_number_input("Crop right", "CROP", 3, (0, 100))

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

    def _add_title(self, title: str):
        title_label = QLabel(title)
        title_label.setContentsMargins(0, 20, 0, 0)
        title_label.setStyleSheet("font: bold 12px;")
        self.scroll_layout.addWidget(title_label)

    def _add_sub_title(self, sub_title: str):
        sub_title_label = QLabel(sub_title)
        sub_title_label.setStyleSheet("font: 10px;")
        self.scroll_layout.addWidget(sub_title_label)

    def _add_number_input(
        self,
        placeholder: str,
        key: int,
        index: int,
        range: Tuple[float, float],
        step: float = 1,
        element=QSpinBox,
    ):
        placeholder_label = QLabel(placeholder)
        placeholder_label.setStyleSheet("font: 10px;")
        self.scroll_layout.addWidget(placeholder_label)
        widget = element()
        widget.setSingleStep(step)
        widget.setRange(*range)
        widget.setValue(self.settings[key][index])
        # using lambda to pass arguments to the callback
        widget.valueChanged.connect(
            lambda value, key=key, index=index: self.update_settings(key, index, value)
        )
        self.scroll_layout.addWidget(widget)

    def update_settings(self, key: str, index: int, value: Any):
        """
        Update the settings and save them in the cache/settings.json file
        """
        self.settings[key][index] = value
        with open("cache/settings.json", "w") as f:
            json.dump(self.settings, f)


if __name__ == "__main__":
    app = QApplication([])
    # open Settings
    settings = Settings()
    settings.show()
    app.exec()
