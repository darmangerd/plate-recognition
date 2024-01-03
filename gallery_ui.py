from typing import List, Optional
import json
import os
import cv2
import glob
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import traceback
from image_detail_ui import ImageDetail
from settings_ui import Settings
from lib.plates import Plate
import lib.img as img

CACHE_FILE = os.path.join("cache", "cache.json")


class Gallery(QWidget):
    def __init__(self, image_folder, n=9, square_size=3):
        super().__init__()

        # Initialize class variables
        self.image_folder = image_folder
        self.n = n
        self.square_size = square_size
        self.current_page = 0
        self.total_pages = 0

        # Get a list of all image file names
        self.image_files = glob.glob(os.path.join(self.image_folder, "*.jpg"))

        # Initialize UI
        self.layout = QGridLayout(self)
        self.setLayout(self.layout)

        # Update the UI
        self.update_ui()

    def update_ui(self):
        # Clear the current UI
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            self.layout.removeWidget(widget)
            widget.deleteLater()

        # Calculate the total number of pages
        self.total_pages = (len(self.image_files) + self.n - 1) // self.n

        # Display images for the current page
        for i in range(self.n):
            index = self.current_page * self.n + i
            if index < len(self.image_files):
                group = QGroupBox()
                image_file = self.image_files[index]
                image = cv2.imread(image_file)
                pixmap = QPixmap.fromImage(
                    QImage(
                        image.data, image.shape[1], image.shape[0], QImage.Format_RGB888
                    ).rgbSwapped()
                )
                label = QLabel()
                label.setPixmap(pixmap.scaledToWidth(200))
                label.mousePressEvent = (
                    lambda event, image_file=image_file: self.image_clicked(image_file)
                )

                group_layout = QVBoxLayout()
                group_layout.addWidget(label)
                group.setLayout(group_layout)

                caption_title = os.path.basename(image_file)
                caption_label = QLabel(caption_title)
                caption_label.setAlignment(Qt.AlignCenter)

                image_cache = self.load_image_cache(image_file)
                if image_cache is not None:
                    if len(image_cache) > 0:
                        if image_cache[0][1] >= 0:
                            caption_cache = (
                                f"{image_cache[0][0]}\n({image_cache[0][1] * 100:.2f}%)"
                            )
                        else:
                            caption_cache = f"{image_cache[0][0]}\n(No annotation found)"
                        caption_cache_label = QLabel(
                            caption_title + "\n" + caption_cache
                        )
                        caption_cache_label.setAlignment(Qt.AlignCenter)
                        group_layout.addWidget(caption_cache_label)
                    else:
                        caption_cache_label = QLabel(
                            caption_title + "\n" + "No text detected"
                        )
                        caption_cache_label.setAlignment(Qt.AlignCenter)
                        group_layout.addWidget(caption_cache_label)
                else:
                    caption_cache_label = QLabel(caption_title + "\n" + "No cache")
                    caption_cache_label.setAlignment(Qt.AlignCenter)
                    group_layout.addWidget(caption_cache_label)

                self.layout.addWidget(
                    group, i // self.square_size, i % self.square_size
                )

        # Add pagination buttons
        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(lambda: self.previous_page())

        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self.next_page())

        page_label = QLabel(
            "Page {} of {}".format(self.current_page + 1, self.total_pages),
            alignment=Qt.AlignCenter,
        )

        # Next row add a settings button, which takes 3 columns
        settings_button = QPushButton("Settings")
        def open_settings():
            settings = Settings()
            settings.show()
        settings_button.clicked.connect(open_settings)

        self.layout.addWidget(previous_button, self.n // self.square_size, 0)
        self.layout.addWidget(page_label, self.n // self.square_size, 1)
        self.layout.addWidget(next_button, self.n // self.square_size, 2)
        self.layout.addWidget(settings_button, self.n // self.square_size + 1, 0, 1, 3)

    def image_clicked(self, image_file: str):
        try:
            plate_detector = Plate(image_file)

            images = [
                ([plate_detector.original_image], f"Original image {image_file}"),
                ([plate_detector.original_image_cropped], "Cropped image"),
                ([plate_detector.canny_image], "Canny image"),
                ([plate_detector.connected_components_image], "Connected components"),
                ([plate_detector.image_with_all_regions], "All regions"),
                ([plate_detector.image_with_potential_regions], "Potential regions"),
                ([i for i in plate_detector.potential_plates], "Potential plates"),
                (
                    [i for i in plate_detector.potential_binarized_plates],
                    "Potential plates binarized",
                ),
                (
                    [
                        i
                        for l, u, i in plate_detector.splited_potential_binarized_plates
                    ],
                    "Plates with split line break",
                ),
                (
                    [
                        img.resize_to_match_width(plate_detector.annotated_image, 500)
                        if plate_detector.annotated_image is not None
                        else None
                    ],
                    "Manually annotated image",
                ),
            ]

            self.add_to_cache(
                image_file, plate_detector.detected_texts, plate_detector.accuracies
            )

            # open new window with image detail
            self.image_detail = ImageDetail(
                images,
                plate_detector.detected_texts,
                plate_detector.accuracies,
                title=f"Image detail {image_file}",
                closeEventParent=self.update_ui,
            )

            self.image_detail.show()
        except Exception as e:
            if isinstance(e, cv2.error):
                QMessageBox.critical(
                    self, "Error", "Error while processing image (Try another Gaussian kernel size): {}".format(e)
                )
            elif e.args[0] == "No plate found in the image.":
                QMessageBox.information(self, "Information", "No plate detected")
            elif e.args[0] == "Annotation not found":
                QMessageBox.information(self, "Information", "Annotation not found")
            else:
                QMessageBox.critical(
                    self, "Error", "Error while processing image: {}".format(e)
                )
                traceback.print_exc()

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_ui()

    def next_page(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.update_ui()

    def load_cache(self) -> dict:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    def save_cache(self, cache: dict):
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

    def load_image_cache(self, image_file: str) -> Optional[dict]:
        cache = self.load_cache()
        image_file = os.path.basename(image_file)

        if image_file in cache:
            return cache[image_file]
        else:
            return None

    def add_to_cache(self, image_file: str, texts: List[str], accuracies: List[float]):
        print(image_file, texts, accuracies)
        cache = self.load_cache()
        image_file = os.path.basename(image_file)

        if len(texts) != len(accuracies):
            accuracies = [-1] * len(texts)

        # merge texts and accuracies
        cache[image_file] = list(zip(texts, accuracies))
        # sort with best accuracy first
        cache[image_file].sort(key=lambda x: x[1], reverse=True)

        self.save_cache(cache)


if __name__ == "__main__":
    app = QApplication([])
    image_viewer = Gallery("./images", 9, 3)
    image_viewer.show()
    app.exec()
