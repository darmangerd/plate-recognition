import json
import cv2
import os
import numpy as np
from typing import Any, List, Tuple
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from bs4 import BeautifulSoup
import re
from lib import img


PROCESSOR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
MODEL = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
IMAGES_PATH = os.path.join("images")
IMAGES_ANNOTATIONS_PATH = os.path.join("annotations", "images")
ANNOTATIONS_PATH = os.path.join("annotations", "annotations")
SETTINGS_PATH = os.path.join("cache", "settings.json")


class Plate:
    """
    Class that aggregates all the steps of the pipeline to detect a plate.
    :param image_path: path to the image to process
    :param annotation_path: path to the annotation file of the image
    :param annotated_image_path: path to the annotated image
    :param original_image: original image
    :param original_image_cropped: original image cropped
    :param canny_image: image after canny edge detection
    :param connected_components_image: image after connected components detection
    :param stats: stats of the connected components
    :param potential_plates_regions: regions that could be a plate
    :param all_regions: all the regions detected by the connected components
    :param image_with_all_regions: image with all the regions detected drawn on it
    :param potential_plates: potential plates images
    :param potential_binarized_plates: potential binarized plates images
    :param splited_potential_binarized_plates: potential binarized plates images splitted in two parts (up and down)
    :param annotated_image: annotated image
    :param detected_texts: detected texts on the plates
    :param text: text of the annotation
    """

    image_path = None
    original_image = None
    original_image_cropped = None
    canny_image = None
    connected_components_image = None
    stats = None
    potential_plates_regions = []
    all_regions = []
    image_with_all_regions = None
    potential_plates = []
    potential_binarized_plates = []
    splited_potential_binarized_plates = []
    annotated_image = None
    detected_texts = []
    accuracies = []
    text = None

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

        # Load the image
        self.original_image = cv2.cvtColor(
            cv2.imread(image_path, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB
        )

        # Load the annotation, create the annotated image and load the text of the annotation if it exists
        self._annotation_path = os.path.join(
            ANNOTATIONS_PATH, os.path.basename(self.image_path).replace(".jpg", ".xml")
        )
        self._annoted_image_path = os.path.join(
            IMAGES_ANNOTATIONS_PATH, os.path.basename(self.image_path)
        )
        if os.path.isfile(os.path.abspath(self._annotation_path)) and os.path.isfile(
            os.path.abspath(self._annoted_image_path)
        ):
            self.text, self.objects = Plate.load_annotation(self._annotation_path)
            self.annotated_image = cv2.cvtColor(
                cv2.imread(self._annoted_image_path, cv2.COLOR_RGB2BGR),
                cv2.COLOR_BGR2RGB,
            )
            self.annotated_image = Plate.draw_annotation(
                self.annotated_image, self.objects
            )

        # Crop the image to remove be able to focus more on the plate and remove extra information from the borders
        CROP = Plate.read_setting("CROP")
        self.original_image_cropped = img.crop_image(self.original_image, *CROP)

        # Use the cany image to detect the edges of the plate
        self.canny_image = Plate.preprocess_image(self.original_image_cropped)

        # Use the connected components to detect the plate
        self.connected_components_image, self.stats = Plate.detect_connected_components(
            self.canny_image
        )

        # Detect the potential plates using connected components and some filters (aspect ratio, height, width)
        self.potential_plates_regions, self.all_regions = Plate.detect_plate(
            self.connected_components_image
        )

        # Draw the regions previously detected on the original image to see if the detection is correct
        self.image_with_all_regions = img.draw_regions(
            self.original_image_cropped.copy(),
            self.all_regions,
            color=(255, 0, 0),
            thickness=3,
        )
        self.image_with_all_regions = img.draw_regions(
            self.image_with_all_regions,
            self.potential_plates_regions,
            color=(0, 255, 0),
            thickness=1,
        )

        # Draw only the potential plates regions on the original image
        self.image_with_potential_regions = img.draw_regions(
            self.original_image_cropped.copy(), self.potential_plates_regions
        )

        # Binarize and split the image in two parts (up and down) to be able to detect the text on the plate
        # only if we found at least one potential plate
        if len(self.potential_plates_regions) > 0:
            self.potential_plates = [
                img.select_region(self.original_image_cropped, r)
                for r in self.potential_plates_regions
            ]

            self.potential_binarized_plates = [
                img.binarize(p) for p in self.potential_plates
            ]

            SPLIT_CROP = Plate.read_setting("SPLIT_CROP")[0]
            self.splited_potential_binarized_plates = [
                Plate.split_plate(p, SPLIT_CROP) for p in self.potential_binarized_plates
            ]

        # Detect the text on both parts of the splitted binarized plate
        self.detected_texts = [
            f"{Plate.detect_text(p[0])[0]}\n{Plate.detect_text(p[1])[0]}"
            for p in self.splited_potential_binarized_plates
        ]

        # Calculate the accuracy of the detected text
        # by comparing it to the ground truth text on every character
        if self.text is not None:
            self.accuracies = [
                Plate.calculate_accuracy(self.text, t) for t in self.detected_texts
            ]

    def load_annotation(
        annotation_path: str,
    ) -> Tuple[str, List[Tuple[str, int, int, int, int]]]:
        """
        Load the annotation of the image if it exists.
        :return: the text of the annotation and the objects of the annotation
        """

        data = open(annotation_path, "r").read()
        bs_data = BeautifulSoup(data, "lxml")
        xml_objects = bs_data.find_all("object")
        objects = []
        for xml_object in xml_objects:
            name = xml_object.find("name").text
            xmin = int(xml_object.find("xmin").text)
            ymin = int(xml_object.find("ymin").text)
            xmax = int(xml_object.find("xmax").text)
            ymax = int(xml_object.find("ymax").text)
            objects.append((name, xmin, ymin, xmax, ymax))

        text = "".join([o[0] for o in objects])

        return text, objects

    def calculate_accuracy(text, detected_text):
        """
        Calculate the accuracy of the detected text by comparing it to the ground truth text on every character.
        """

        if text is None or detected_text is None:
            return 0

        # Replace characters that are not in the alphabet or numbers by empty string
        text = re.sub(r"[^a-zA-Z0-9]", "", text)
        detected_text = re.sub(r"[^a-zA-Z0-9]", "", detected_text)

        shortest = min(len(text), len(detected_text))

        return (
            sum([1 if text[i] == detected_text[i] else 0 for i in range(shortest)])
            / shortest
        )

    def draw_annotation(
        image: cv2.Mat, objects: List[Tuple[str, int, int, int, int]]
    ) -> cv2.Mat:
        """
        Draw the annotation on the image.
        """

        image = image.copy()
        for name, xmin, ymin, xmax, ymax in objects:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(
                image, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
        return image

    def preprocess_image(image: cv2.Mat) -> cv2.Mat:
        """
        Preprocess the image by applying:
        - Gaussian filter to remove noise
        - Canny to detect the edges
        """

        # Convert to Gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian filter to remove noise
        GAUSSIAN_KERNEL = Plate.read_setting("GAUSSIAN_KERNEL")
        cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0, gray)

        canny = cv2.Canny(gray, 50, 150)

        return cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

    def detect_connected_components(
        image: cv2.Mat,
    ) -> Tuple[List[cv2.Mat], List[Tuple[int, int, int, int]]]:
        """
        From a Canny image, detect the connected components and return the image with the components and the stats.
        """

        # Find connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), connectivity=8
        )
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # Minimum size of particles we want to keep (number of pixels)
        MIN_SIZE = Plate.read_setting("MIN_SIZE_CONNECTED_COMPONENT")[0]

        # Answer image
        img2 = np.zeros((output.shape))
        # For every component in the image, we keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= MIN_SIZE:
                img2[output == i + 1] = 255

        # convert to 8UC1
        img2 = img2.astype(np.uint8)

        # img2 to opencv
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        return img2, stats

    def detect_plate(image: cv2.Mat) -> Tuple[List[cv2.Mat], List[cv2.Mat]]:
        """
        Detect the potential plates in the image, by finding the contours of the connected components and filtering it based
        on the ratio and size of the bounding rectangle.

        :param image: The image with the connected components.
        :return: The contours of the potential plates and all the contours.
        """

        all_contours, hierarchy = cv2.findContours(
            image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter the contours to keep only the ones that could be a plate (based on the ratio and size)
        def is_plate(contour):
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            ratio = float(w) / h

            ASPECT_RATIO_MIN_MAX = Plate.read_setting("ASPECT_RATIO_MIN_MAX")
            WIDTH_MIN_MAX = Plate.read_setting("WIDTH_MIN_MAX")
            HEIGHT_MIN_MAX = Plate.read_setting("HEIGHT_MIN_MAX")

            # Check if the contour is a plate
            return (
                ASPECT_RATIO_MIN_MAX[0] <= ratio <= ASPECT_RATIO_MIN_MAX[1]
                and WIDTH_MIN_MAX[0] <= w <= WIDTH_MIN_MAX[1]
                and HEIGHT_MIN_MAX[0] <= h <= HEIGHT_MIN_MAX[1]
            )

        # Filter the contours
        contours = list(filter(is_plate, all_contours))

        # Remove the contours that are inside other contours
        def is_inside(contour):
            x, y, w, h = cv2.boundingRect(contour)
            for other_contour in contours:
                x2, y2, w2, h2 = cv2.boundingRect(other_contour)
                if x > x2 and y > y2 and x + w < x2 + w2 and y + h < y2 + h2:
                    return True
            return False

        contours = list(filter(lambda contour: not is_inside(contour), contours))

        return contours, all_contours

    def split_plate(
        image: cv2.Mat,
        margin: int = 15
    ) -> Tuple[cv2.Mat, cv2.Mat, cv2.Mat]:
        """
        Split the plate in two parts, the upper part and the lower part.
        Because the OCR cannot isn't efficient at reading a text that as multiple lines, we need to split the plate in two
        :param image: The image of the plate.
        :param margin: The margin to add to the image to avoid the edges.
        :return: The upper part and the lower part of the plate.
        """
        # The signature starts by removing the edges of the image
        signature = image[margin:-margin, margin:-margin]

        # default value is the middle of the image
        separation_index = len(image) // 2
        # Find the row with the most black pixel
        for i, row in enumerate(signature):
            # If row is only black (0), then we found the separation index
            if np.count_nonzero(row) == 0:
                separation_index = i + margin
                break
            # Check for out of bounds error and if the row has more black pixel than the current separation index
            elif i < len(signature) // 2 and np.count_nonzero(row) > np.count_nonzero(
                signature[min(separation_index, len(signature) - 1)]
            ):
                separation_index = i + margin

        # Check if bound
        if separation_index < margin or separation_index > len(image) - margin:
            print("WARNING: separation index is out of bound")
            return None, None

        upper_plate = image[:separation_index]
        lower_plate = image[separation_index:]

        image_with_separation = cv2.line(
            image,
            (0, separation_index),
            (len(image[0]), separation_index),
            (255, 0, 0),
            2,
        )

        return upper_plate, lower_plate, image_with_separation

    def detect_text(image: cv2.Mat) -> str:
        """
        Detect the text in the image using the OCR model.
        :param image: The image to detect the text from.
        :return: The text detected in the image.
        """

        # Convert to PIL image
        pil_image = Image.fromarray(image)

        # Model process
        inputs = PROCESSOR(pil_image, return_tensors="pt", padding=True)
        outputs = MODEL.generate(**inputs, max_new_tokens=30)
        return PROCESSOR.batch_decode(outputs, skip_special_tokens=True)

    def read_setting(key: str) -> Any:
        """
        Read a setting from the settings file.
        :param key: The key of the setting.
        :return: The value of the setting.
        """

        with open(SETTINGS_PATH, "r") as file:
            obj = json.load(file)
            if key in obj:
                return obj[key]
            return None
