import cv2
from cv2 import Mat
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable


def print_infos(image: Mat, title: str = None) -> None:
    """
    Print image infos

    - the name
    - the definition (width x height)
    - the data type
    - the size in bytes
    - the number of channels
    - the minimum value
    - the maximum value
    - the average value
    - the standard deviation
    - the mode
    """

    print("Image name: ", title)
    print("Image shape: ", image.shape)
    print("Image size: ", image.size)
    print("Image type: ", image.dtype)
    print("Image min: ", np.min(image))
    print("Image max: ", np.max(image))
    print("Image mean: ", np.mean(image))
    print("Image std: ", np.std(image))
    print("Image mode: ", np.argmax(np.bincount(image.flatten())))


def print_images_infos(
    images: List[Mat], title: Callable[[Mat, int], str] = None
) -> None:
    """
    Print images infos

    - the name
    - the definition (width x height)
    - the data type
    - the size in bytes
    - the number of channels
    - the minimum value
    - the maximum value
    - the average value
    - the standard deviation
    - the mode
    """

    for i, image in enumerate(images):
        print_infos(image, title(image, i) if title else f"Image {i+1}")
        print()


def show(image: Mat, name: str = None, cmap: str = None) -> None:
    """
    Show image with matplotlib
    """

    plt.title(name)
    plt.imshow(image, cmap=cmap)
    plt.show()


def read(path: str) -> Mat:
    """
    Read image from disk
    """

    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def read_images(paths: List[str]) -> List[Mat]:
    """
    Read images from disk
    """

    return [read(path) for path in paths]


def subplot_images(
    images: List[Mat],
    title: Callable[[Mat, int], str] = None,
    cmap: str = None,
    n_rows: int = 1,
    n_cols: int = None,
    figsize: Tuple[int, int] = (20, 10),
) -> None:
    """
    Show images in subplots with matplotlib
    """

    if n_cols is None:
        n_cols = len(images)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for i, image in enumerate(images):
        axs[i].set_title(title(image, i) if title else f"Image {i+1}")
        axs[i].imshow(image, cmap=cmap)
        axs[i].axis("off")
    plt.show()


def subplot_channels(
    image: Mat, name: str, channels: Tuple[str, str, str] = ("b", "g", "r")
) -> None:
    """
    Show colors channels in subplots with matplotlib
    """

    images = cv2.split(image)
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    for i, image in enumerate(images):
        axs[i].set_title(f"{name} - Channel {i+1} ({channels[i]})")
        axs[i].imshow(image, cmap="gray")
        axs[i].axis("off")
    plt.show()


def subplot_channels_images(
    images: List[Mat],
    title: Callable[[Mat, int, str], str] = None,
    channels: Tuple[str, str, str] = ("b", "g", "r"),
) -> None:
    """
    Show colors channels in subplots with matplotlib
    """

    images_channels = [cv2.split(images) for images in images]
    fig, axs = plt.subplots(len(images), 3, figsize=(20, 10))
    for i, images in enumerate(images_channels):
        for j, image in enumerate(images):
            axs[i, j].set_title(
                title(image, i, channels[j])
                if title
                else f"Image {i+1} - Channel {j+1} ({channels[j]})"
            )
            axs[i, j].imshow(image, cmap="gray")
            axs[i, j].axis("off")
    plt.show()


def histogram(image: Mat, name: str) -> None:
    """
    Plot histogram of image
    """

    plt.title(name)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def subplot_histograms(
    images: List[Mat], title: Callable[[Mat, int], str] = None
) -> None:
    """
    Plot histograms of images
    """

    fig, axs = plt.subplots(1, len(images), figsize=(20, 10))
    for i, image in enumerate(images):
        axs[i].set_title(title(image, i) if title else f"Image {i+1}")
        axs[i].hist(image.ravel(), 256, [0, 256])
    plt.show()


def subplot_channels_histogram(
    image: Mat, name: str, channels: Tuple[str, str, str] = ("b", "g", "r")
) -> None:
    """
    Plot colors histogram of image
    """

    images = cv2.split(image)
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    for i, image in enumerate(images):
        axs[i].set_title(f"{name} - Channel {i+1} ({channels[i]})")
        axs[i].hist(image.ravel(), 256, [0, 256])
    plt.show()


def subplot_channels_histogram_images(
    images: List[Mat],
    title: Callable[[Mat, int, str], str] = None,
    channels: Tuple[str, str, str] = ("b", "g", "r"),
) -> None:
    """
    Plot colors histogram of image
    """

    images_channels = [cv2.split(images) for images in images]
    fig, axs = plt.subplots(len(images), 3, figsize=(20, 10))
    for i, images in enumerate(images_channels):
        for j, image in enumerate(images):
            axs[i, j].set_title(
                title(image, i, channels[j])
                if title
                else f"Image {i+1} - Channel {j+1} ({channels[j]})"
            )
            axs[i, j].hist(image.ravel(), 256, [0, 256])
    plt.show()


def save(image: Mat, path: str = None) -> None:
    """
    Save image to disk
    """

    cv2.imwrite(path, image)


def save_images(images: Mat, title: Callable[[Mat, int], str] = None) -> None:
    """
    Save images to disk
    """

    for i, image in enumerate(images):
        save(image, title(image, i) if title else f"Image {i+1}")


def split(image: Mat) -> Mat:
    """
    Split image into channels
    """

    return cv2.split(image)


def color_channels(image: Mat) -> Tuple[Mat, Mat, Mat]:
    """
    Split image into 3 channels, return a tuple of 3 images (R, G, B)
    """

    b, g, r = cv2.split(image)

    return r, g, b


def color_channels_with_numpy(image: Mat) -> Tuple[Mat, Mat, Mat]:
    """
    Split image into 3 channels, return a tuple of 3 images (R, G, B)
    """

    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    return r, g, b


def normalize(image: Mat, min: int = 0, max: int = 255) -> Mat:
    """
    Normalize image to [min, max]
    """

    return cv2.normalize(image, None, min, max, cv2.NORM_MINMAX)


def equalize_histogram(image: Mat) -> Mat:
    """
    Equalize image histogram
    """

    return cv2.equalizeHist(image)


def clahe(
    image: Mat, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)
) -> Mat:
    """
    Apply CLAHE to image
    """

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)


def negative(image: Mat) -> Mat:
    """
    Apply negative to image
    """

    return cv2.bitwise_not(image)


def increase_contrast_image_with_lut(image: Mat) -> Mat:
    """
    Increase contrast of image using LUT
    """

    # Create a LUT to increase the contrast
    lut = np.zeros(256, dtype=image.dtype)
    for i, v in enumerate(lut):
        lut[i] = 255 * (i / 255) ** 2

    # Apply the LUT to the image
    return cv2.LUT(image, lut)


def apply_masks(image: Mat, masks: List[int]) -> List[Mat]:
    """
    Apply masks to image and return a list of images, should be binary masks (0b00000000)
    """

    images = []

    for mask in masks:
        images.append(cv2.bitwise_and(image, mask))

    return images


def gray(image: Mat) -> Mat:
    """
    Convert image to gray scale
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rgb(image: Mat) -> Mat:
    """
    Convert image to RGB
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def hsv(image: Mat) -> Mat:
    """
    Convert image to HSV
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convolve(image: Mat, kernel: Mat) -> Mat:
    """
    Apply convolution to image
    """

    return cv2.filter2D(image, -1, kernel)


def threshold(
    image: Mat, threshold: int, max_t: int = 255, type: int = cv2.THRESH_BINARY
) -> Mat:
    """
    Apply threshold to image
    """

    return cv2.threshold(image, threshold, max_t, type)[1]


def morphological_transformations(
    image: Mat, kernel: Mat, iterations: int = 1, type: int = cv2.MORPH_OPEN
) -> Mat:
    """
    Apply morphological transformations to image
    """

    return cv2.morphologyEx(image, type, kernel, iterations=iterations)


def gaussian_blur(image: Mat, kernel_size: Tuple[int, int] = (5, 5)) -> Mat:
    """
    Apply gaussian blur to image
    """

    return cv2.GaussianBlur(image, kernel_size, 0)


def dilation(image: Mat, kernel: Mat, iterations: int = 1) -> Mat:
    """
    Apply dilation to image
    """

    return cv2.dilate(image, kernel, iterations=iterations)


def erosion(image: Mat, kernel: Mat, iterations: int = 1) -> Mat:
    """
    Apply erosion to image
    """

    return cv2.erode(image, kernel, iterations=iterations)


def opening(image: Mat, kernel: Mat, iterations: int = 1) -> Mat:
    """
    Apply opening to image
    """

    return morphological_transformations(image, kernel, iterations, cv2.MORPH_OPEN)


def closing(image: Mat, kernel: Mat, iterations: int = 1) -> Mat:
    """
    Apply closing to image
    """

    return morphological_transformations(image, kernel, iterations, cv2.MORPH_CLOSE)


def border_with_xor(imageA: Mat, imageB: Mat) -> Mat:
    """
    Get the borders of subjects with dilation to image
    """

    return cv2.bitwise_xor(imageA, imageB)


def fourier_transform(image: Mat) -> Mat:
    """
    Apply fourier transform to image
    """

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    return f, fshift


def phase_spectrum(image: Mat) -> Mat:
    """
    Apply phase spectrum to image
    """

    return np.angle(fourier_transform(image)[1])


def magnitude_spectrum(image: Mat) -> Mat:
    """
    Apply magnitude spectrum to image
    """

    return 20 * np.log(np.abs(fourier_transform(image)[1]))


def slot_threshold(image: Mat, low_threshold: int, high_threshold: int) -> Mat:
    """
    Apply slot threshold to image:
    - low_threshold < img < high_threshold --> 255
    - img < low_threshold or img > high_threshold --> 0
    """

    threshold, binary = cv2.threshold(
        image, high_threshold, 255, cv2.THRESH_TOZERO_INV
    )  # > high_threshold --> 0
    threshold, binary = cv2.threshold(
        binary, low_threshold, 255, cv2.THRESH_TOZERO
    )  # < low_threshold --> 0
    threshold, binary = cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY)  # !=0 --> 255

    return binary


def resize(
    image: Mat, width: int, height: int, interpolation: int = cv2.INTER_NEAREST
) -> Mat:
    """
    Resize image
    :param image: image to resize
    :param width: new width
    :param height: new height
    :param interpolation: interpolation method
    :return: resized image
    """

    return cv2.resize(image, (width, height), interpolation=interpolation)


def scale(image: Mat, scale: float, interpolation: int = cv2.INTER_NEAREST) -> Mat:
    """
    Scale image
    :param image: image to scale
    :param scale: scale factor
    :param interpolation: interpolation method
    :return: scaled image
    """

    return resize(
        image,
        int(image.shape[1] * scale),
        int(image.shape[0] * scale),
        interpolation=interpolation,
    )


def crop_image(
    image: Mat, start_top: int, start_left: int, end_bottom: int, end_right: int
) -> cv2.Mat:
    im = image[start_top:-end_bottom, start_left:-end_right]
    return im


def draw_regions(
    image: cv2.Mat,
    regions,
    color=(0, 255, 0),
    thickness=2,
    label: Callable[[int], str] = None,
):
    """
    Draw regions on image
    :param image: image to draw regions on
    :param regions: regions to draw
    :param color: color of the regions
    :param thickness: thickness of the regions
    :param label: label of the regions
    :return: image with regions drawn
    """

    # Draw the regions on the original image
    for i, region in enumerate(regions):
        x, y, w, h = cv2.boundingRect(region)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)

        if label is not None:
            cv2.putText(
                image,
                label(i),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color=color,
                thickness=thickness,
            )

    return image


def select_region(image: cv2.Mat, region) -> cv2.Mat:
    """
    Select a region from an image
    :param image: image to select region from
    :param region: region to select
    :return: selected region
    """

    x, y, w, h = cv2.boundingRect(region)
    return image[y : y + h, x : x + w]


def binarize(image: cv2.Mat) -> cv2.Mat:
    """
    Binarize an image using Otsu's method
    :param image: image to binarize
    :return: binarized image
    """

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binarize
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def resize(
    image: cv2.Mat, width: int, height: int, interpolation=cv2.INTER_LINEAR
) -> cv2.Mat:
    """
    Resize image
    :param image: image to resize
    :param width: new width
    :param height: new height
    :param interpolation: interpolation method (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST)
    :return: resized image
    """

    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_to_match_width(
    image: cv2.Mat, width: int, interpolation=cv2.INTER_LINEAR
) -> cv2.Mat:
    """
    Resize image to match a given width
    :param image: image to resize
    :param width: new width
    :param interpolation: interpolation method (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST)
    :return: resized image
    """

    s = width / image.shape[1]
    return scale(image, s, interpolation=interpolation)
