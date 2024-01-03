# Plates

## Description
During our studies, and more precisely during our course of image processing, we realized the project "Plates". This project consists in developing an application which implements algorithms and methods of image processing in order to solve a specific problem.

The objective of this application is to detect the identification numbers of vehicle license plates from pictures taken by a surveillance camera. These images are always captured from the same angle by the same camera, which facilitates the implementation of the algorithm. Moreover, since these are Vietnamese license plates, the processing will be adapted to the specificities of these plates.

To extract the characters contained on the license plates, we implemented several image processing algorithms such as Canny, Hough and other image segmentation techniques. The application was developed in Python, and we used the OpenCV library and a pre-trained Transformers model. The images come from a dataset from a project on GitHub .

We also created a user-friendly graphical user interface (GUI) that allows users to browse through a gallery of images, modify the detection parameters, start the license plate detection process, and display the results detailing the different steps. In addition, the accuracy of the score obtained by the model is also displayed.

## Pipeline

![pipeline]([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg](https://github.com/darmangerd/plate-recognition/blob/main/docs/pipeline.png)?raw=true)

During this project, our main objective was to implement functional license plate detection using classical image processing algorithms, while minimizing the use of Machine Learning. To begin, we implemented edge detection on the image. After performing a series of tests with the different methods seen in class (derivative, Prewitt, Sobel, Canny), we found that Canny's method is the most efficient. However, to obtain the best possible results, it is necessary to apply a Gaussian filter beforehand to remove the noise from the image, the size of the kernel can be modified in the program parameters in order to improve the detection on a specific image (`"GAUSSIAN_KERNEL"= [15, 15]`, works well in general). 

After having obtained the contours, we can extract from this image all the related components thanks to OpenCV, this method extracts the plate well, but we also recover a lot of regions not including any plate. To overcome this problem, we applied rules that each region must respect to be considered as license plates: the aspect ratio, the length and the height must be included in a range (which can be changed via the parameters).
```
"ASPECT_RATIO_MIN_MAX" = [0.3, 1.5]
"HEIGHT_MIN_MAX" = [40, 150]
"WIDTH_MIN_MAX" = [40, 140]
```
Moreover we can specify the minimum size of the related components in order to filter out those that are too small (`"MIN_SIZE_CONNECTED_COMPONENT" = [10]`). Once the regions potentially containing patches are extracted, we can highlight and extract this part of the image from the original one. However, despite this, it may happen that the image processing process selects wrong regions that are not plaques, which is one of the limitations of our method.

Once the potential license plate regions are identified, we need to split them into two parts in order to isolate the plate characters on a single line of each. This process is mandatory, as the artificial intelligence model used only supports single-line text. In order to do this, we implemented a signature-based technique to separate into two upper and lower parts.  

This technique consists in searching the line that contains the most black pixels, as it often corresponds to the separation line between the two parts of the plate. This technique consists of looking for the line that contains the most black pixels (if possible on the whole line), and once found, symbolize it as the separation between the two lines. However, it is important to take into account that the edges of the image may contain noise that could distort the separation solution. To avoid this, we excluded the edges of the image from our search for the separation line, limiting ourselves to a central area of the image. 

This method allowed us to obtain a separation of the plate into two distinct parts in an efficient and, above all, adaptable way to different images in which the plates are not always the same size, or in the same angle.

Finally, each region potentially detected as a license plate is converted into a binary image to facilitate character recognition. The characters are then detected using a pre-trained Transformers model and the OpenCV library. The detected text is then displayed to the user, along with a correctness score for each prediction.

## File structure
- `*_ui.py` : GUI files
- `lib/`: Image processing files (Plate specific & image processing)
- `docs/`: Documentation files
- `cache/`: Cache files & settings files
  - `cache/settings.json`: Settings file (parameters)
  - `cache/cache.json`: Cache file (save the plate detection results)
- `annotations/`: Annotations files (for the model)
- `images/`: All the images used in the project, you can add your own images in this folder
- `*.ipynb`: Jupyter notebook files, used for etablishing the pipeline

## How to start
### Prerequisites
- Python 3.10.4

### Installation and execution
```sh
pip install -r requirements.txt
python gallery_ui.py
```
