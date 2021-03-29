 _drone-object-detection-using-haar-cascades_
 -*- coding: utf-8 -*-
 -------------------------------------------------------------------------
Variable | Content
------------ | -------------
Author | Thomas Sabbe
Mail | thomas.sabbe@student.kuleuven.be
Date | 25/03/2021
 -------------------------------------------------------------------------
Welcome to the repository of object detection using haar cascades.

Disclaimer: some of the functions inside this repo have options to debug. This contains text output and/or visual output.
If you wish to see a practical tutorial, please visit [this link](https://youtu.be/_pKXK4r-oxs)

# Content of this repository, in correct order of the library.py script:
  1. Color detection: Determine the color of an object using 
  2. Drone Vision: Given two parameters (scantime and desired color), return a pixel-distance between drone and object, which has been detected with a cascade model. 
  3. Data augmentation: Augment positive images (images with desired object) to create a larger dataset.
  4. Test accuracy of a haar-cascade model (generated with OpenCV 'train cascade'; see documentation of that) 
  5. Test accuracy of color detection, using KNN Nearest Neighbours and Kmeans clustering: Detect color of an object, and print this on an image. 
 
# How do I install and run the repository?

1. Clone the repository to your desired location.
1. Open the 'real' directory with PyCharm or any Python IDE.
1. Make sure following files are present and in the correct folder:

Type of function | Data needed
------------ | -------------
For object detection | cascade.xml: '../src/cascade/cascade.xml', see [how to train cascade](https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/wiki/Haar-cascade-training)
For color detection | %color%.png: '../data/training_dataset/%color%.png' (for each color)
For data-augmentation | sample_images: Folder with images of desired object, to be augmented (is an argument so choice is arbitrary where the folder is located)
For cascade and color accuracy | N.A.

4. In the main script, call any desired method. Read the required parameters and explanation and run the script.
		There's NO need to open the 'VisualDetectionLibrary'. Every parameter necessary is stated in the 'main' script.
5. Check your results in the folders (if applicable) or read your outputs in the console (if applicable).
			Have fun!
&nbsp;
# Further (optional) detailed behaviour about content of library.py:

## 1. Color Detection

##### Uses following libraries:

* `<import csv>`
`<import operator>`
`<import os>`

##### Name of definition(s), input parameter(s) and return variable(s):
*/* definitions
Definition name | Context
------------ | -------------
def color_controller(source_image)	| _Name_: Main definition that returns color based upon child definitions below.
"" | _File usage_: Creates these files if it doesn't exist yet: ../src/training.data & ../src/test.data
def training |	_Name_: For every training-image, 'color_histogram_of_training_image' is called to create a histogram for the colors.
"" | _Folder usage_:	requires following folder: ../data/training_dataset/%color% (with color images)						
def color_histogram_of_training_image |	_Name_:Creates a .data file that contains color-cluster data. This definition is called by 'training'.
"" | _File usage_: Creates these files if it doesn't exist yet: ../src/training.data
def color_histogram_of_test_image | _Name_: Creates a .data file that contains color-cluster data. This color data is extracted from a test image that is presented to the training.data model/file.
"" | _File usage_: Creates these files if it doesn't exist yet: ../src/test.data
def knn_classifiermain | _Name_: Main definition for returning a prediction of the model to 'color_controller'. Uses both test.data and training.data files created in the definitions above, if they didn't exist yet.
def load_dataset | _Name_: Converts the test.data and training.data files to .csv vector files, in order to create training feature vectors and a test feature vector.
def response_of_neighbors | _Name_: Returns the amount votes of each neighbor around the color, presented in the test-image. This definition is called by 'knn_classifiermain', but uses argument 'neighbors' from 'k_nearest_neighbors' (in order to count the nearest neighbors, ofcourse...)
def k_nearest_neighbors | _Name_: Returns the k-amount nearest neighbors (and their distance) around the color value of the test-image. If k is set to '3', three possible clusters that are close to the value of the test-image will be returned, in an array.
def calculate_euclideandistance | _Name_: Used by 'k_nearest_neighbors' to determine the nearest distance between two positions (pixel-coördinates).

*/* parameters necessary to use function

Parameter name | Context
------------ | -------------
source_image | _Source_: image needed for color_controller definition. This is the only parameter of the whole 'color detection'-section.
##### Example to determine the color of an image of an object, as displayed in main.py:
* `<color_controller(image.png)>` 		_image.png_ : the image where the color needs to be determined.
##### Debug options
* There is an option to enable text output to display if the .data-files need to be generated or not.
 
 &nbsp;
## 2. Drone Vision 

    MAKE SURE YOU HAVE ONLY ONE VIDEO CAPTURE DEVICE CONNECTED.
##### Uses following libraries:
* `<import datetime>`
`<import os>`
`<from math import sqrt>`
`<import math>`
`<import cv2>`

##### Name of definition(s), input parameter(s) and return variable(s):
*/* definitions
Definition name | Context
------------ | -------------
def average	| _Name_: Determines the average of a list.
def addtolist |	_Name_: Defintion to add a variable to a list.
def color_switch | _Name_: Switches a color (str format) to RGB values (syntax: int, int, int)
def calculate_radialdistance_theta | _Name_: Calculate euclidian distane and theta, given a safezone and distance between camera and object.
def visual_algorithm | _Name_: Main definition of the object-detection script. Main goal is to return the return variables from 'calculate_radialdistance_theta'.

*/* parameters necessary to use function

Parameter name | Context
------------ | -------------
safezonesize | ONLY FOR DEBUGGING USE: Default 50, but change this if debugging is enabled.This will print a rectangle on the screen as an indication if the cross-object would be above the drone.
scantime | The time to scan for cross objects before def 'visual_algorithm' returns a radialdistance and direction theta.
desiredcolor | The color of the object you want to detect. For example; red, because your cross-object should be red and not blue.

##### Example to determine the position of an object, with respect to the center of the camera frame:
* `<visual_algorithm(2,50,"white")>`				
Will scan 2 seconds for a white cross and return the radial distance to that object, in case it has been found.
_DISCLAIMER!_ If no object has been found in the 2 second period, the definition will not stop until an object has been found.
##### Debug options
* There is an option to enable text output in the runtime box and to display a live-view image of where the object is (the bounding box of the object), for every 'scantime' seconds.
 
 &nbsp;
## 3. Data augmentation

##### Uses following libraries:
* `<import numpy as npe>`
`<import torchvision.transforms as transforms>`
`<import torchvision.datasets as datasets>`
`<from torchvision.utils import save_image>`

##### Name of definition(s), input parameter(s) and return variable(s):
*/* definitions
Definition name | Context
------------ | -------------
def create_transformation | _Name_: Main and only definition to create a series of augmented images.

*/* parameters necessary to use function

Parameter name | Context
------------ | -------------
path_to_importfolder| Folder with images that need to be augmented. 
path_to_exportfolder | Folder where augmented images should be stored.
start_number | Starting number for image output. Change to something else than 0 if you would perform Data Augmentation multiple times in the same folder.
repeat_variable | How many times the same image needs to be augmented. For ex.: 101 sample pictures will result into 2020 pictures, if repeat_variable is  set to 20.
	Attention: no return variable(s) exist for this function!
##### Debug options
* No debug options provided
##### Example to call function:
* `<create_transformation(G:\PyCharmProjects\sample_images, G:\PyCharmProjects\augmented_images, 0, 10))>`
 
 &nbsp;
## 4. & 5.: Accuracy of haar-cascade model and color detection.
	NOTICE THAT THE RESULTS WILL BE STORED TO "../data/accuracy_results.txt"! This cannot be changed from the main method
##### Uses following libraries:
* `<import cv2>`
##### Name of definition(s), input parameter(s) and return variable(s):
*/* definitions
Definition name | Context
------------ | -------------
def calculate_imagecount | _Name_: Counts the amount of images inside the 'sample_images'-folder.
def test_accuracy | _Name_: For each image in the 'sample_images'-folder, it lets the cascade determine an object (if there is an object present). Then, it determines the color of the object and prints it on an image in the 'export_images' folder.
"" | _Case_: If no object is found, the image will be written in the 'export_images' folder as "IMGNAME+NO-OBJECT".
"" | _Case_:  If there is an object, the image will be written in in the 'export_images' folder as "IMGNAME", with the bounding box on the detected object printed on the same image. Also, a "IMGNAME+CUT" image will be written. This is a cut image of only the area that the bounding box covers.

*/* parameters necessary to use function

Parameter name | Context
------------ | -------------
sample_images | Images that need to be examined by the cascade-model or the color-clustering algorithm
export_images | Export folder for the results.
##### Example to determine accuracy of a model using a sample dataset:
* `<test_accuracy("../data/sample_images/", "../data/export_images/")>`
##### Debug options
* No debug options provided, because this definition has already all debug functionalities by default.
	


##### *END OF README.md from drone-object-detection-using-haar-cascades**
