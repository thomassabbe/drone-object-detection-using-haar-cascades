#-*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Thomas Sabbe
# --- Mail           : thomas.sabbe@student.kuleuven.be
# --- Date           : 25/03/2021
# -------------------------------------------------------------------------

# All the definitions in one script with correct documentation.
# It is recommended to keep grammatical mistake alerts off when opening this file.
# All imports are listed below this line:

# For color and object detection:
import csv
import datetime
import os
from math import sqrt
import math
import cv2
import operator
# For data-augmentation:
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image


######COLORDETECTION######
def calculate_euclideandistance(variable1, variable2, length):
    """
    Used by 'k_nearest_neighbors' to determine the nearest distance between two positions (pixel-coördinates).

    Keyword arguments:
    variable1 -- First coördinate
    variable2 -- Second coördinate
    length -- Amount of clusters in test-instance image.

    Return variables:
    distance (int, DEFAULT 0)
    """
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


def k_nearest_neighbors(training_feature_vector, testInstance, k):
    """
    Returns the k-amount nearest neighbors (and their distance) around the color value of the test-image. If k is set to '3',
    three possible clusters that are close to the value of the test-image will be returned, in an array.

    Keyword arguments:
    training_feature_vector -- vector file of the test image
    testInstance -- specific value at position 'x' in the training dataset array
    k -- K-Means amount of clusters

    Return variables:
    neighbors (int, DEFAULT 0)
    """
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculate_euclideandistance(testInstance,
                                           training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def response_of_neighbors(neighbors):
    """
    Returns the amount votes of each neighbour around the color, presented in the test-image.
    This definition is called by 'knn_classifiermain', but uses argument 'neighbors' from 'k_nearest_neighbors'
    (in order to count the nearest neighbors, ofcourse...)

    Keyword arguments:
    neighbors -- amount of neighbors around the test feature cluster centroid

    Return variables:
    sortedVotes[0][0] (int, DEFAULT 0)
    """
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def load_dataset(filename, filename2, training_feature_vector=[], test_feature_vector=[]):
    """
    Converts the test.data and training.data files to .csv vector files,
    in order to load training feature vectors and a test feature vector.

    Keyword arguments:
    filename -- training .csv file
    filename2 -- test_feature .csv file
    training_feature_vector=[] -- empty array for training_feature_vector data
    test_feature_vector=[] -- empty array for test_feature_vector data

    Return variables:
    none
    """
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])

    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            test_feature_vector.append(dataset[x])


def knn_classifiermain(training_data, test_data):
    """
    Main definition for returning a prediction of the model to 'color_controller'.
    Uses both test.data and training.data files created in the other definitions, if they didn't exist yet.

    Keyword arguments:
    training_data -- training.data file
    test_data -- test.data file

    Return variables:
    classifier_prediction[0] (string, DEFAULT "Color wasn't found due to an error")
    """
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    load_dataset(training_data, test_data, training_feature_vector, test_feature_vector)
    classifier_prediction = []  # predictions
    k = 3  # K value of k nearest neighbor
    for x in range(len(test_feature_vector)):
        neighbors = k_nearest_neighbors(training_feature_vector, test_feature_vector[x], k)
        result = response_of_neighbors(neighbors)
        classifier_prediction.append(result)
    try:
        return classifier_prediction[0]
    except:
        return "Color wasn't found due to an error"


def color_histogram_of_test_image(debugparam, test_src_image):
    """
    Creates a .data file that contains color-cluster data.
    This color data is extracted from a test image that is presented to the training.data model/file.

    Keyword arguments:
    debugparam -- bool operator that enables text output when debugging is enabled
    test_src_image -- image that's been imported, in array format

    Return variables:
    none
    """
    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            if debugparam:
                print(feature_data)

    with open('../data/color_recognition/test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):
    """
    Creates a .data training file that contains color-cluster data. This definition is called by 'training'.

    Keyword arguments:
    img_name -- image file from the training data collection

    Return variables:
    none
    """
    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('../data/color_recognition/training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():
    """
    For every training-image, 'color_histogram_of_training_image' is called to create a histogram for the colors.

    Keyword arguments:
    none

    Return variables:
    none
    """
    # red color training images
    for f in os.listdir('../data/training_dataset/red'):
        color_histogram_of_training_image('../data/training_dataset/red/' + f)

    # yellow color training images
    for f in os.listdir('../data/training_dataset/yellow'):
        color_histogram_of_training_image('../data/training_dataset/yellow/' + f)

    # green color training images
    for f in os.listdir('../data/training_dataset/green'):
        color_histogram_of_training_image('../data/training_dataset/green/' + f)

    # orange color training images
    for f in os.listdir('../data/training_dataset/orange'):
        color_histogram_of_training_image('../data/training_dataset/orange/' + f)

    # white color training images
    for f in os.listdir('../data/training_dataset/white'):
        color_histogram_of_training_image('../data/training_dataset/white/' + f)

    # black color training images
    for f in os.listdir('../data/training_dataset/black'):
        color_histogram_of_training_image('../data/training_dataset/black/' + f)

    # blue color training images
    for f in os.listdir('../data/training_dataset/blue'):
        color_histogram_of_training_image('../data/training_dataset/blue/' + f)


def color_controller(source_image, debugparam):
    """
    Main (parent) definition that returns color based upon child definitions.

    Keyword arguments:
    source_image -- test image source (array)
    debugparam -- bool operator that enables text output when debugging is enabled

    Return variables:
    prediction (string, DEFAULT "Color wasn't found due to an error")
    """
    # checking whether the training data is ready
    PATH = '../data/color_recognition/training.data'

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        if debugparam:
            print('training data is ready, classifier is loading...')
    else:
        if debugparam:
            print('training data is being created...')
        open('../data/color_recognition/training.data', 'w')
        training()
        if debugparam:
            print('training data is ready, classifier is loading...')

    # get the prediction
    color_histogram_of_test_image(debugparam, source_image)
    prediction = knn_classifiermain('../data/color_recognition/training.data', '../data/color_recognition/test.data')

    return prediction


######END OF COLORDETECTION######


######OBJECT DETECTION WITH HAAR CASCADE######


def average(lst):
    """
    Definition to determine an average of a list.

    Keyword arguments:
    lst - list with integers to determine the average

    Return variables:
    sum(lst) / len(lst) (int, DEFAULT 0)
    """
    return sum(lst) / len(lst)


def addtolist(integer, lst):
    """
    Defintion to add a variable to a list.

    Keyword arguments:
    integer -- integer which needs to be appended to the list lst
    lst -- target list to append the integer to

    Return variables:
    none
    """
    lst.append(integer)


def color_switch(color):
    """
    Color switch for bounding box color, in case debugging is enabled.

    Keyword arguments:
    color -- color, in string format

    Return variables:
    integer, integer, integer (int, int, int; DEFAULT 0,0,0)
    """
    if color == "white":
        return 255, 255, 255
    elif color == "black":
        return 0, 0, 0
    elif color == "red":
        return 234, 74, 28
    elif color == "yellow":
        return 231, 234, 28
    elif color == "orange":
        return 324, 147, 28
    elif color == "blue":
        return 28, 44, 234
    elif color == "green":
        return 28, 234, 34


def calculate_radialdistance_theta(x_between_centerandobject, y_between_centerandobject):
    """
    Calculate euclidian distance and theta, given a safezone and distance between camera and object.

    Keyword arguments:
    x_between_centerandobject -- horizontal distance between object and center camera frame
    y_between_centerandobject -- vertical distance between object and center camera frame

    Return variables:
    theta, radial_distance (int, int; DEFAULT 0, 0)
    """
    radial_distance = sqrt((x_between_centerandobject * x_between_centerandobject) + (
            y_between_centerandobject * y_between_centerandobject))
    # If structures to correctly determine the angle where the object is positioned.
    if y_between_centerandobject < 0:
        if x_between_centerandobject < 0:
            cos = (abs(x_between_centerandobject) / abs(radial_distance))
            theta_firstquadrant = math.degrees(math.acos(cos))
            theta = theta_firstquadrant + 180
            return theta, radial_distance
        if x_between_centerandobject > 0:
            cos = -1 * (abs(x_between_centerandobject) / abs(radial_distance))
            theta_secondquadrant = math.degrees(math.acos(cos))
            theta = theta_secondquadrant + 180
            return theta, radial_distance
    else:
        theta = math.degrees(math.acos(x_between_centerandobject / radial_distance))
        return theta, radial_distance


def visual_algorithm(scantime, safezonesize, desiredcolor, debug, videosource):
    """
    Main definition of the object-detection script. Main goal is to return bools and await direction completions (async def's) from drone script.

    Keyword arguments:
    scantime -- time to collect object positions before returning it
    safezonesize -- debug parameter to know how when an object is close to the center camera frame
    desiredcolor -- color (string) that needs to be detected
    debugparam -- bool operator that enables text output when debugging is enabled
    videosource -- source of video input

    Return variables:
    theta, radial_distance (int, int; DEFAULT 0, 0)
    """
    # Create empty lists for x and y coördinates
    np_list_x = list()
    np_list_y = list()
    # Add cross_cascade classifier.
    cross_cascade = cv2.CascadeClassifier('../data/cascade/cascade.xml')
    # Define a start time
    start_time = datetime.datetime.now()
    # Initialize capture of camera or video; depends on what has been chosen.
    if videosource == "":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        try:
            cap = cv2.VideoCapture(videosource)
        except:
            # If the capture couldn't initialize, webcam capture will be initialised anyways.
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        debug = True
    cv2.namedWindow("img")

    while cap.isOpened():
        # Determines the size of the safezone.
        #   If this is set too big, this can cause the drone to not land properly on object, when object is small
        #   (f. ex. when side lenght of cross equals 20 cm)
        safezone = safezonesize
        # Determine time before position of cross is sent to drone.
        #   If this time is too long, a quick moving drone will get faultive directions.
        #   If this is too short, the drone will be updated with directions too much.
        seconds = datetime.timedelta(seconds=scantime)
        # Bool variable that indicates when the correct color-object has been found.
        foundcorrectcross = False

        while True:
            # Read the capture initialised above.
            ret, img = cap.read()
            # apply medianBlur.
            blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #cv2.medianBlur(img, 3)
            # Get window width and height of image frame.
            window_width = cv2.getWindowImageRect("img")[2]
            window_height = cv2.getWindowImageRect("img")[3]
            # Detect object, 'crosses'.
            crosses = cross_cascade.detectMultiScale(blur, 100, 100)
            if debug:
                cv2.imshow('img', img)
                cv2.circle(img, (int(window_width / 2), int(window_height / 2)), int((safezone / 2)), (255, 0, 255), 2)

            # For loop: 'for every cross being detected, do...'
            for (x, y, w, h) in crosses:
                # Get center of a cross which had been detected.
                center_x = x + (w / 2)
                center_y = y + (h / 2)
                # Cut the existing image to get a piece of the color of the cross.
                img_cut = img[int(center_y - w): int(w + center_y),
                          int(center_x - w):int(
                              w + center_x)]

                img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
                # Call RGBController to check wheter the cross/object is Red, Blue or Green.
                # This method will return 'red', 'blue' or 'green', in a text format.
                try:
                    os.remove('../data/color_recognition/test.data')
                except:
                    pass
                color = str(color_controller(img_cut, False))
                if debug:
                    r, g, b = color_switch(color)
                    color_bgr = b, g, r
                    cv2.rectangle(img, (x, y), (x + w, y + h), (color_bgr), 5)
                    cv2.putText(img, 'Color: ' + color, (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 2, (color_bgr), 1, cv2.LINE_AA)
                    cv2.imshow('img', img)
                if color == desiredcolor:
                    addtolist(center_x, np_list_x)
                    addtolist(center_y, np_list_y)
                    foundcorrectcross = True
                if color != desiredcolor:
                    if debug:
                        print('Not the desired color')
                    if not foundcorrectcross:
                        start_time = datetime.datetime.now()
                # Determine current time and time delta between current time and start time.
                current_time = datetime.datetime.now()
                elapsed_time = current_time - start_time
                # If the elapsed time is greater than the determined scan time (above), we continue this if loop.
                #   Otherwise, the program continues scanning for more crosses and adds them to the list.
                if elapsed_time > seconds:
                    # Determine average position of all crosses added to the list:
                    av_x = average(np_list_x)
                    av_y = average(np_list_y)
                    # Clear the list off coördinates from previous for loop. These aren't needed anymore because
                    #   the average was just taken.
                    np_list_x.clear()
                    np_list_y.clear()
                    foundcorrectcross = False
                    # Determine distance between desired position of cross and actual position.
                    x_between_centerandobject = int(av_x - (window_width / 2))
                    y_between_centerandobject = int(window_height / 2 - av_y)
                    theta, radial_distance = calculate_radialdistance_theta(x_between_centerandobject,
                                                                            y_between_centerandobject)
                    # If debugging is enabled, directions for the drone to move will be displayed in order to know if the
                    # program functions correctly.
                    current_time = datetime.datetime.now()
                    if debug:
                        print(theta)
                        if 0 < theta < 90:
                            print('move drone to the right and up')
                        if 90 < theta < 180:
                            print('move drone to the left and up')
                        if 180 < theta < 270:
                            print('move drone to the left and down')
                        if 270 < theta < 360:
                            print('move drone to the right and down')
                        # When the object is very close to the drone, a 'safezone' trigger can be activated.
                        if radial_distance - int((safezone / 2)) < av_x < radial_distance + int((safezone / 2)):
                            print('In Safezone (X).')
                            if av_y > (window_height / 2 + safezone / 2):
                                print('move drone down')
                            if av_y < (window_height / 2 - safezone / 2):
                                print('move drone up')
                        if radial_distance - int((safezone / 2)) < av_y < radial_distance + int((safezone / 2)):
                            print('In Safezone (Y).')
                            if av_x > (window_width / 2 + safezone / 2):
                                print('move drone to the right')
                            if av_x < (window_width / 2 - safezone / 2):
                                print('move drone to the left')
                    if not debug:
                        return theta, radial_distance
                break
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


######END OF OBJECT DETECTION WITH HAAR CASCADE######


######DATA AUGMENTATION######

def create_transformation(path_to_importfolder, path_to_exportfolder, start_number, repeat_variable):
    """
    Main definition to create the transformations and store them.

    Keyword arguments:
    path_to_importfolder -- path to the import folder
    path_to_exportfolder -- path to the export folder
    start_number -- start number for writing images to the export folder
    repeat_variable -- amount of times an augment needs to be created from the same image

    Return variables:
    none
    """
    # Create a collection of transformations. The choices which tranformations are arbitrary.
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.5, hue=0.5, saturation=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

    # Create a dataset from an image-folder. This image-folder should contain all the sample-images you want to be augmented.
    val_set = datasets.ImageFolder(root=path_to_importfolder, transform=my_transforms)

    # Set a img_num. Default 0, but can be different if you want to add more images upon existing ones.
    img_num = start_number
    # For loop which will repeat 20 times per image in the dataset, explained above.
    # Ex.: 101 sample pictures will result into 2020 pictures.
    for _ in range(repeat_variable):
        for img, label in val_set:
            save_image(img, path_to_exportfolder + str(img_num) + '.jpg')
            img_num += 1


######END OF DATA AUGMENTATION######

######BEGIN OF TESTING CASCADE ACCURACY######

def calculate_imagecount(Foldername):
    """
    Calculate the number of files in a folder.

    Keyword arguments:
    Foldername -- Folder where images are stored.

    Return variables:
    len(image_list) (default (0))
    """
    image_list = []
    for filename in os.listdir(Foldername):
        image_list.append(filename)
    return len(image_list)


def test_accuracy(sample_images, export_images, boolcolor):
    """
    Main definition to test the accuracy of a haar-cascade model and/or the color detection algorithm

    Keyword arguments:
    sample_images -- sample images folder filepath
    export_images -- export images folder filepath
    boolcolor -- wheter the color should be detected and printed on the image or not (bool variable)

    Return variables:
    none
    """
    # Create text document for result data:
    f = open("../accuracy_results.txt", "w+")
    # Create Image list with succesful image detections:
    image_list_succesful_color = []
    countlistimages = 0
    seconds = datetime.timedelta(seconds=1)
    cross_cascade = cv2.CascadeClassifier('../src/cascade/cascade.xml')
    active = False
    for filename in os.listdir(sample_images):
        try:
            print(filename)
            f.write(filename + "\r\n")
            countlistimages = countlistimages + 1
            img = cv2.imread(os.path.join(sample_images, filename))
            # For example: "../data/export_images/"
            cv2.imwrite(export_images + "/" + str(filename) + 'NO-OBJECT' + '.jpg', img)
            gray_uncontrast = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            an_array = np.asarray(gray_uncontrast)
            multiplied_array = an_array * 1
            gray = multiplied_array
            crosses = cross_cascade.detectMultiScale(gray, 100, 100)
            for (x, y, w, h) in crosses:
                count = 0
                # Get center of a cross which had been detected.
                center_x = x + (w / 2)
                center_y = y + (h / 2)
                safezonesize = w
                # Cut the existing image to get a piece of the color of the cross.
                images = img[int(center_y - safezonesize): int(
                    safezonesize + center_y), int(center_x - safezonesize):int(safezonesize + center_x)]
                img_cut = images
                try:
                    os.remove(export_images + str(filename) + 'NO-OBJECT' + '.jpg')
                except:
                    pass
                cv2.imwrite(export_images + str(filename) + 'CUT' + '.jpg', img_cut)
                # img_cut = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
                if boolcolor == True:
                    color = str(color_controller(img_cut, False))
                if active == False:
                    filename_drawn = img
                    r, g, b = color_switch(color)
                    color_bgr = b, g, r
                    cv2.rectangle(filename_drawn, (x, y), (x + w, y + h),
                                  (color_bgr), 2)
                    if boolcolor == True:
                        cv2.putText(
                            filename_drawn,
                            'Prediction: ' + color,
                            (int(200), 60),
                            cv2.FONT_HERSHEY_PLAIN, 1, (color_bgr),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.imwrite(export_images + str(filename) + '.jpg', filename_drawn)
                    image_list_succesful_color.append(filename)
                    active = True
                    f.write("Added Picture with cross & color\r\n")
                    break
                if active == True:
                    for i in image_list_succesful_color:
                        if i != filename and count == 0:
                            image_list_succesful_color.append(filename)
                            f.write("Added Picture with cross & color\r\n")
                            filename_drawn = img
                            r, g, b = color_switch(color)
                            color_bgr = b, g, r
                            cv2.rectangle(filename_drawn, (x, y), (x + w, y + h),
                                          (color_bgr), 2)
                            if boolcolor == True:
                                cv2.putText(
                                    filename_drawn,
                                    'Prediction: ' + color,
                                    (int(200), 60),
                                    cv2.FONT_HERSHEY_PLAIN, 5, (color_bgr),
                                    1,
                                    cv2.LINE_AA,
                                )
                            cv2.imwrite(export_images + str(filename) + '.jpg', filename_drawn)
                            count = count + 1
                break
        except Exception as e:
            print(str(e))

    print("You can find the results of the exported data under '../results.txt'")
    f.write("Total amount of images where an object has been detected (can be True or False): " + str(len(
        image_list_succesful_color)) + "\r\n")
    f.write("Total amount of images observed: " + str(countlistimages) + "\r\n")


cv2.destroyAllWindows()
######END OF TESTING CASCADE ACCURACY######
