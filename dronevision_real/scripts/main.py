# Import only Library of VisualDetection
import cv2
from dronevision_library import test_accuracy
from dronevision_library import color_controller
from dronevision_library import visual_algorithm
from dronevision_library import create_transformation

# Select which function out of the library you would like to choose (CHOOSE ONE AT A TIME!)
# If set to true, the library will be called upon execution of the main.py script.
object_detection_active = False  # Enables functions belonging to the object detection.
accuracy_test_active = False  # Enables functions belonging to the accuracy testing (of object and color).
color_detection_active = False  # Enables functions belonging to the color detection.
data_augmentation_active = True  # Enables functions belonging to the data augmentation.

##################################################################
# SAMPLE DEFINITIONS, TO CALL THE FUNCTIONS IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md

# /* object_detection(): */
if object_detection_active:
    # Please adjust these parameters before running the script:
    scantime = 3  # Time that the function will scan before returning an output.
    safezonesize = 0  # DEBUG PARAMETER! Will not be used if you do not wish to debug.
    desiredcolor = "white"  # Color of object that should be detected. For example "white". Visit README.md for available options.
    debug = False  # This will enable debugging output, such as text in the runtime environment.
    # Attention! Debugging will enable a infinite loop of output, and no variables will be returned!
    radialdistance, theta = visual_algorithm(scantime, safezonesize, desiredcolor, debug)
    print(radialdistance, theta)  # Remove this if you don't want the output to be displayed (non-debug mode)

# /* accuracy_test(): */
if accuracy_test_active:
    #Note that the result.txt file will be stored in the root folder of your project.
    # Please adjust these parameters before running the script:
    sample_images = '../data/sample_images/' #"PATH_TO_SAMPLE-IMAGES_FOLDER"  # Should be in following syntax: '../ANY_FOLDER_IN_ROOT_OF_PROJECT/sample_images/'
    export_images = '../data/export_images/' #"PATH_TO_EXPORT-IMAGES_FOLDER"  # Should be in following syntax: '../ANY_FOLDER_IN_ROOT_OF_PROJECT/export_images/'
    boolcolor = True  # If set to True, the color of the detected object (if applicable) will be printed on the image.
    test_accuracy(sample_images, export_images, boolcolor)

# /* color_detection(): */
if color_detection_active:
    # Please adjust these parameters before running the script:
    source_image_path = "../data/training_dataset/blue/blue2.jpg" #"PATH_TO_IMAGE"  # Fill in the path to the file where your image is stored. Syntax: '../ANYFOLDER/%IMAGE%.png' !!Case sensitive!!
    debugparam = True  # This will enable debugging output, such as text in the runtime environment.
    # Debugging outputs (text) of color will be enabled by default.
    source_image = cv2.imread(source_image_path)
    color = color_controller(source_image, debugparam)
    print(color)  # Remove this if you don't want the color to be displayed

# /* data_augmentation(): */
if data_augmentation_active:
    # Please adjust these parameters before running the script:
    path_to_importfolder = 'G:/Programmeren_Bachelorproef/dronevision_real (github)/data/sample_images/' #'PATH_TO_IMPORTFOLDER'  # Folder with images that need to be augmented,
    # format like: 'drive_letter:/folder/folder/' (don't forget '/' at the end)
    path_to_exportfolder = 'G:/Programmeren_Bachelorproef/dronevision_real (github)/data/export_images/' #'PATH_TO_EXPORTFOLDER'  # Folder where augmented images should be stored,
    # format like: 'drive_letter:/folder/folder/' (don't forget '/' at the end)
    start_number = 0  # Starting number for image output. Change to something else than 0 if you would perform Data Augmentation multiple times in the same folder.
    repeat_variable = 2  # How many times the same image needs to be augmented.
    # For ex.: 101 sample pictures will result into 2020 pictures, if repeat_variable is set to 20.
    create_transformation(path_to_importfolder, path_to_exportfolder, start_number, repeat_variable)
