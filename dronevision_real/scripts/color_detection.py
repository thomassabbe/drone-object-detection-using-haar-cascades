from dronevision_library import color_controller
import cv2

##################################################################
# SAMPLE DEFINITION, TO CALL THE FUNCTION IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md


def color_detection():
    # Please adjust these parameters before running the script:
    source_image_path = ""  # "PATH_TO_IMAGE"  # Fill in the path to the file where your image is stored. Syntax: '../ANYFOLDER/%IMAGE%.png' !!Case sensitive!!
    debugparam = True  # This will enable debugging output, such as text in the runtime environment.
    # Debugging outputs (text) of color will be enabled by default.
    source_image = cv2.imread(source_image_path)
    print_test_and_training = True  # Will print a 3D plot of the HSV value of the observed picture inside the clusters of the training dataset.
    debugcolor = "RGB"  # Choose between HSV or RGB here to display a(nother) plot after color detection. Only active when debugparam = True. Has to be a string ("...")
    color = color_controller(source_image, debugparam, debugcolor, print_test_and_training)
    print(color)  # Remove this if you don't want the color to be displayed
