from dronevision_library import test_accuracy

##################################################################
# SAMPLE DEFINITION, TO CALL THE FUNCTION IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md


def accuracy_test():
    # Note that the result.txt file will be stored in the root folder of your project.
    # Please adjust these parameters before running the script:
    sample_images = '../data/sample_images/'  # "PATH_TO_SAMPLE-IMAGES_FOLDER"  # Should be in following syntax: '../ANY_FOLDER_IN_ROOT_OF_PROJECT/sample_images/'
    export_images = '../data/export_images/'  # "PATH_TO_EXPORT-IMAGES_FOLDER"  # Should be in following syntax: '../ANY_FOLDER_IN_ROOT_OF_PROJECT/export_images/'
    boolcolor = True  # If set to True, the color of the detected object (if applicable) will be printed on the image.
    test_accuracy(sample_images, export_images, boolcolor)
