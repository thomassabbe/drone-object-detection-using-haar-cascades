from dronevision_library import create_transformation

##################################################################
# SAMPLE DEFINITION, TO CALL THE FUNCTION IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md


# /* data_augmentation():
    # Please adjust these parameters before running the script:
    path_to_importfolder = 'PATH_TO_IMPORTFOLDER'  # Folder with images that need to be augmented,
    # format like: 'drive_letter:/folder/folder/' (don't forget '/' at the end)
    # !!!IMPORTANT!!! Your sample images have to be in the subfolder of the folder you've chosen for sample_images.
    # Otherwise, the script will generate an error saying it coudln't find any images.
    path_to_exportfolder = 'PATH_TO_EXPORTFOLDER'  # Folder where augmented images should be stored,
    # format like: 'drive_letter:/folder/folder/' (don't forget '/' at the end)
    start_number = 0  # Starting number for image output. Change to something else than 0 if you would perform Data Augmentation multiple times in the same folder.
    repeat_variable = 0  # How many times the same image needs to be augmented.
    # For ex.: 101 sample pictures will result into 2020 pictures, if repeat_variable is set to 20.
    create_transformation(path_to_importfolder, path_to_exportfolder, start_number, repeat_variable)
