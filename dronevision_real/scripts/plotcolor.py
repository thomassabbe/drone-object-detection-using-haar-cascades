from dronevision_library import plotcolor
import cv2

##################################################################
# SAMPLE DEFINITION, TO CALL THE FUNCTION IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md

# If you wish to plot every color value in a 3D HSV/RGB plot of a picture
def plotcolor_of_image()
  #In order to read the image first, please specify the location:  
  source_image_path = ""  # "PATH_TO_IMAGE"  # Fill in the path to the file where your image is stored. Syntax: '../ANYFOLDER/%IMAGE%.png' !!Case sensitive!!
  # Debugging outputs (text) of color will be enabled by default.
  source_image = cv2.imread(source_image_path)
  color_space = "" # Please specify "RGB" or "HSV"
  plotcolor(source_image, color_space)
