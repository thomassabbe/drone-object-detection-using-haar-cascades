from dronevision_library import plotcolor

##################################################################
# SAMPLE DEFINITION, TO CALL THE FUNCTION IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md

# If you wish to plot every color value of pictures inside a folder (put one image to only obtain the HSV plot of one image)
def plotcolor_of_image()
 "Keyword arguments:
  #Read the image first, please specify the location:  
  
  image -- Image that needs to be plotted.
    color_space -- "RGB" or "HSV"
    # Please adjust these parameters before running the script:
    scantime = 0  # Time that the function will scan before returning an output.
    safezonesize = 0  # DEBUG PARAMETER! Will not be used if you do not wish to debug.
    desiredcolor = "white"  # Color of object that should be detected. For example "white". Visit README.md for available options.
    debug = False  # This will enable debugging output, such as text in the runtime environment.
    videosource = ''  # Determine the video source (video-file). If no input, or the video file could not be opened, camera capture will initialize.
plotcolor(image, color_space):
