from dronevision_library import visual_algorithm

##################################################################
# SAMPLE DEFINITION, TO CALL THE FUNCTION IN THE LIBRARY, BELOW#
##################################################################
# For further documentation, please visit https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/README.md


def object_detection():
    # Please adjust these parameters before running the script:
    scantime = 0  # Time that the function will scan before returning an output.
    safezonesize = 0  # DEBUG PARAMETER! Will not be used if you do not wish to debug.
    desiredcolor = "white"  # Color of object that should be detected. For example "white". Visit README.md for available options.
    debug = True  # This will enable debugging output, such as text in the runtime environment.
    videosource = '../data/video_source/sample_video.mp4'  # Determine the video source (video-file). If no input, or the video file could not be opened, camera capture will initialize.
    # Syntax for videosource: '../folder/videosource.avi'
    # Attention! Debugging will enable a infinite loop of output, and no variables will be returned! (or until the video is finished)
    radialdistance, theta = visual_algorithm(scantime, safezonesize, desiredcolor, debug, videosource)
    print(radialdistance, theta)  # Remove this if you don't want the output to be displayed (non-debug mode)
