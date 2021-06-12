"""
creates an array of frames using image source and xy array

VARIABLE SET BY SCRIPT:
frames    AN ARRAY OF 2D FRAMES

v. 2021 05 27

"""

# source of image(s) for frames. 
frames_image = 0       

# define array of frame centers (xy) cropping multiple boxes from a single image 
frames_xy = integerXY

# frame parameters
frames_width = 3        
frames_height = 3

######################################################################
# do not change code below this line
######################################################################
######################################################################
frames = pa.cropframes(frames_image,frames_xy,bx=frames_width,\
                       by=frames_height,imgDF=imageDF)
print("{} frames made from image.".format(len(frames)))
print('frame dimensions: {}'.format(frames[0].shape))

        