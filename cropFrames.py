"""
creates an array of frames using image source and xy array

VARIABLE SET BY SCRIPT:
frames    AN ARRAY OF 2D FRAMES

v. 2021 05 27

"""

# source of image(s) for frames. 
frames_image = 0       

# define array of frame centers (xy) cropping multiple boxes from a single image 
frames_xy = XY

# frame parameters
frames_border = 3        

dims = (1920,1200)

xrange = (frames_border,dims[0]-frames_border-1)
yrange = (frames_border,dims[1]-frames_border-1)

XY_c0 = frames_xy[ frames_xy[:,0] >= xrange[0] ]
XY_c1 = XY_c0[ XY_c0[:,0] <= xrange[1] ]
XY_c2 = XY_c1[ XY_c1[:,1] >= yrange[0] ]
XY_c3 = XY_c2[ XY_c2[:,1] <= yrange[1] ]


######################################################################
# do not change code below this line
######################################################################
######################################################################
frames = pa.cropframes(frames_image,XY_c3,bx=frames_border,\
                       by=frames_border, imgDF=imageDF)
print("{} frames made from image.".format(len(frames)))
print('frame dimensions: {}'.format(frames[0].shape))

        