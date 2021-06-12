"""
Loads and displays image. 

VARIABLE SET BY SCRIPT:
image         the loaded image

v. 2021 02 10

"""
image_number = 0

##############################################################################
##############################################################################

image = pa.loadim( imageDF['path'][image_number] )

print('displaying image number {}: '.format(image_number) + \
      imageDF['path'][image_number] )
print('image dimensions: {} '.format(image.shape) )

pa.showimage( image, imageDF['path'][image_number] )

