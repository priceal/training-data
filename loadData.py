"""
load data directory

v. 2021 02 21

"""
runfile('initialize.py', current_namespace=True)

# define data directory and image file range
image_directory = 'C:/Users/priceal/Desktop/DOCUMENTS/research/PROJECTS/OLD/BLT_old/BLT/E51'

#############################################################################
#############################################################################
imageDF = pa.loadDir(image_directory)
print("loaded :", image_directory)
