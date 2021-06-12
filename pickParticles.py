# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:03:51 2021

@author: priceal

script to pick pixels. Usage:
    
<space>                     pick a point
<backspace> or <delete>     remove last point
<enter>                     end and return points

"""

image = 0           # set image number to use
saveData = False     # save the points or not

if saveData:
    dataFileName = 'testdata.npy'  # set if needed

##############################################################################
pickedXY = pa.pickxy(image,imgDF=imageDF)

integerXY = np.array(np.round(pickedXY),dtype=int)

if saveData:
    np.save(dataFileName,integerXY)