#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:27:26 2021

@author: allen
"""


file1 = 'TRUE_2022_01_11.pkl'
file2 = 'FALSE_2022_01_11.pkl'

saveName = 'ALL_2022_01_11.pkl'

##############################################################################
##############################################################################
# load in training set
with open(file1, 'rb') as file:
      (header1, parameters1, frames1, xyclass1)  = pickle.load(file)
(coordinates1,sigmaWidth1,sigmaLength1,angles1,amp1) = parameters1
        
# load in training set
with open(file2, 'rb') as file:
      (header2, parameters2, frames2, xyclass2)  = pickle.load(file)
(coordinates2,sigmaWidth2,sigmaLength2,angles2,amp2) = parameters2
     
frames = np.vstack((frames1,frames2))
xyclass = np.hstack((xyclass1,xyclass2))
coordinates = np.vstack((coordinates1,coordinates2))
sigmaWidth = np.hstack((sigmaWidth1,sigmaWidth2))
sigmaLength = np.hstack((sigmaLength1,sigmaLength2))
angles = np.hstack((angles1,angles2))
amp= np.hstack((amp1,amp2))

parameters = (coordinates,sigmaWidth,sigmaLength,angles,amp) 
with open(saveName,'wb') as file:
    pickle.dump( (header1, parameters, frames, xyclass), file )
        
        
        