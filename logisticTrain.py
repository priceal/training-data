# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)


trainingSetFile = 'nearMissData.pkl'

##############################################################################
##############################################################################
# load in training set
with open(trainingSetFile, 'rb') as file:
    (frames,classification,xy) = pickle.load(file)
    
# read parameters from training set dimenstions
numberSamples, yDim, xDim = frames.shape
print('{} frames, {} x {}'.format(numberSamples,yDim,xDim))

# pre-processing of frames
scaled_frames = frames/frames.max()

# reshaping of frames and loading into X, Y arrays for training
X = scaled_frames.reshape( (numberSamples, yDim*xDim) )
Y = classification 

# define regression model and train!
reg = linear_model.LogisticRegression()
reg.fit(X,Y)

# output results
print(reg.coef_)
print(reg.intercept_)


