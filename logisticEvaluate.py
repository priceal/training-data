
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)

testSetFile = 'test.pkl'

regModel = reg

##############################################################################
##############################################################################
# load in test set
with open(testSetFile, 'rb') as file:
    (frames,classification,coords) = pickle.load(file)
    
# read parameters from training set dimenstions
numberSamples, yDim, xDim = frames.shape
print('{} frames, {} x {}'.format(numberSamples,yDim,xDim))

# pre-processing of frames
scaled_frames = frames/frames.max()

# reshaping of frames and loading into X, Y arrays for training
X = scaled_frames.reshape( (numberSamples, yDim*xDim) )
Y = classification

# define regression model and train!
Xpredict = regModel.predict(X)

predictedHits = Xpredict.sum()
actualHits = Y.sum()
actualMisses = numberSamples - actualHits

truePositives = (Xpredict & Y).sum()
falsePositives = (Xpredict & ~Y).sum()
trueNegatives = (~Xpredict & ~Y).sum()
falseNegatives = (~Xpredict & Y).sum()
print('')
print('test set containt {} positive and {} negatives'\
      .format(actualHits,numberSamples-actualHits))

print('true positives', truePositives, '({:2.1f}%)'.format(100*truePositives/actualHits))
print('false negatives', falseNegatives, '({:2.1f}%)'.format(100*falseNegatives/actualHits))
print('')

print('true negatives', trueNegatives, '({:2.1f}%)'.format(100*trueNegatives/actualMisses))
print('false positives', falsePositives, '({:2.1f}%)'.format(100*falsePositives/actualMisses))

# plot points colored red/blue if predictid true/false
clrs = [ 'red' if cl else 'blue' for cl in Xpredict ]
plt.figure()
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(coords[:,1],coords[:,0],c=clrs)

