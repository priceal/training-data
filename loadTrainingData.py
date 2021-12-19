# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)


trainingSetFile = 'test.pkl'

# hit region --- where value is TRUE
#xrange = [3.5,4.5]
#yrange = [3.5,4.5]


##############################################################################
##############################################################################
# load in training set
with open(trainingSetFile, 'rb') as file:
#    (header,frames,coordinates,sigmaWidth,sigmaLength,angles,amp) = pickle.load(file)
      (frames,xyclass)  = pickle.load(file)
      
# read parameters from training set dimenstions
numberSamples, yDim, xDim = frames.shape
print('{} frames, {} x {}'.format(numberSamples,yDim,xDim))

# now create classification for training
#xclass = (coordinates[:,1]>xrange[0]) & (coordinates[:,1]<xrange[1])
#yclass = (coordinates[:,0]>yrange[0]) & (coordinates[:,0]<yrange[1])
#xyclass = xclass & yclass

# plot targets to check
clrs = [ 'red' if cl else 'blue' for cl in xyclass ]
plt.figure()
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(coordinates[:,1],coordinates[:,0],c=clrs)