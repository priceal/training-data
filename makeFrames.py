# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)

# define parameters
frameDims = (5,5)   # dimensions of frame
xSpan = (-0.5,4.5)   # x range of peak positions
ySpan = (-0.5,4.5)   # y range
amp = 150.0          # amp of gaussian
sigma = 1.5         # std of gaussian
numberSamples = 235

# hit region --- where value is TRUE
xrange = [1.,3.]
yrange = [1.,3.]

# define noise characteristics if wanted
noiseMean = 20
noiseSigma = 10.0
addNoise = False

# define data file name if you want to save it
saveData  = False
if saveData:
    saveFileName = 'data/test.pkl'

##############################################################################
##############################################################################
# setup training input frames
coordinates = g.multiRand2D(numberSamples,xr=xSpan,yr=ySpan)
frames = g.multiGaussFrame(coordinates,sigma,amp,frameDims)
if addNoise:
    for frame in frames:
        frame += g.randFrame(noiseSigma,frameDims,mean=noiseMean)
    frames = np.clip(frames,0,np.Inf)

frames = np.array(frames, dtype='int')

# now create classification for training
xclass = (coordinates[:,1]>xrange[0]) & (coordinates[:,1]<xrange[1])
yclass = (coordinates[:,0]>yrange[0]) & (coordinates[:,0]<yrange[1])
xyclass = xclass & yclass

# plot targets to check
clrs = [ 'red' if cl else 'blue' for cl in xyclass ]
plt.figure()
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(coordinates[:,1],coordinates[:,0],c=clrs)

if saveData:
    with open(saveFileName,'wb') as file:
        pickle.dump((frames,xyclass,coordinates),file)
        
        