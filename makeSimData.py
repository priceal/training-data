# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:14:33 2021

@author: eeste
"""
runfile('initialize.py', current_namespace=True)

# define size of frame
frameDims = (7,7)    # dimensions of frame

# span of where particle is randomly placed
xSpan = (2.0, 4.0)
ySpan = (2.0, 4.0)

# hit region --- where particle is in "hit" class (for now, square)
lowerLimit, upperLimit = 2.5, 3.5

#define parameters describing particle
ampMean = 150.0          # mean amplitude of particle
ampSTD = 30.0            # STD of particle amplitudes
HWMean = 1.2          # mean halfwidth of particle
HWSTD = 0.2          # std of halfwidths

#define uniform background characteristics (set all = 0 if not wanted)
backgroundMean = 40
backgroundSTD = 10

# define noise characteristics if wanted (set all = 0 if not wanted)
noiseSigma = 5

# parameters for desired sample numbers
numberHits = 5000
numberNearMiss = 2500
numberBackground = 2500

# define data file name if you want to save it
saveData  = True
if saveData:
    saveFileName = 'nearMissData.pkl'

##############################################################################
##############################################################################
#  define some usefule variables from user supplied parameters:
xrange = [lowerLimit,upperLimit]
yrange = [lowerLimit,upperLimit]
center =  [ (frameDims[0]-1)/2 , (frameDims[1]-1)/2 ] 

Coordinates = []
frameList = []
# create the hits
for h in range(0, numberHits):
    currentCoord = g.rand2D(xrang=xrange,yrang=yrange)
    Coordinates.append(currentCoord)
    sigma = np.random.normal(HWMean,HWSTD)
    amp = np.random.normal(ampMean,ampSTD)
    frameList.append(g.GaussFrame(currentCoord,sigma,amp,frameDims) )

# create the near misses
for n in range(0, numberNearMiss):
    missCoordinate = center
    while missCoordinate[0]>lowerLimit and missCoordinate[0]<upperLimit and missCoordinate[1]>lowerLimit and missCoordinate[1]<upperLimit:
        missCoordinate = g.rand2D(xrang=xSpan,yrang=ySpan)
    Coordinates.append(missCoordinate)
    sigma = np.random.normal(HWMean,HWSTD)
    amp = np.random.normal(ampMean,ampSTD)
    frameList.append(g.GaussFrame(missCoordinate,sigma,amp,frameDims) )

# create the background frames and put all together
frames = np.concatenate((np.array(frameList),\
                         np.zeros((numberBackground,frameDims[0],frameDims[1]))
                         ))
xyclass = np.array([0]*numberHits+[1]*numberNearMiss+[2]*numberBackground)
coordinates = np.array(Coordinates) 

# now add noise and background
for frame in frames:
    frame += g.randFrame(noiseSigma,frameDims,mean=0)
    frame += np.random.normal(backgroundMean,backgroundSTD)

# convert to 8 bit image, and clip off negative pixels
frames = np.clip(frames,0,np.Inf)
frames = np.array(frames, dtype='int')

# plot targets to check
clrs = [ 'red' if cl else 'blue' for cl in xyclass[:-numberBackground] ]
plt.figure()
ax = plt.subplot()
ax.set_aspect('equal') 
ax.grid()
ax.scatter(coordinates[:,1],coordinates[:,0],c=clrs)

if saveData:
    with open(saveFileName,'wb') as file:
        pickle.dump((frames,xyclass,coordinates),file)
        