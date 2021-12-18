# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal

parameters:
    
    frame shape
    x,y spans of particle positions
    amplitude model (mean, std)
    sigma models (distribution)
    
    

v. 2021 12 18

"""
runfile('initialize.py', current_namespace=True)


# particle parameters
amplitude = (200.0, 40.0)    # (mean, sigma)
ampCut = (50.0, 255.0)     # (low, high)
ampClip = (-np.Inf, np.Inf)      # (low, high)

length = (1.0, 1.5)             # (mean, sigma)
lengthCut = (0.5, 2.0)      # (low, high)
lengthClip = (-np.Inf, np.Inf)      # (low, high)

ecc = (1.0, 1.0)              # (mean, sigma)
eccCut = (0.5, 3.0)           # (low, high)
eccClip = (1.0, np.Inf)      # (low, high)

theta = (0, 2.0*np.pi)              # (mean, sigma)

# define parameters
frameDims = (9,9)   # dimensions of frame
xSpan = (3,5)   # x range of peak positions
ySpan = (3,5)   # y range
numberSamples = 500

# hit region --- where value is TRUE
xrange = [2.5,3.5]
yrange = [2.5,3.5]

# define noise characteristics if wanted
noiseMean = 10
noiseSigma = 5.0
addNoise = True

# define data file name if you want to save it
saveData  = False
if saveData:
    saveFileName = 'data/test.pkl'

##############################################################################
##############################################################################
# setup training input frames
coordinates = g.multiRand2D(numberSamples,xr=xSpan,yr=ySpan)

sigmaLength = sf.gSamples( length, numberSamples, cut=lengthCut, clip=lengthClip)
eccentricities = sf.gSamples( ecc, numberSamples, cut=eccCut, clip=eccClip)
sigmaWidth = sigmaLength / eccentricities

angles = np.random.uniform( theta[0], theta[1], numberSamples)

amp = sf.gSamples( amplitude, numberSamples, cut=ampCut, clip=ampClip )


frames = g.multiGaussFrame(coordinates,sigmaWidth, sigmaLength,angles,amp,frameDims)
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
        
        