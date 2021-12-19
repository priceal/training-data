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

length = (1.25, 1.5)             # (mean, sigma)
lengthCut = (0.75, 1.75)      # (low, high)
lengthClip = (-np.Inf, np.Inf)      # (low, high)

ecc = (1.0, 1.0)              # (mean, sigma)
eccCut = (1.0, 3.0)           # (low, high)
eccClip = (1.0, np.Inf)      # (low, high)

theta = (0, 2.0*np.pi)              # (mean, sigma)

# define parameters
frameDims = (9,9)   # dimensions of frame
xSpan = (2.5,5.5)   # x range of peak positions
ySpan = (2.5,5.5)   # y range
numberSamples = 10000

# hit region --- where value is TRUE
xrange = [3.5,4.5]
yrange = [3.5,4.5]

# define noise characteristics if wanted
noiseMean = 20
noiseSigma = 10.0
addNoise = True

# define data file name if you want to save it
saveData  = True
if saveData:
    saveFileName = 'test.pkl'

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

header = 'header,frames,coordinates,sigmaWidth,sigmaLength,angles,amp'
if saveData:
    with open(saveFileName,'wb') as file:
#        pickle.dump((header,frames,coordinates,sigmaWidth,sigmaLength,angles,amp),file)
        pickle.dump( (frames, xyclass), file )
        
        