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

# particle amplitude parameters
amplitude = (200.0, 50.0)    # (mean, sigma)
ampCut = (70.0, 500.0)     # (low, high)
ampClip = (-np.Inf, np.Inf)      # (low, high)

# particle size and eccentricity
length = (0.7, 0.1)             # (mean, sigma)
lengthCut = (0.5, 1.1)      # (low, high)
lengthClip = (-np.Inf, np.Inf)      # (low, high)
ecc = (1.0, 0.3)              # (mean, sigma)
eccCut = (1.0, 1.3)           # (low, high)
eccClip = (1.0, np.Inf)      # (low, high)

# angular spread---a uniform distribution
theta = (0, 2.0*np.pi)              # (theta min, theta max)

# define parameters for size of frames and distribution of positions (uniform)
frameDims = (7,7)   # dimensions of frame
xSpan = (-3,9)   # x range of peak positions
ySpan = (-3,9)   # y range of peak positions
xExclude = ( 2.5,3.5)      # x range to exclude (can exclude some rectangles)
yExclude = ( 2.5,3.5)      # x range to exclude (only if [0][<[1])
numberSamples = 2500

# hit region --- where value is TRUE
xrange = [2.5,3.5]
yrange = [2.5,3.5]

# define noise characteristics if wanted
noiseMean = 20
noiseSigma = 5.0
addNoise = True

# define data file name if you want to save it
saveData  = True
if saveData:
    saveFileName = 'FALSE_2022_01_11.pkl'

##############################################################################
##############################################################################
# setup training input frames
coordinates = np.zeros((numberSamples,2))
for n in range(numberSamples):
    excluded = True
    while excluded:
        nx, ny = g.rand2D(xrang=xSpan,yrang=ySpan)
        excluded = nx<xExclude[1] and nx>xExclude[0] and \
                   ny<yExclude[1] and ny>yExclude[0]
    coordinates[n,0] = ny
    coordinates[n,1] = nx

sigmaLength = sf.gSamples( length, numberSamples, cut=lengthCut, clip=lengthClip)
eccentricities = sf.gSamples( ecc, numberSamples, cut=eccCut, clip=eccClip)
sigmaWidth = sigmaLength / eccentricities
angles = np.random.uniform( theta[0], theta[1], numberSamples)
amp = sf.gSamples( amplitude, numberSamples, cut=ampCut, clip=ampClip )
frames = g.multiGaussFrame(coordinates,sigmaWidth, sigmaLength,angles,amp,frameDims)

if addNoise:
    for frame in frames:
        frame += g.randFrame(noiseSigma,frameDims,mean=noiseMean)

frames = np.clip(frames,0,255)
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

header = 'header,(coordinates,sigmaWidth,sigmaLength,angles,amp),frames,class'
parameters = (coordinates,sigmaWidth,sigmaLength,angles,amp)
if saveData:
    with open(saveFileName,'wb') as file:
#        pickle.dump((header,frames,coordinates,sigmaWidth,sigmaLength,angles,amp),file)
        pickle.dump( (header,parameters, frames, xyclass), file )
        
        