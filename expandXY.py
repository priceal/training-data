#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:08:24 2021

@author: allen

Expands a given array of pixel coordinates by adding displaced and randomly
chosen locations. Removes repeated coordinate pairs and also correclty labels
resulting points as 0:hits, 1:near misses and 2:background.

Stores results in pkl file as a tuple of arrays (coordinate, labels)

"""
runfile('initialize.py', current_namespace=True)

xyFile = "/home/allen/projects/training-data/reviewedXY_07-1.npy"
span = 5
dims = (1920,1200)
nseed = 202264721

# define data file name if you want to save it
saveData  = True
if saveData:
    saveFileName = 'expandedXY.pkl'
    
##############################################################################
np.random.seed(nseed)

# load XY array from pick particles (integerXY)
XY  = np.load(xyFile)
numHits = len(XY)
print('loaded pixel coordinate data file ' + xyFile +' with {} points.'.format(numHits))

# Randomly add values between +/- span to every x and y value in data set
dXY = XY.copy()
for coord in dXY:
    xOffset, yOffset = 0, 0
    while xOffset == 0 & yOffset == 0:
        xOffset = np.random.randint(-span, span)
        yOffset = np.random.randint(-span, span)
    coord[0] += xOffset
    coord[1] += yOffset
print('Generated {} displaced coordinates.'.format(numHits))
    
# now remove repeats from displaced XY
udXY = np.unique(dXY,axis=0)
print("... {} unique remaining after removing duplicates".format(len(udXY)))

# now remove any displaced that are same as hits
mask = np.ones(len(udXY),dtype=bool)
for value in XY:
    tempmask = udXY != value
    tempmask2 = tempmask[:,0] | tempmask[:,1]
    mask = mask & tempmask2
udXY_final = udXY[mask]
numNearMisses = len(udXY_final)
print("... {} remaining after pruning hits from displaced coordinates.".format(numNearMisses))

# put together hits and near misses 
expXY = np.vstack((XY,udXY_final))

# generate random coordinates and remove repeats
rXY = sd.multiRand2D(numHits, xr = [0,dims[0]], yr = [0,dims[1]] ).astype(int)
urXY = np.unique(rXY,axis=0)
print('Generated {} random coordinates.'.format(numHits))
print("... {} unique remaining after removing duplicates".format(len(urXY)))

# now remove any random that are same as hits or displaced
mask = np.ones(len(urXY),dtype=bool)
for value in expXY:
    tempmask = urXY != value
    tempmask2 = tempmask[:,0] | tempmask[:,1]
    mask = mask & tempmask2
urXY_final = urXY[mask]
numBackground = len(urXY_final)
print("... {} remaining after pruning hits/displaced from random coordinates.".format(numBackground))

# put all together & output results
expXY_final = np.vstack((expXY,urXY_final))
expXY_class = np.array([0]*numHits + [1]*numNearMisses + [2]*numBackground)
print('final counts:')
print(numHits," hits.")
print(numNearMisses, " near misses.")
print(numBackground, " background")
print( len(expXY_final), " total.")
plt.plot(XY[:,0],XY[:,1],'b.')
plt.plot(udXY_final[:,0],udXY_final[:,1],'g.')
plt.plot(urXY_final[:,0],urXY_final[:,1],'r.')

if saveData:
    with open(saveFileName,'wb') as file:
        pickle.dump((expXY_final,expXY_class),file)
        
     
        