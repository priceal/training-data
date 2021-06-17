#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:08:24 2021

@author: allen

Loads and plots a set of coordinates with labels and plots these, coloring
the different labels. Expects labels to be 0, 1 and 3.

Input file should be a tuple of arrays (coordinate, labels)


"""
runfile('initialize.py', current_namespace=True)

xyFile = "expandedXY.pkl"

##############################################################################

# load XY array from pick particles (integerXY)

with open(xyFile,'rb') as file:
    XY , XY_class = pickle.load(file)

numHits = (XY_class == 0).sum()
numNearMisses = (XY_class == 1).sum()
numBackground = (XY_class == 2).sum()

print('loaded pixel coordinate data file ' + xyFile )

print(numHits," hits.")
print(numNearMisses, " near misses.")
print(numBackground, " background")
print( len(XY), " total.")

plt.figure()
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(XY[:,0],XY[:,1],c=XY_class,marker='.')

