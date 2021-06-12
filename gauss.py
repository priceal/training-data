"""

generates simulated data for tracking

v. 2021 01 27

"""

import random as rd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#import cPickle as pickle

#################################################################

###############################################################
# generates a random 2D vector
#
# xrang, yrang           the x and y ranges
# 
# the output is a 2-tuple
###################################################################
def rand2D(xrang = [0,1],yrang = [0,1] ):
    
    xl = xrang[1]-xrang[0]
    yl = yrang[1]-yrang[0]
    
    return xrang[0]+rd.random()*xl, yrang[0]+rd.random()*yl

###############################################################
# generates n random 2D vectors
#
# n               the number of vectors
# xr, yr          the x and y ranges
# 
# the output is an array n X 2
###################################################################
def multiRand2D(n, xr = [0,1], yr = [0,1] ):

    result = [rand2D( xrang = xr,yrang = yr ) for i in range(n)]
    return np.array(result)

##############################################################
# generates a grid of 2D vectors
#
# n               [nx, ny] = grid dimensions
# xr, yr          the x and y ranges
# 
# the output is an array n X 2
###################################################################
def grid2D(n, xr = [0,1], yr = [0,1] ):

    x = np.linspace(xr[0],xr[1],n[0])
    y = np.linspace(yr[0],yr[1],n[1])
    
    X, Y = np.meshgrid(x,y)
    return np.array([X.flatten(), Y.flatten()]).transpose()

###############################################################
# displays an array of 2D points
#
###################################################################
def plot2D(data):
    
    fig, ax = plt.subplots()
    for values in data:
        ax.plot(values)

    return

###############################################################
# returns a frame with a Gaussian distributed random intensity
#
# center         a 2-tuple of x,y coords of center
# sig          the sigma value
# I           the integrated intensity of the frame
# dim          a 2-tuple of the dimensions of the frame
#
###################################################################
def randFrame(sigma,dim,mean=0.0):
    
    frame = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            frame[i,j] = rd.gauss(mean,sigma)
            
    return frame

###############################################################
# returns a frame with a gaussian intensity
#
# center         a 2-tuple of x,y coords of center
# sig          the sigma value
# I           the integrated intensity of the frame
# dim          a 2-tuple of the dimensions of the frame
#
###################################################################
def GaussFrame(center,sig,I,dim):
    
    frame = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            frame[i,j] = Gaussian(i,j,center[0],center[1],sig,1.0)
            
    frame = frame * I 
            
    return frame

###############################################################
# returns am array of frames with a gaussian intensity at
# a list of centers, all with same A and sig
#
# centers         a list of 2-tuples of x,y coords of center
# sig          the sigma value
# A           the amplitude
# dim          a 2-tuple of the dimensions of the frame
#
###################################################################
def multiGaussFrame(centers,sig,I,dim):
    
    result = []
    for center in centers:
        result.append(GaussFrame(center,sig,I,dim))
    return np.array(result)


def plotrack(track,t0=0,t1=0):
    
    if t1==0:
        t1=len(track)
    lbl = ['x','y','sig','amp']
    fig, ax = plt.subplots(5,1)
    for n in [0,1,2,3]:
        ax[n].plot(track[t0:t1,n])
        ax[n].set_ylabel(lbl[n])

    return

#################################################################      
###############################################################

def Gaussian(x, y, x0, y0, sigma, A):
    """
    The 2D Gaussian which is the basis of the fitting routine
    returns an array which has same shape as input x and y

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.
    y0 : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return A * np.exp( -0.5*( (x-x0)**2 + (y-y0)**2 )/(sigma**2) )

###############################################################
###################################################################

def _Gaussian(xy, x0, y0, sigma, A ):
    """
    The raveled 2D Gaussian which is the basis of the fitting routine
    this version is for fitting only xo, yo and A
    the sigma is passed in params[]

    Parameters
    ----------
    xy : TYPE
        [X,Y] 2-tuple of X, Y. X & Y should be 2D arrays from meshgrid.
    x0,y0 : TYPE
        the center of distribution.
    sigma : TYPE
        sigma.
    A : TYPE
        amplitude.

    Returns
    -------
    TYPE
        the output is a 1D array.

    """
    x,y = xy
    arr = Gaussian( x, y, x0, y0, sigma, A )
    return arr.ravel()

###############################################################


       
            
            
    
    
