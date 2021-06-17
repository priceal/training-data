# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:01:40 2021

@author: priceal
"""

import scipy.special as ss
import numpy as np
import pylab as plt
import simDataFunctions as sdf
import random as rd


###############################################################
###################################################################
def rand2D(xrang = [0,1],yrang = [0,1] ):
    """
    generates a random 2D vector

    Parameters
    ----------
    xrang : 2-tuple, optional
        the x and y ranges. The default is [0,1].
    yrang : 2-tuple, optional
        the x and y ranges. The default is [0,1].

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    xl = xrang[1]-xrang[0]
    yl = yrang[1]-yrang[0]
    
    return xrang[0]+rd.random()*xl, yrang[0]+rd.random()*yl

###############################################################
###################################################################
def multiRand2D(n, xr = [0,1], yr = [0,1] ):
    """
    generates n random 2D vectors

    Parameters
    ----------
    n : TYPE
        the number of vectors
    xr : 2-tuple, optional
        the x and y ranges. The default is [0,1].
    yr : 2-tuple, optional
        the x and y ranges. The default is [0,1].

    Returns
    -------
    TYPE
        the output is an array n X 2

    """

    result = [rand2D( xrang = xr,yrang = yr ) for i in range(n)]
    return np.array(result)

##############################################################
###################################################################
def grid2D(n, xr = [0,1], yr = [0,1] ):
    """
    generates a grid of 2D vectors

    Parameters
    ----------
    n : TYPE
        [nx, ny] = grid dimensions
    xr : 2-tuple, optional
        the x and y ranges. The default is [0,1].
    yr : 2-tuple, optional
        the x and y ranges. The default is [0,1].

    Returns
    -------
    TYPE
        the output is an array n X 2

    """

    x = np.linspace(xr[0],xr[1],n[0])
    y = np.linspace(yr[0],yr[1],n[1])
    
    X, Y = np.meshgrid(x,y)
    return np.array([X.flatten(), Y.flatten()]).transpose()

###############################################################
###################################################################
def plot2D(data):
    """
    displays an array of 2D points
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots()
    ax.plot(data.T[0],data.T[1],'o')

    return

###############################################################
###################################################################
def makeRandFrame(sigma,dim,mean=0.0):
    """
    returns a frame with a Gaussian distributed random intensity

    Parameters
    ----------
    sigma : TYPE
        the sigma value
    dim : TYPE
        a 2-tuple of the dimensions of the frame
    mean : TYPE, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    frame : TYPE
        DESCRIPTION.

    """
    
    frame = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            frame[i,j] = rd.gauss(mean,sigma)
            
    return frame

###############################################################         
###################################################################
def GaussFrame(center,sig,I,dim):
    """
    returns a frame with a gaussian intensity

    Parameters
    ----------
    center : TYPE
        a 2-tuple of x,y coords of center
    sig : TYPE
        the sigma value
    I : TYPE
        the integrated intensity of the frame
    dim : TYPE
        a 2-tuple of the dimensions of the frame

    Returns
    -------
    frame : TYPE
        DESCRIPTION.

    """
    
    frame = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            frame[i,j] = Gaussian((i,j),center,sig,1.0)
            
    frame = frame * I / frame.sum()
            
    return frame

###############################################################
###################################################################
def multiGaussFrame(centers,sig,I,dim):
    """
    returns am array of frames with a gaussian intensity at
    a list of centers, all with same A and sig

    Parameters
    ----------
    centers : TYPE
        a list of 2-tuples of x,y coords of center
    sig : TYPE
        the sigma value
    I : TYPE
        DESCRIPTION.
    dim : TYPE
        a 2-tuple of the dimensions of the frame

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    result = []
    for center in centers:
        result.append(GaussFrame(center,sig,I,dim))
    return np.array(result)
