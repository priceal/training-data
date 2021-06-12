"""

generates simulated data for tracking

v. 2021 01 27

"""

#import random as rd
#import os
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
#import cPickle as pickle

#################################################################



def plotAiry(xmin,xmax,numpoint):
    x = np.linspace(xmin,xmax,numpoint)
    plt.plot(x,AiryDisk(x))
    return
    
def plotFunc(xmin,xmax,numpoint,func):
    x = np.linspace(xmin,xmax,numpoint)
    plt.plot(x,func(x))
    return

def plotFunc2D(xr,yr,numpoint,func):
    x = np.linspace(xr[0],xr[1],numpoint)
    y = np.linspace(yr[0],yr[1],numpoint)
    value = np.zeros((numpoint,numpoint))
    for i in range(len(x)):
        for j in range(len(y)):
            value[i,j] = func( (x[i], y[j]) )
    plt.imshow(value)
    return




def AiryDisk(x):
    return 4.0*(ss.j1(x)/x)**2

def square(x):
    return np.heaviside(1.0-np.abs(x),1.0)

def circle(r):
    r = np.array(r)
    rr = (r*r).sum()
    return square(rr)


#################################################################      
###############################################################

def Gaussian( r, ro, sigma, A ):
    """
    The 2D Gaussian which is the basis of the fitting routine
    returns an array which has same shape as input x and y

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    ro : TYPE
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
    r = np.array(r)
    ro = np.array(ro)
    
    return A * np.exp( -0.5*((r-ro)*(r-ro)).sum()/sigma/sigma )


#################################################################      
###############################################################

def MultiGaussian( r, ro, sigma, A ):
    """
    The 2D Gaussian which is the basis of the fitting routine
    returns an array which has same shape as input x and y

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    ro : TYPE
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
    r = np.array(r)
    ro = np.array(ro)
    
    total = 0
    
    for p in zip(ro,sigma,A):
        total += p[2] * np.exp( -0.5*((r-p[0])*(r-p[0])).sum()/p[1]/p[1] )

    return total


#####################################################################
#####################################################################

def makeGaussian( ro, sigma, A ):
    """
    returns a function which is a Gaussian centered at ro with given
    values of sigma and A (amplitude)

    Parameters
    ----------
    ro : TYPE
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

    return lambda r: Gaussian( r, ro, sigma, A )         
 
def makeMultGaussian( ro, sigma, A ):    
 
 
    return lambda r: MultiGaussian( r, ro, sigma, A )  
 
    
#####################################################################
#####################################################################
           
def cosineBG( r, ro, lam, A ):
 
    
    r = np.array(r)
    ro = np.array(ro)
    lam = np.array(lam)
    
    return 0.5*A*( 1.0+np.cos( (2.0*np.pi*(r-ro)/lam).sum() ) )    
    
#####################################################################
#####################################################################


def linearBackground( r, ro, A, B):
    """
    calculate a linear background value, given the defined parameters
    background = B + vec(A).( vec(r) - vec(ro) )

    Parameters
    ----------
    r : 2d array
        the point at which the function is evaluated
    ro : 2d array
        the "center" of the background, where the value = B, intended to
        be the center of the frame
    A : 2d array
        the background vector, direction gives direction of increase
        in the linear background, and the magnitude = slope
    B : TYPE
        value of background at ro

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    r = np.array(r)
    ro = np.array(ro)
    A = np.array(A)
    
    return B + np.dot(A,r-ro)

def makeLinearBackground( ro, A, B ): 
    return lambda r: linearBackground( r, ro, A, B )

def makeCosineBG( ro, lam, A ):
    
    
    return lambda r: cosineBG( r, ro, lam ,A )



def returnFrame( fxn, dims ):
    
    print(fxn((0,0)))
    value = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            value[i,j] = fxn((i,j))
            print(i,j)
            
    return value


