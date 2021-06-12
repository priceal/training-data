"""
particle analysis module
contains functions for analyzing video data from BLAs, can detect 
peaks and allow user to examine in various ways

v. 2021 02 10

"""

from heapq import heappush, heappop
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.optimize import curve_fit
#import gauss as g
import pandas as pd
import gauss as g

##############################################################################
##############################################################################
#                                                                #############
# VARIOUS USEFUL FUNCTIONS USED IN MODULE                        #############
#                                                                #############
##############################################################################
##############################################################################

def heapsort(iterable):
    """
    this is necessary to sort image file names in input directory
    into ascending sequence of frame numbers. Note, it requires
    that sorting file names into alphanumeric sequence puts them
    into the correct time order.

    Parameters
    ----------
    iterable : iterable
        list or array that is to be sorted

    Returns
    -------
    sorted list : list
        the sorted list
    """
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]

##############################################################################
##############################################################################

def box(center,shape,bx=3,by=3):
    """
    Returns a 4-tuple of corner coordinates of the box surrounding center
    Dimensions of box are 2*bx+1 X 2*by+1. Properly sizes box near edges of
    image so as not to go over edge.

    Parameters
    ----------
    center : tuple/list/array
        coordinates of the center
    shape : tuple/list/array
        dimensions of image
    bx : int, optional
        width of horizontal buffer around center. The default is 3.
    by : TYPE, optional
        width of vertical buffer around center. The default is 3.

    Returns
    -------
    4-tuple
        lower y, upper y, lower x, upper x limits of box

    """

    xi, yi = int(center[0]), int(center[1]) #truncate x,y to get indices
    # return box coords but make sure does not go outside size of image
    return max([0,yi-by]), min([shape[0],yi+by+1]), \
        max([0,xi-bx]), min([shape[1],xi+bx+1]) 

##############################################################################
##############################################################################
#                                                                #############
# THIS SECTIONS CONTAINS FUNCTIONS NEEDED FOR READING IN DATA    #############
#                                                                #############
##############################################################################
##############################################################################
    
def pad0(directory,image='image'):
    """
    needed to rename image files from Frank to put into alphanumeric
    order so heapsort() will work on them. Frank puts frame number
    at front of filename. like this: 034_38728379857.jpg
    CAUTION: this will rename all files in directory!

    Parameters
    ----------
    directory : string
        the directory containing files to rename   
    image : string, optional
       the root name for renamed files. The default is 'image'.

    Returns
    -------
    None.

    """
    # set input directory and get list of files
    input_directory = returnpath(directory)
    dir_list = os.listdir(input_directory)
    
    # loop trough every file in input directory and rename
    for filename in dir_list:
        
        front, ext = os.path.splitext(filename)
        imagenumber = front.split('_')[0] #get frame number
        
        #create the string representation of the frame number
        # assumes < 1000 image files
        if int(imagenumber) < 10:
            numbertag = '00' + imagenumber
        elif int(imagenumber) < 100:
            numbertag = '0' + imagenumber
        else:
            numbertag = imagenumber
            
        # create new file name and rename file
        outputname = image + '_' + numbertag + ext
        os.rename(os.path.join(input_directory,filename),os.path.join(input_directory,outputname))
        
    return 

###############################################################
#################################################################

def loadim(file):
    """
    loads in an image file and returns gray scale openCV image

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    file_path = returnpath(file)
    
    return cv.imread(file_path,0) # read/return gray scale
    
##################################################################
##################################################################

def returnpath(file):
    """
    return absolute path to file give either relative or absolute path

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    file_path : TYPE
        DESCRIPTION.

    """
    
    if file[0] == '/': # determine if absolute path is given
        file_path = file
    else: # must be relative
        cwd = os.getcwd()
        file_path = os.path.join(cwd, file)

    return file_path

###############################################################
#################################################################

def loadimm(directory, rang=(0,-1)):
    """
    load multiple image files and return a list of gray scale openCV images

    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    rang : TYPE, optional
        DESCRIPTION. The default is (0,-1).

    Returns
    -------
    listall : TYPE
        DESCRIPTION.

    """

    initial, final = rang
    # set input directory and get list of files, sort
    input_directory = returnpath(directory)
    dir_list = os.listdir(input_directory)
    dir_list_sorted = heapsort(dir_list)
    
    listall = []
    for file in dir_list_sorted[initial:final]:
        file_path = os.path.join(input_directory,file)
        raw = cv.imread(file_path,0)
        listall.append(raw)
        
    return listall


###############################################################
#################################################################

def loadDir(directory, rang=(0,-1)):
    """
    load multiple image files and return a list of gray scale openCV images

    Parameters
    ----------
    directory : string
        absolute or relative path to image directory
    rang : 2-tuple, optional
        range of images to include. The default is (0,-1).

    Returns
    -------
    dataframe
        ordered paths to each file with index

    """

    initial, final = rang
    # set input directory and get list of files, sort
    input_directory = returnpath(directory)
    dir_list = os.listdir(input_directory)
    dir_list_sorted = heapsort(dir_list)
    return_list = [ os.path.join(input_directory,s) for s in dir_list_sorted ]
        
    directoryDict = { 'path' : return_list }
        
    return pd.DataFrame(directoryDict)

##############################################################################
##############################################################################
#                                                             ################
# THIS SECTIONS CONTAINS FUNCTIONS FOR MANIPULATING IMAGES,   ################
# AND LISTS/ARRAYS OF IMAGES                                  ################
#                                                             ################
##############################################################################
##############################################################################

def thresholdSubtract(image0,image1,thresholdLevel):
    """
    returns the difference between two images, with values below a threshold
    set to zero.

    Parameters
    ----------
    image0, image1 : 2D arrays
        DESCRIPTION.
    thresholdLevel : TYPE
        DESCRIPTION.

    Returns
    -------
    thresholded : TYPE
        image0 - image 1.

    """

    difference = cv.subtract(image0,image1)
    ret,thresholded = cv.threshold(difference,thresholdLevel,255,cv.THRESH_TOZERO)

    return thresholded

##############################################################################
##############################################################################

def backgroundSubtract(image0,image1,thresholdLevel):
    """
    returns the background subtracted image, with pixels where difference is
    less than threshold set to zero.

    Parameters
    ----------
    image0, image1 : 2D arrays
        DESCRIPTION.
    thresholdLevel : TYPE
        DESCRIPTION.

    Returns
    -------
    final : TYPE
        image0 - image 1.

    """

    difference = cv.subtract(image0,image1)
    ret,mask = cv.threshold(difference,thresholdLevel,255,cv.THRESH_BINARY)
    final = cv.bitwise_and(image0,mask)

    return final

##############################################################################
##############################################################################

def particleFilter(image,sigma,dims,norm=1.0):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    dims : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    center =  ( (dims[1]-1)/2 , (dims[0]-1)/2 )
    pkernel = g.gframe(center,sigma,1.0,dims)
    kernel = pkernel - pkernel.mean()
    kernel = norm*kernel/np.clip(kernel,0,np.Inf).sum()
    
    return cv.filter2D(image,cv.CV_16U,kernel)

##############################################################################
##############################################################################
#                                                            #################
# THIS SECTIONS CONTAINS FUNCTIONS FOR CREATING/MODIFYING    #################
# FRAMES, AND LISTS/ARRAYS OF FRAMES                         #################
#                                                            #################
##############################################################################
##############################################################################

def cutframe_file(center,bx=3,by=3,data='.',rang=(0,-1)):
    """
    load & cut out subframe from image files and store in array.
    returns a 3D array, the first dim is the frame number,
    followed by y, x indices: frames[number,y,x]. note: in an array, 
    the row number corresponds to the y coord, and the column corresponds 
    to the x coord. Current directory is assumed to be where image files 
    are located, if no data input is given. There can be no other files 
    in the directory. Images are loaded in alphanumeric order.

    Parameters
    ----------
    center : TYPE
        [x,y], list or array of center of box to cut
    bx : TYPE, optional
        box x buffer parameter, width = 2*bx +1. The default is 3.
    by : TYPE, optional
        DESCRIPTIONbox y buffer parameter, height = 2*by +1. The default is 3.
    data : TYPE, optional
        name of data directory. The default is '.'.
    rang : TYPE, optional
        DESCRIPTION. The default is (0,-1).

    Returns
    -------
    TYPE
        3D array described above

    """

    # set input directory 
    input_directory = returnpath(data)
    dir_list = os.listdir(input_directory)
    dir_list_sorted = heapsort(dir_list)
    
    # read in first image to get image size
    file_path = os.path.join(input_directory,dir_list_sorted[0])
    raw = cv.imread(file_path,0)
    
    # define box around center of frame
    yl,yu,xl,xu = box(center,raw.shape,bx=bx,by=by)

    # initialize variables for loop and begin loop
    numberFrames = len(dir_list_sorted)
    listall, frame, count = [], 0, 0
    for file in dir_list_sorted[rang[0]:rang[1]]:
        
        # ID file to open and append cropped frame to listall
        file_path = os.path.join(input_directory,file)
        raw = cv.imread(file_path,0)
        listall.append(raw[yl:yu,xl:xu].copy())
        
        #update loop variables and output current count
        frame += 1
        count += 1
        if count == 100:
            print( file_path, frame, ' of ', numberFrames )
            count = 0
        
    return np.array(listall)

###############################################################    
###############################################################    
    
def cutframe(images,center,bx=3,by=3):
    """
    load & cut out subframe from list of images
    this is essenially same as cutframe_file, but is used when
    the entire list of images is stored in memory.
    can make routines faster for going through entire data set

    Parameters
    ----------
    images : TYPE
        list or array of images in time order.
    center : TYPE
        [x,y], list or array of center of box to cut.
    bx : TYPE, optional
        box x buffer parameter, width = 2*bx +1. The default is 3.
    by : TYPE, optional
        box y buffer parameter, height = 2*by +1. The default is 3.

    Returns
    -------
    TYPE
        3D array described above in cutframe_file.

    """

    # define box around center of frame
    yl,yu,xl,xu = box(center,images[0].shape,bx=bx,by=by)
        
    # initialize variable for loop and begin loop
    listall = []
    for image in images:
        listall.append(image[yl:yu,xl:xu].copy())
        
    return np.array(listall)

#################################################################
#################################################################

def cropframes(image,xy,bx=3,by=3,imgDF=None):
    """
    cut out subframes from image file and store in array.
    returns a 3D array, the first dim is the frame number,
    followed by y, x indices -- returns a new array, not a view

    Parameters
    ----------
    image : TYPE
        image object.
    xy : TYPE
        list or array of [x,y] positions .
    bx : TYPE, optional
        box size parameter, width = 2*bx +1. The default is 3.
    by : TYPE, optional
        box size parameter, height = 2*by +1. The default is 3.

    Returns
    -------
    TYPE
         a list of 2D arrays described above.

    """
    if isinstance(image,int):  # image index given
        image = loadim( imgDF['path'][image] )
    elif isinstance(image,str):  # image filename or path given
        image = loadim( returnpath(image) )
    else:  # image passed to function
        pass
    
    if isinstance(xy,pd.DataFrame):
        xy = xy[['x','y']].to_numpy()
    
    listall = []
    for center in xy:
        yl,yu,xl,xu = box(center,image.shape,bx=bx,by=by)
        listall.append(image[yl:yu,xl:xu].copy())
    
    return np.array(listall)

#########################################################################
#########################################################################

def background_single(frame):
    """
    background subtract a frame -- returns the corrected frame. Does not act
    on array in place---returns new array.
    
    Parameters
    ----------
    frame : 2D array
        Image that will be background subtracted. It is not modified.

    Returns
    -------
    2D array
        The background subtracted image. 

    """

    n, m = frame.shape
    borderSum = frame.sum() - frame[1:-1,1:-1].sum()
    background_level = np.round(borderSum / 2 / (n+m-2))
    return cv.subtract( frame, background_level ) 

##############################################################################
##############################################################################

def background(frames):
    """
    Backgound subtracts an array of frames, see above. Does not act on 
    frames in place---returns new array.

    Parameters
    ----------
    frames : 3D array
        An array of frames, Slowest (left most) index corresponds to frame
        number

    Returns
    -------
    3D array
        An array of corrected frames, Slowest (left most) index corresponds to frame
        number

    """
    
    listall = []
    for frame in frames:
        listall.append(background_single(frame))
        
    return np.array(listall)

##############################################################################
##############################################################################
#                                                     ########################
# THIS SECTIONS CONTAINS FUNCTIONS CREATING/MODIFYING ########################
# LISTS OF PEAK LOCATIONS                             ########################
#                                                     ########################
##############################################################################
##############################################################################

def inputxy(file):
    """
    reads in xy list of peaks from text file of format:
    (note: this is imagej format from particle analysis)
    peak x  y
    1    x1 y1
    2    x2 y2
    ...
    N    xN yN

    Parameters
    ----------
    file : string
         name of input file

    Returns
    -------
    list
        list of coordinate pairs [[x1,y1],[x2,y2]...[xN,yN]] 

    """
    # set input file
    file_path = returnpath(file)
    temp = np.loadtxt(file_path,skiprows=1) #ignores column title row

    return temp[:,1:] #does not return first column

#################################################################
#################################################################

def subtractxy(xy1,xy2,bx=3,by=3):
    """
    uses set subtraction to remove one set of peaks from a list, approximate
    it does this by picking each peak in xy1, then asking if there exists
    a peak in xy2 which is w/in the buffer distance, if yes, then that
    peak is removed from xy1.  

    Parameters
    ----------
    xy1, xy2 : TYPE
        input lists.
    bx, by : TYPE, optional
        the buffer width, height. The default is 3, 3.

    Returns
    -------
    TYPE
        xy1 - x2, where "-" means set subtractionN.

    """
    
    num1 = len(xy1)
    listall = []
    for n in range(num1):
        one = xy1[n]
        for two in xy2:
            if abs(one[0]-two[0])<bx and abs(one[1]-two[1])<by:
                listall.append(n)
                break
    return np.delete(xy1,listall,axis=0)

########################################################################
########################################################################

def selectxy(xy,lst):
    """
    selects a sub list of peaks from a peak list

    Parameters
    ----------
    xy : TYPE
        list of peaks .
    lst : TYPE
        list of which peaks to select [n1,n2,n3,..].

    Returns
    -------
    TYPE
        returns the selected peaks.

    """
    
    selection = [xy[i] for i in lst]
    
    return np.array(selection)


########################################################################
########################################################################

def countPeaks_old(img,bx=3,by=3,maxnum=1000,minval=0,border=True):
    """
    returns number of peaks found---just runs findpeaks and returns  length
    of list of peaks

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    bx : TYPE, optional
        DESCRIPTION. The default is 3.
    by : TYPE, optional
        DESCRIPTION. The default is 3.
    maxnum : TYPE, optional
        DESCRIPTION. The default is 1000.
    minval : TYPE, optional
        DESCRIPTION. The default is 0.
    border : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    dum1, dum2 = findpeaks(img,bx=bx,by=by,maxnum=maxnum,minval=minval,border=border)

    return len(dum2)               
    
##############################################################################
##############################################################################   
 
def pixelMax(image):
    """
    Uses simpe slow algorithm for finding individual pixels which are greater
    than any adjacent pixel. Returns ALL such pixels.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    n,m = image.shape
    listxy, listvalue= [], []
    for i in range(1,n-1):
        for j in range(1,m-1):
            window = np.copy(image[i-1:i+2,j-1:j+2])
            window[1,1] = 0
#            print(window)
#            print(image[i,j])
            if image[i,j] > window.max():
 #               print(i,j)
                listxy.append([j,i])
                listvalue.append(image[i,j])
    return np.array(listxy), np.array(listvalue)

##############################################################################
##############################################################################
  
  
    
    
def pickxy(image, imgDF = None, n = 100000 ):
    """
    allows use to pick peaks manually, user must enter number of
    peaks that will be picked

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(image,int):  # image index given
        image = loadim( imgDF['path'][image] )
    elif isinstance(image,str):  # image filename or path given
        image = loadim( returnpath(image) )
    else:  # image passed to function
        pass
    
    fig, ax = plt.subplots()
    ax.imshow(image,cmap='gray',interpolation='nearest')
    out = plt.ginput(n=n,timeout=-1,mouse_add=None,mouse_pop=None,mouse_stop=None)
    
    return np.array(out)

##############################################################################
##############################################################################
 
def countPeaks(img,bx=3,by=3,maxnum=1000,minval=0,border=False):
    """
    find peaks in a provided image. image should be a PIL image. 
    finds peaks by locating maximum intensity, then setting surrounding
    box to zero and repeating.

    Parameters
    ----------
    img : TYPE
        PIL image object
    bx : TYPE, optional
       box size parameter, width = 2*bx +1. The default is 3.
    by : TYPE, optional
        box size parameter, height = 2*by +1. The default is 3.
    maxnum : TYPE, optional
        max number of peaks returned. The default is 1000.
    minval : TYPE, optional
        min value of intensity of peaks returned. The default is 0.
    border : TYPE, optional
        Remove border if true--protects against issues arrising from
        picking peaks too near the edge. The default is True.

    Returns
    -------
    integer
        count of all isolated peaks found

    """
    
    # set border around image to zero to eliminate peaks at edge
    if border == True:
        frame = cv.copyMakeBorder(img[by:-by,bx:-bx],by,by,bx,bx,\
                                  cv.BORDER_CONSTANT,0)
    else:
        frame = np.copy(img) 
    
    n = 0; imax = 99999
    while n <= maxnum and imax >= minval :
        
        ymax, xmax = np.unravel_index(np.argmax(frame),frame.shape)
        imax = frame[ymax,xmax]
        yl,yu,xl,xu = box((xmax,ymax),frame.shape,bx=bx,by=by)
        frame[yl:yu,xl:xu]=np.zeros([yu-yl,xu-xl])
 #       print('n, yxmax, height, box', n, (ymax,xmax), h, yl, yu, xl, xu)
        n += 1
    
    return n-1

##############################################################################
##############################################################################

def findPeaks(image, outputDF, imgDF = None, bx=3, by=3, maxnum=1000, \
              minval=0, border=False):
    """
    find peaks in a provided image. image should be a PIL image. 
    finds peaks by locating maximum intensity, then setting surrounding
    box to zero and repeating.

    Parameters
    ----------
    img : TYPE
        PIL image object
    bx : TYPE, optional
       box size parameter, width = 2*bx +1. The default is 3.
    by : TYPE, optional
        box size parameter, height = 2*by +1. The default is 3.
    maxnum : TYPE, optional
        max number of peaks returned. The default is 1000.
    minval : TYPE, optional
        min value of intensity of peaks returned. The default is 0.
    border : TYPE, optional
        Remove border if true--protects against issues arrising from
        picking peaks too near the edge. The default is True.

    Returns
    -------
    2D array
        list of peak coordinates in format [[x1,y1], [x2,y2] ...]

    """
    if isinstance(image,int):  # image index given
        image = loadim( imgDF['path'][image] )
    elif isinstance(image,str):  # image filename or path given
        image = loadim( returnpath(image) )
    else:  # image passed to function
        pass
     
    # set border around image to zero to eliminate peaks at edge
    if border == True:
        frame = cv.copyMakeBorder(image[by:-by,bx:-bx],by,by,bx,bx,\
                                  cv.BORDER_CONSTANT,0)
    else:
        frame = np.copy(image)
    
    n = 0; imax = 99999
    while n <= maxnum and imax >= minval :
        
        ymax, xmax = np.unravel_index(np.argmax(frame),frame.shape)
        imax = frame[ymax,xmax]
        outputDF.loc[n] = [ xmax, ymax, imax ]
        yl,yu,xl,xu = box((xmax,ymax),frame.shape,bx=bx,by=by)
        frame[yl:yu,xl:xu]=np.zeros([yu-yl,xu-xl])
        n += 1
        
    outputDF.drop(len(outputDF)-1,inplace=True) # drop last one
    return 

##############################################################################
##############################################################################

 
##############################################################################
##############################################################################
#                                                             ################
# VARIOUS FUNCTIONS FOR DISPLAYING FRAMES, IMAGES AND DATA    ################
#                                                             ################
##############################################################################
##############################################################################

def showPeaks(image, xy, imgDF=None):
    """
    display an image and overlays peak locations on top
    image can be any image format or NxM array of intensities
    the xy array should be in format of inputxy() above.

    Parameters
    ----------
    image : TYPE
        NxM array of intensities
    xy : TYPE
       list of peaks [[x1,y1],[x2,y2]...[xN,yN]]

    Returns
    -------
    None.

    """
    if isinstance(image,int):  # image index given
        image = loadim( imgDF['path'][image] )
    elif isinstance(image,str):  # image filename or path given
        image = loadim( returnpath(image) )
    else:  # image passed to function
        pass
    
    fig, ax = plt.subplots()
    ax.imshow(image,cmap='gray',interpolation='nearest')
    ax.plot(xy[:,0],xy[:,1],'or', fillstyle='none')
    plt.show()
    
    return

#################################################################
#################################################################

def showframes(frames,delt=0.001):
    """
    makes  movie of an array of frames --- a little buggy

    Parameters
    ----------
    frames : TYPE
        DESCRIPTION.
    delt : TYPE, optional
        DESCRIPTION. The default is 0.001.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()
    count = 0
    for single in frames:
        ax.imshow(single,cmap='gray',interpolation='nearest',vmin=0,vmax=255)
#        plt.title('frame ' + str(count))
        fig.suptitle('frame '+ str(count))
        fig.canvas.draw_idle()
#        print count
        count += 1
        plt.pause(delt)
        ax.cla()
    
    return

#################################################################
#################################################################

def displayframes(frames,n,m,rescale=False,count=0):
    """
    displays an array of frames in a n x m grid
    len(frames) is assumed greater than n*m, but can be less

    Parameters
    ----------
    frames : array
        array of frames
    n : int
        rows.
    m : int
        columns.
    rescale : bool, optional
        sets autoscale. The default is False.

    Returns
    -------
    None.

    """
    
    low, high = 0, 255
    if rescale:
        low = None
        high = None
        
    fig, ax = plt.subplots(n,m)
    ax_flatten = ax.flatten()
    for axs,frame in zip(ax_flatten,frames):
        axs.imshow(frame,cmap='gray',interpolation='nearest',vmin=low,vmax=high)
        axs.text(-4,3,str(count))
 #       axs.set_title(str(count),loc='left',fontdict = {'fontsize':5})
        axs.set_axis_off()
        count += 1
    return

##################################################################
##################################################################

def makeplot(data,xlabel=False,ylabel=False,title=False):
    """
    create plots of tracking and analysis results

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is False.
    ylabel : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    numplots = data.shape[1]
    fig, ax = plt.subplots(numplots,1,sharex='col')
    if title != False:
        fig.suptitle(title)

    for i in range(numplots):
        ax[i].plot(data[:,i],'b.-')
        if ylabel != False:
            ax[i].set_ylabel(ylabel[i])
        
    if xlabel != False:
        ax[-1].set_xlabel(xlabel)
    else:
        ax[-1].set_xlabel('data index')
    
    return fig, ax

##################################################################
##################################################################

def makeplots(data,xlabel=False,ylabel=False,title=False):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    xlabel : TYPE, optional
        DESCRIPTION. The default is False.
    ylabel : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    numtracks, numplots = data.shape[0], data.shape[2]
    fig, ax = plt.subplots(numplots,1,sharex='col')
    if title != False:
        fig.suptitle(title)

    for i in range(numplots):
        for j in range(numtracks):
            ax[i].plot(data[j,:,i],'b-')
        if ylabel != False:
            ax[i].set_ylabel(ylabel[i])
        
    if xlabel != False:
        ax[-1].set_xlabel(xlabel)
    else:
        ax[-1].set_xlabel('data index')
    
    return fig, ax


def showimage(image,title):
    """
    

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plt.imshow(image,cmap='gray',interpolation='nearest')
    plt.title(title)
    plt.show()
    
    return

##############################################################################
##############################################################################
#                                          ###################################
# VARIOUS FUNCTIONS FOR ANALYSIS OF FRAMES ###################################
#                                          ###################################
##############################################################################
##############################################################################

def statframe(inputframe, outputDF, back = False, partNum=0, imageNum=0):
    """
    returns statistics for the peak in a single frame. 
    note that first index is vertical coordinate, and second is horizontal
    the function will try to correct for background with a simple method
    of subtracting the mean value of the border pixels. Then it will
    set negative values to zero.

    Parameters
    ----------
    frame : TYPE
        the image frame = an array
    back : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------

    """
    # correct for backround if asked to
    if back == True:
        frame = background_single(inputframe)
    else:
        frame = np.copy(inputframe)
            
    # the matrices for the moments
    total = frame.sum(dtype=float)
    X, Y = np.meshgrid(  np.arange(frame.shape[1]), np.arange(frame.shape[0]) )
    XX, YY, XY = X*X, Y*Y, X*Y
    
    Xmean = (frame*X).sum(dtype=float)/total
    Ymean = (frame*Y).sum(dtype=float)/total
    XXmean = (frame*XX).sum(dtype=float)/total
    YYmean = (frame*YY).sum(dtype=float)/total
    XYmean = (frame*XY).sum(dtype=float)/total
    
    DX2 = XXmean - Xmean*Xmean
    DY2 = YYmean - Ymean*Ymean
    DXY = XYmean - Xmean*Ymean

    outputDF.loc[len(outputDF)]=[partNum, imageNum, frame.max(), frame.min(), \
                                 frame.sum(), Xmean, Ymean, DX2, DXY, DY2]

    return 

#################################################################
#################################################################

def statframes(frames, outputDF, back = False, indices=False, imageNum=0):
    """
    returns stats on multiple frames. see statframe for details.

    Parameters
    ----------
    frames : TYPE
        DESCRIPTION.
    back : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(indices,bool):
        indices = range(len(frames))
        
    for single,index in zip(frames,indices):
        statframe(single,outputDF,back=back,partNum=index,imageNum=imageNum)
    
    return 

#################################################################
#################################################################

def statframe_(inputframe, back = False):
    """
    legacy version that takes array
    returns statistics for the peak in a single frame. 
    note that first index is vertical coordinate, and second is horizontal
    the function will try to correct for background with a simple method
    of subtracting the mean value of the border pixels. Then it will
    set negative values to zero.

    Parameters
    ----------
    frame : TYPE
        the image frame = an array
    back : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    Xmean : TYPE
        DESCRIPTION.
    Ymean : TYPE
        DESCRIPTION.
    DX2 : TYPE
        DESCRIPTION.
    DXY : TYPE
        DESCRIPTION.
    DY2 : TYPE
        DESCRIPTION.

    """
   
    # correct for backround if asked to
    if back == True:
        frame = background_single(inputframe)
    else:
        frame = np.copy(inputframe)
            
    # the matrices for the moments
    total = frame.sum(dtype=float)
    x, y = np.arange(frame.shape[1]), np.arange(frame.shape[0])
    X, Y = np.meshgrid(x, y)   
    XX, YY, XY = X*X, Y*Y, X*Y
    
    Xmean = (frame*X).sum(dtype=float)/total
    Ymean = (frame*Y).sum(dtype=float)/total
    XXmean = (frame*XX).sum(dtype=float)/total
    YYmean = (frame*YY).sum(dtype=float)/total
    XYmean = (frame*XY).sum(dtype=float)/total
    
    DX2 = XXmean - Xmean*Xmean
    DY2 = YYmean - Ymean*Ymean
    DXY = XYmean - Xmean*Ymean

    return frame.max(), frame.min(), frame.sum(), Xmean, Ymean, DX2, DXY, DY2

#################################################################
#################################################################

def statframes_(frames, back = False):
    """
    returns stats on multiple frames. see statframe for details.
    legacy version that takes array of arrays.

    Parameters
    ----------
    frames : TYPE
        DESCRIPTION.
    back : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    listall = []
    for single in frames:
        listall.append(statframe_(single, back = back))
    
    return np.array(listall)

#################################################################
#################################################################

def pcaframe(inputframe, meani, vlist, back = False):
    """
    returns PCA for the peak in a single frame. 

    Parameters
    ----------
    frame : TYPE
        the image frame = an array, not flattened.
    meani : TYPE
        the mean to subtrace to center, not flattened .
    vlist : TYPE
        list of PCs, should be flattened.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
   # correct for backround if asked to, in which case it is assumed that
   # the meani was calculated with the background corrected frames
    if back == True:
        frame = background_single(inputframe)
    else:
        frame = np.copy(inputframe)
        
    framec = frame - meani
    amplitude = []
    for v in vlist:
        amplitude.append(np.dot(v,framec.flatten()))

    return np.array(amplitude)

#################################################################
#################################################################

def pcaframes(frames, meani, vlist, back = False ):
    """
    returns PCA for the peak in an array of frames, can be used
    for tracking if array is time ordered images of same particle

    Parameters
    ----------
    frames : TYPE
        array of image frames, not flattened.
    meani : TYPE
        the mean to subtrace to center, not flattened .
    vlist : TYPE
        list of PCs, should be flattened.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    listall = []
    for single in frames:
        listall.append(pcaframe(single, meani, vlist, back = back))
    
    return np.array(listall)

###############################################################
#################################################################

def fitframes(frames, back = False):
    """
    returns fist on multiple frames. see fitframe for details.

    Parameters
    ----------
    frames : TYPE
        DESCRIPTION.
    bck : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    listall = []
    for single in frames:
        listall.append(fitframe(single, back = back))
    
    return np.array(listall)

#################################################################            
#################################################################

def fitframe(inputframe, back = False):
    """
    returns best fit parameters for the peak in a single frame. 

    Parameters
    ----------
    frame : TYPE
        the image frame = an array. note that first index is vertical 
        coordinate, and second is horizontal
    back : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    # correct for backround if asked to
    if back == True:
        frame = background_single(inputframe)
    else:
        frame = np.copy(inputframe)

    ynum, xnum = frame.shape  # size in pixels
    xmax, ymax = xnum - 1, ynum - 1   # max indices
    
    # use stats to determine initial values for fit
    Ao, imin, isum, xo, yo, vxx, vxy, vyy = statframe_(frame)
    sigmao = np.sqrt(max(vxx,vyy))
    
    x = np.linspace( 0, xmax,  xnum) # x range = 0 to xmax
    y = np.linspace( 0, ymax,  ynum) # y range = 0 to ymax
    X, Y = np.meshgrid( x, y )       # X,Y are the arrays of x and y coords
    p0 = [xo,yo,sigmao,Ao]        # the initial guesses for parameters
    lowerBounds = [0.0,  0.0, 0.1, 0.001]
    upperBounds = [float(xmax), float(ymax), float(xmax), 9999.9] 
    err = 0
    
    try:
        # sends 2-tuple of list of x and y coords to _gaussian
        popt, pcov = curve_fit(g._gaussian, [X.ravel(),Y.ravel()], frame.ravel(), p0=p0, bounds=(lowerBounds,upperBounds))
        fit = g.gaussian(X,Y,popt[0],popt[1],popt[2],popt[3])
 
    except:
#        print("Error - fitframe failed")
        err = 1
        popt = np.array([xo,yo,sigmao,Ao])
        pcov = 0

    result = np.concatenate([popt,np.array([err])])
    
    return result

#################################################################      
###############################################################


#################################################################
###############################################################
# run through output of integration and return lifetime of each
# particle, using threshold method
#
#################################################################

def threshold(data, threshold=100):

    duration = len(data)   # number of frames
    frame = 0
    while (data.iloc[frame,1] > threshold) & (data.iloc[frame,0] < duration-1):
        frame = frame + 1

    return frame







