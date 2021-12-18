#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 08:25:46 2021

@author: allen
"""

import numpy as np

def gSamples( dp, numberSamples, cut=(-np.Inf,np.Inf), clip=(-np.Inf,np.Inf) ):
    
    meanValue, sigmaValue = dp
    lowCutOff, hiCutOff = cut
    lowClip, hiClip = clip

    sampleList = []
    for i in range(numberSamples):
        sample = np.random.normal( meanValue, sigmaValue )
        while sample > hiCutOff or sample < lowCutOff:
            sample = np.random.normal( meanValue, sigmaValue )
        sampleList.append( sample )
    samples = np.array(sampleList)

    return np.clip(samples,lowClip,hiClip)
    
    