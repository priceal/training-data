#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 08:25:46 2021

@author: allen
"""

numberSamples = 10000

parameters = (100.0,  30.0)     # (mean, sigma)
cutOffs = (40.0, 170.0)          # (low, high)
clips = (50.0, 150.0)           # (low, high) 


#samples = gSamples( parameters, numberSamples, cut=cutOffs, clip=clips)
samples = gSamples( parameters, numberSamples , clip=clips)

plt.figure()
plt.hist( samples, bins = 60 )