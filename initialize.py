# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:52:17 2021

@author: priceal
"""

print('testing name definitions for import...')

try:
    np
except NameError:
    print(' importing numpy as np.')
    import  numpy as np
    
try:
    plt
except NameError:
    print(' importing pylab as plt.')
    import pylab as plt
    
try:
    g
except NameError:
    print(' importing gauss as g.')
    import gauss as g
    
try:
    pickle
except NameError:
    print(' importing pickle.')
    import pickle

try:
    pa
except NameError:
    print(' importing particleAnalysis as pa.')
    import particleAnalysis as pa

try:
    pd
except NameError:
    print(' importing pandas as pd.')
    import pandas as pd

try:
    linear_model
except NameError:
    print(' importing linear_model from sklearn.')
    from sklearn import linear_model
   
try:
    sd
except NameError:
    print(' importing simdata as sd.')
    import simdata as sd
    
try:
    confusion_matrix
except NameError:
    print(' importing confusion_matrix.')
    from sklearn.metrics import confusion_matrix

    
    