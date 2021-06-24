
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal
"""

runfile('initialize.py', current_namespace=True)

testSetFile = 'nearMissData.pkl'

regModel = reg

##############################################################################
##############################################################################
# load in test set
with open(testSetFile, 'rb') as file:
    (frames,classification,coords) = pickle.load(file)
    
# read parameters from training set dimenstions
numberSamples, yDim, xDim = frames.shape
print('{} frames, {} x {}'.format(numberSamples,yDim,xDim))

# pre-processing of frames
scaled_frames = frames/frames.max()

# reshaping of frames and loading into X, Y arrays for training
X = scaled_frames.reshape( (numberSamples, yDim*xDim) )
Y = classification

# define regression model and train!
m = len(Y)
Xpredict = regModel.predict(X)
predHits = Xpredict.sum()
predMisses = m - predHits
actualHits = Y.sum()
actualMisses = numberSamples - actualHits

#truePositives = (Xpredict & Y).sum()
#falsePositives = (Xpredict & ~Y).sum()
#trueNegatives = (~Xpredict & ~Y).sum()
#falseNegatives = (~Xpredict & Y).sum()

cf = confusion_matrix(Y,Xpredict)
truePositives = cf[1,1]
falsePositives = cf[0,1]
trueNegatives = cf[0,0]
falseNegatives = cf[1,0]
negatives = trueNegatives + falseNegatives
positives = truePositives + falsePositives

print('')
print('test set contains {} TRUE and {} FALSE'.format(actualHits,actualMisses))
print('predictions contain {} POSITIVE and {} NEGATIVE'.format(predHits,predMisses))
print('')

print('recall {:2.1f}%'.format(100*truePositives/actualHits) )
print('precision {:2.1f}%'.format(100*truePositives/(truePositives+falsePositives)) )

print('')
print('           CONFUSION MATRIX')
print('           FALSE      TRUE       TOTAL')
print('NEGATIVE   {:<10} {:<10} {:<10}'.format(trueNegatives,falseNegatives,negatives))
print('POSITIVE   {:<10} {:<10} {:<10}'.format(falsePositives,truePositives,positives))
print('TOTAL      {:<10} {:<10} {:<10}'.format(actualMisses,actualHits,m))



# plot points colored red/blue if predictid true/false
clrs = [ 'red' if cl else 'blue' for cl in Xpredict[:7500] ]
plt.figure()
ax = plt.subplot()
ax.set_aspect('equal')
ax.grid()
ax.scatter(coords[:7500,1],coords[:7500,0],c=clrs)

