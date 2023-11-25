

import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd





def SaliencyMapFilt(Fixies,SD=25,Ind=0):
    ''' Gaussian filter of fixations counts, Ind=1 for individual, Ind=0 for group '''
    if Ind==0:
        Smap=ndimage.filters.gaussian_filter(np.mean(Fixies,0),SD)
    else:
        Smap=ndimage.filters.gaussian_filter(Fixies,SD)
    return Smap


def SaccadesTrial(TrialX,TrialY):
    ''' transform 2 arrays of fixations x-y positions, into approximate saccaddes
    with start and end locations '''
    StartTrialX=TrialX[0:-1]
    StartTrialY=TrialY[0:-1]     
    EndTrialX=TrialX[1:]
    EndTrialY=TrialY[1:]
    return StartTrialX,StartTrialY,EndTrialX,EndTrialY




def ScanpathL(x,y):
    ''' input 2 arrays for x and y ordered fixations
    output 1: average amplitude of  saccacdes
    output 2: total  length of scanpath'''
    x1,y1,x2,y2=SaccadesTrial(x,y)
    lengths=np.sqrt((x2-x1)**2+(y2-y1)**2)
    return np.mean(lengths),np.sum(lengths)



def StatEntropy(StatP): 
    """Calculate entropy of probability distribution
    without nans, result should be the same as scipy.stats.entropy with base=2"""
    LogP=np.log2(StatP)   
    LogP[np.isfinite(LogP)==0]=0   # replace nans with zeros    
    return -np.sum(StatP*LogP)


