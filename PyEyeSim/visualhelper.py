
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
import matplotlib.ticker as ticker
from math import atan2, degrees
from matplotlib.patches import Ellipse



def MeanPlot(N,Y,yLab=0,xtickL=0,color='darkred',label=None,ax=0):
    ''' expects data format: row- subjects column-stimuli '''
    if type(ax)==int:
        fig,ax=plt.subplots(figsize=(N/2,5))
    ax.errorbar(np.arange(N),np.nanmean(Y,0),stats.sem(Y,0,nan_policy="omit")*2,linestyle='None',marker='o',color=color,label=label)
    if type(xtickL)!=int:
        ax.set_xticks(np.arange(N),labels=xtickL,fontsize=9,rotation=60)
    ax.set_xlabel('Stimulus',fontsize=14)
    ax.set_ylabel(yLab,fontsize=14)
    return None


def VisBinnedProg(bins,Y,ylabel,col='navy',label='',axin=0):
    if type(axin)==int:
        fig,axin=plt.subplots()
    axin.errorbar((bins[0:-1]+bins[1:])/2,np.nanmean(Y,0),stats.sem(Y,0,nan_policy='omit'),color=col,linewidth=2,label=label)
    axin.set_xlabel('time (ms)')
    axin.yaxis.set_major_locator(ticker.MaxNLocator(5))
    axin.set_ylabel(ylabel)
    return axin



def PlotDurProg(nmax,Y,error,label=''):
    plt.fill_between(np.arange(1,nmax+1),Y-error,Y+error,alpha=.5)
    plt.plot(np.arange(1,nmax+1),Y,label=label)
    plt.xticks(np.intp(np.linspace(1,nmax,5)),np.intp(np.linspace(1,nmax,5)))
    plt.xlabel('fixation number',fontsize=14)
    plt.ylabel('fixation duration (ms)',fontsize=14)
    return 



def JointBinnedPlot(bins,y1,y2,col1='olive',col2='orange',ylabel1='',ylabel2=''):
    fig,ax1=plt.subplots(figsize=(5,3))
    ax1.errorbar((bins[0:-1]+bins[1:])/2,np.nanmean(y1,0),stats.sem(y1,0,nan_policy='omit'),color=col1,linewidth=2)
    ax1.set_xlabel('time (ms)')
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.set_ylabel(ylabel1,color=col1)
    ax1.tick_params(axis='y', labelcolor=col1)

    ax2 = ax1.twinx() 
    ax2.set_ylabel(ylabel2,color=col2)
    ax2.errorbar((bins[0:-1]+bins[1:])/2,np.nanmean(y2,0),stats.sem(y2,0,nan_policy='omit'),color=col2,linewidth=2)
    ax2.tick_params(axis='y', labelcolor=col2)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))

    return ax1,ax2