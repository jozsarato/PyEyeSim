import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
import xarray as xr
import matplotlib.ticker as ticker
from math import atan2, degrees

from matplotlib.patches import Ellipse
#%%
from .visualhelper import draw_ellipse

def MySaccadeVis(self,ax,XYdat,lengths,title='',alpha=1):
    ''' saccade visualization, on input ax, based on combined data 2d array, and lengths 1d array'''
    ax.set_title(title)
    ax.set_xlim([0,self.x_size])
    ax.set_ylim([self.y_size,0])
    ax.scatter(XYdat[:,0],XYdat[:,1],c=np.arange(len(XYdat[:,0])),alpha=alpha)
    Idxs=np.cumsum(lengths)
    for ci in range(len(lengths)):
        if ci==0:
            start=0
        else:
            start=Idxs[ci-1]
        ax.plot(XYdat[start:Idxs[ci],0],XYdat[start:Idxs[ci],1],alpha=alpha)
    return

def VisLOOHMM(self,Dat,lengths,ScoresLOO,nshow=3,title='',showim=False,stimname=0):
    ''' visualize least and most typical sequences, from above fitted HMM'''
    Sorted=np.argsort(ScoresLOO)
    fig,ax=plt.subplots(ncols=nshow,nrows=2,figsize=(10,7))
    for a in range(nshow):
        if showim==True:
            ax[0,a].imshow(self.images[stimname],cmap='gray')
            ax[1,a].imshow(self.images[stimname],cmap='gray')
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,5,vis=0,rand=0,totest=Sorted[a])
        self.MySaccadeVis(ax[1,a],DatTest,lenTest,title='max'+str(a)+' logL: '+str(np.round(ScoresLOO[Sorted[a]],2)))
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,5,vis=0,rand=0,totest=Sorted[-a-1])
        self.MySaccadeVis(ax[0,a],DatTest,lenTest,title='min'+str(a)+' logL: '+str(np.round(ScoresLOO[Sorted[-a-1]],2)))
    plt.suptitle(title)
    plt.tight_layout() 

def VisHMM(self,dat,hmmfitted,ax=0,showim=1,stim=0,lengths=0,incol=False):
    ''' visualize fixations and fitted hidden markov model
    dat: fixations
    hmmfitted: fitted hidden markov model
    ax: if not provided, new figure opens '''
    
    colors=['k','gray','salmon','olive','m','c','g','y','navy','orange','darkred','r','darkgreen','k','gray','salmon','olive','y','m','g','c']
    if type(ax)==int:
       fig,ax= plt.subplots()
    if showim:
        ax.imshow(self.images[stim])
        alph=.5
    else:
        alph=.2
    if np.shape(dat)[0]>200:
        alph=.1

    preds=hmmfitted.predict(dat,lengths)

    ax.scatter(hmmfitted.means_[:,0],hmmfitted.means_[:,1],color='darkred',s=50)
    
    for c1 in range(hmmfitted.n_components):
        if incol:
            color1=colors[c1]  # color for scatter 
            color2=colors[c1] # color for patch

        else:
            color1='k'
            color2='olive'

        ax.scatter(dat[preds==c1,0],dat[preds==c1,1],color=color1,alpha=alph)

        draw_ellipse((hmmfitted.means_[c1,0],hmmfitted.means_[c1,1]),hmmfitted.covars_[c1],ax=ax,facecolor='none',edgecolor=color2,linewidth=2)
        for c2 in range(hmmfitted.n_components):
            if c1!=c2:
                ax.plot([hmmfitted.means_[c1,0],hmmfitted.means_[c2,0]],[hmmfitted.means_[c1,1],hmmfitted.means_[c2,1]],linewidth=hmmfitted.transmat_[c1,c2]*5,color='r')
    ax.set_ylim([self.y_size,0])
    ax.set_xlim([0,self.x_size])
    ax.set_yticks([])
    ax.set_xticks([])   



def VisScanPath(self,stimn,ax=False,alpha=.5,allS=True,col='salmon',visFix=False):
    ''' 
    stimn: stimulus index
    ax: if not provided, makes new figure
    alpha: transparency
    allS:  if not provided all participants, otherise it is a number/index of a participant
    col: Color, default color is salmon
    VisFix: visualize fixations

    '''

    if ax==False:
        fig,ax=plt.subplots()
    ax.imshow(self.images[self.stimuli[stimn]])  
    if type(allS)==bool:
        for cs in range(self.ns):
            fixx,fixy=self.GetFixationData(self.subjects[cs],self.stimuli[stimn])
            ax.plot(fixx,fixy,alpha=alpha,color=col)
            if visFix:
                ax.scatter(fixx,fixy,color='gray',alpha=alpha)
        
    else:
        fixx,fixy=self.GetFixationData(self.subjects[allS],self.stimuli[stimn])
        ax.plot(fixx,fixy,alpha=alpha,color=col)
        if visFix:
            ax.scatter(fixx,fixy,color='gray',alpha=alpha)
    ax.set_xlim([0,self.x_size])
        
    ax.set_ylim([self.y_size,0])
    ax.set_xticks([])
    ax.set_yticks([])

def MyTrainTestVis(self, DatTr,DatTest,lenTrain,lenTest,totest=0):    
    ''' make figure for training test - set visualization'''
    fig,ax=plt.subplots(ncols=2,figsize=(8,3.5))
    self.MySaccadeVis(ax[0],DatTr,lenTrain,title='training data',alpha=.3)
    if type(totest)==int:
        titStr=''
    else:
        if len(lenTest)==1:
            titStr='subj '+str(int(self.suseHMM[totest][0]))
        else:
            titStr='n subj '+str(len(totest))

    self.MySaccadeVis(ax[1],DatTest,lenTest,title='test data '+titStr)
    return 
          
def MySaccadeVis(self,ax,XYdat,lengths,title='',alpha=1):
    ''' saccade visualization, on input ax, based on combined data 2d array, and lengths 1d array'''
    ax.set_title(title)
    ax.set_xlim([0,self.x_size])
    ax.set_ylim([self.y_size,0])
    ax.scatter(XYdat[:,0],XYdat[:,1],c=np.arange(len(XYdat[:,0])),alpha=alpha)
    Idxs=np.cumsum(lengths)
    for ci in range(len(lengths)):
        if ci==0:
            start=0
        else:
            start=Idxs[ci-1]
        ax.plot(XYdat[start:Idxs[ci],0],XYdat[start:Idxs[ci],1],alpha=alpha)
    return
        

