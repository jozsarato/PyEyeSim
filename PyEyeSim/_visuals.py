import numpy as np
#from numpy import matlib
#from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
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
        alph=.8
    else:
        alph=.8
    if np.shape(dat)[0]>200:
        alph=.6

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


def VisScanPath(self, stimn, ax=None, alpha=0.5, allS=True, scan_path_col='salmon', fixation_col='blue', visFix=False, num_fixations=None,center=False):
    ''' 
    Description: Visualize scan path for a given stimulus.
    Arguments:
    stimn: stimulus index.
    ax: if not provided, a new figure is created.
    alpha: Transparency level for scan path. Defaults to 0.5.
    allS:  Default=True, visualize scan paths for all participants; otherwise specify participant index.
    scan_path_col: Color for the scan path. Defaults to 'salmon'.
    fixation_col: Color for fixation points. Defaults to 'blue'.
    VisFix: Default=False. If True, Visualize fixations with scatter points.
    num_fixations: Number of fixations to visualize. If not provided all fixations will be enumerated.
    Returns:
    '''
    if ax is None:
        fig, ax = plt.subplots()
    if center:
        xs1=(self.x_size-np.shape(self.images[self.stimuli[stimn]])[1])/2
        xs2=self.x_size-xs1
        ys1=(self.y_size-np.shape(self.images[self.stimuli[stimn]])[0])/2
        ys2=self.y_size-ys1
        ax.imshow(self.images[self.stimuli[stimn]],extent=[xs1,xs2,ys2,ys1])
    else:
        ax.imshow(self.images[self.stimuli[stimn]])

    if type(allS) == bool:
        for cs in range(self.ns):
            fixx, fixy = self.GetFixationData(self.subjects[cs], self.stimuli[stimn])
            ax.plot(fixx, fixy, alpha=alpha, color=scan_path_col)
            if visFix:
                ax.scatter(fixx, fixy, color=fixation_col, alpha=alpha, s=20)

            # Enumerate all fixations by default
           # num_fixations = len(fixx) if num_fixations is None else num_fixations

            #for i, (x, y) in enumerate(zip(fixx[:num_fixations], fixy[:num_fixations])):
             #   ax.text(x, y, str(i + 1), color="white", fontsize=10, ha='center', va='center')

    else:
        fixx, fixy = self.GetFixationData(self.subjects[allS], self.stimuli[stimn])
        ax.plot(fixx, fixy, alpha=alpha, color=scan_path_col)
        if visFix:
            ax.scatter(fixx, fixy, color=fixation_col, alpha=alpha, s=20)

        # Enumerate all fixations by default
  #      num_fixations = len(fixx) if num_fixations is None else num_fixations

#        for i, (x, y) in enumerate(zip(fixx[:num_fixations], fixy[:num_fixations])):
 #           ax.text(x, y, str(i + 1), color="white", fontsize=10, ha='center', va='center')

    ax.set_xlim([0, self.x_size])
    ax.set_ylim([self.y_size, 0])
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

def VisGrid(self,vals,Stim,ax=0,alpha=.3,cmap='inferno',cbar=0,vmax=0,inferS=0):
    '''  
    visualize transparent grid of values on stimulus image

    Arguments:
    vals: values to lay over image
    Stim: stimulus name


    optional:
    ax: provide axis to plot, if not new figure is opened
    alpha: transparency
    inferS: needed if not full screen images, with background--> not calculating value for full image in this case, but using BoundsX&Y
    cb: visualize colorbar --> also returned   
    vmax: to fix colormap max and minimum values at x/- vmax(for stimulus comparabilty)
    
    '''
    if type(ax)==int:
        fig,ax=plt.subplots()
    idims=np.shape(self.images[Stim])
    yimsize,ximsize=idims[0],idims[1]
    
    ax.imshow(self.images[Stim])
    if inferS==0:
        horcells=np.linspace(0,ximsize,np.shape(vals)[1]+1)
        vercells=np.linspace(0,yimsize,np.shape(vals)[0]+1)
    else:
        stimId=np.nonzero(self.stimuli==Stim)[0]   
        print(stimId)
        horcells=np.linspace(self.boundsX[stimId,0],self.boundsX[stimId,1],np.shape(vals)[1]+1).flatten()
        vercells=np.linspace(self.boundsY[stimId,0],self.boundsY[stimId,1],np.shape(vals)[0]+1).flatten()
    if vmax==0:
        cols=ax.pcolormesh(horcells,vercells,vals,alpha=alpha,cmap=cmap)
    else:
        cols=ax.pcolormesh(horcells,vercells,vals,alpha=alpha,cmap=cmap,vmin=-vmax,vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar:
        cb=plt.colorbar(cols,ax=ax,shrink=.6)
    else:
        cb=0
    return cb
        

def Highlight_Sign(self,Stim,pvals,axis):
    idims=np.shape(self.images[Stim])
    yimsize,ximsize=idims[0],idims[1]
    x,y=np.linspace(0,ximsize,np.shape(pvals)[1]+1),np.linspace(0,yimsize,np.shape(pvals)[0]+1)
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            if pvals[j,i]<.05:
                if pvals[j,i]<.01:
                    linestyle='-'
                else:
                    linestyle='--'
                axis.plot([x[i], x[i + 1]], [y[j], y[j]], color='k',linestyle=linestyle)  # Horizontal lines
                axis.plot([x[i], x[i]], [y[j], y[j + 1]], color='k',linestyle=linestyle)  # Vertical lines

                if i<len(x):
                    axis.plot([x[i+1], x[i+1]], [y[j], y[j + 1]], color='k',linestyle=linestyle)  # Vertical lines
                if j <len(y):
                    axis.plot([x[i], x[i + 1]], [y[j+1], y[j+1]], color='k',linestyle=linestyle)  # Horizontal lines
    axis.set_xlabel(f'num sign p<.05: {np.sum(pvals<.05)} ,chance expectation: {np.round(np.shape(pvals)[0]*np.shape(pvals)[1]*.05,1)} ')  
    return 

# %%
