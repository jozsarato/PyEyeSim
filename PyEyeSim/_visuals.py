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
    '''
    saccade visualization, on input ax, based on combined data 2d array, and lengths 1d array
   
     Positional arguments
    ----------
    ax : axis handle
    XYdat : XY data of fixations
    lengths : legnth of segments -- so that fixations from subsequent participants are not connected with "fake saccades"
    
    
     Optional arguments
     ----------
    title : image title The default is ''.
    alpha : line transparency, The default is 1.

    Returns
    -------
    None.

    '''
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
    '''
    visualize least and most typical sequences of fixations based on log-likelihood, from  fitted HMM

    
     Positional arguments
     ----------
    Dat : concateanted eye movement sequences .
    lengths : list of lengths for each segment
    ScoresLOO :  leave-one-out cross validation scores 
    
     Optional arguments
     ----------
    nshow : num of colums of examples
    title : figure title The default is ''.
    showim : show stimulus below fixations, The default is False.
    stimname : stimulus name, The default is 0, has to be provided if showim=True

    Returns
    -------
    None.

    '''
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
    '''
     visualize fixations and fitted hidden markov model
    hmmfitted: fitted hidden markov model
    ax: if not provided, new figure opens 

    
     Positional arguments
     ----------
    dat : sequence of fixations
    hmmfitted : fitted hmmlearn hidden markov model object

    
     Optional arguments
     ----------
    ax:  provide axis handle for the plot, if not new figure is opened The default is 0.
    showim : show stimulus if True The default is 1.
    stim : stimulus name
    lengths : length of time series sequences (needed for multiple sequences)
    incol : If True, use sequence of colorsm differing for each component. The default is False.

    Returns
    -------
    None.

    '''
    
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
        #print(np.shape(hmmfitted.covars_[c1]))
       # draw_ellipse((hmmfitted.means_[c1,0],hmmfitted.means_[c1,1]),hmmfitted.covars_[c1],ax=ax,facecolor='none',edgecolor=color2,linewidth=2)
        draw_ellipse(hmmfitted.means_[c1,0],hmmfitted.means_[c1,1],hmmfitted.covars_[c1],ax=ax,facecolor='none',edgecolor=color2,linewidth=2)

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
    
    
     Positional arguments
     ----------
    stimn: stimulus index.
    
    Optional arguments
    ----------
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
        if hasattr(self,'images'):
            xs1=(self.x_size-np.shape(self.images[self.stimuli[stimn]])[1])/2
            xs2=self.x_size-xs1
            ys1=(self.y_size-np.shape(self.images[self.stimuli[stimn]])[0])/2
            ys2=self.y_size-ys1
            ax.imshow(self.images[self.stimuli[stimn]],extent=[xs1,xs2,ys2,ys1])
    else:
        if hasattr(self,'images'):
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
    '''
    make figure for training test - set saccade visualization
    creates a figure with 2 subplots, visualizing the training and test set saccades

    Positional arguments
    ----------
    DatTr : training data HMM
    DatTest : test data HMM
    lenTrain : list of lengths of training set sequences
    lenTest : list of lengths of test set sequences
    Optional argument
    ----------
    totest : if list, specified participants for test. The default is 0.

    Returns
    -------
    None.

    '''
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

def VisGrid(self,vals,Stim,ax=0,alpha=.3,cmap='inferno',cbar=0,vmax=0):
    '''  
    visualize transparent grid of values on stimulus image

     Positional arguments
     ----------
    vals: values to lay over image
    Stim: stimulus name


     Optional arguments
     ----------
    ax: provide axis handle for the plot, if not new figure is opened
    alpha: transparency
    inferS: needed if not full screen images, with background--> not calculating value for full image in this case, but using BoundsX&Y
    cb: visualize colorbar --> also returned   
    vmax: to fix colormap max and minimum values at x/- vmax(for stimulus comparabilty)
    
    '''
    if type(ax)==int:
        fig,ax=plt.subplots()
    if hasattr(self,'images'):
        ax.imshow(self.images[Stim])
  
    stimId=np.nonzero(self.stimuli==Stim)[0]   
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
    '''
    Positional arguments
    ----------
    highlight cells in grid comparisons
    Stim: stimulus name
    pvals:  matrix of p values for the grid
    axis: axis handle
    '''
    if hasattr(self,'images'):

        idims=np.shape(self.images[Stim])
        yimsize,ximsize=idims[0],idims[1]
    else:
        yimsize,ximsize=self.y_size, self.x_size
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


def VisHeatmap(self,Stim,smap,ax=0,cutoff=0,alpha=.5,cmap='plasma',cbar=False,cbarlabel=False,title=''):
    '''
    visualize pre-calculated heatmap 
    
    Positional arguments
    ----------
    Stim : stimulus name
    smap : previously calculated heatmap matrix
    
    Optional arguments
    ----------
    ax : axis handle, if zero opens new figure
    cutoff: threshold, below values are shown transparent (makes the image below the heatmap more visible)
    alpha:  transparenc of heatmap. The default is .5.
    cmap : colormap of heatmap The default is 'plasma'.
    cbar : create colorbar if True The default is False.
    cbarlabel : y label of colorbar  The default is False.
    title : title of the figure (string) The default is ''.

    Returns
    -------
    None.

    '''
    if cutoff=='median':
        cutThr=np.median(smap)
    elif cutoff>0:
        cutThr=np.percentile(smap,cutoff) 
    else:
        cutThr=0
    smap[smap<cutThr]=np.nan  # replacing below threshold with NAN
    if ax==False:
        fig,ax=plt.subplots()
    if hasattr(self,'images')==True:
        ax.imshow(self.images[Stim])
        ax.set_xlim(0,np.shape(self.images[Stim])[1])
        ax.set_ylim(np.shape(self.images[Stim])[0],0)
    
    cols=ax.imshow(smap,alpha=alpha,cmap=cmap) 
    if cbar:
        cb=plt.colorbar(cols,ax=ax,shrink=.6)
        if cbarlabel!=' ':
            cb.ax.get_yaxis().set_ticks([])
            cb.ax.get_yaxis().labelpad = 15
            cb.ax.set_ylabel(cbarlabel, rotation=270)

    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title)
   
    return


def VisSimmat(self,simdat,ax=0,title=''):
    '''
    Visualize similarity matrix, with highest value for each column- most similar stimulus for each stimulus
    
    
     Positional argument
     ----------
    simdat : matrix values (num stimuli  * num stimuli length)
    
     Optional arguments
     ----------
    ax : axis handle, if 0 (default) opens new figure
    title : title of the figure. The default is '' - no visible title.

    Returns
    -------
    None.

    '''

    if type(ax)==int:
        fig,ax=plt.subplots()
    cols=ax.pcolor(simdat)
    ax.set_xticks(np.arange(self.np)+.5,self.stimuli,rotation=70)
    ax.set_yticks(np.arange(self.np)+.5,self.stimuli) #,rotation=50)
    ax.scatter(np.arange(self.np)+.5,np.argmax(simdat,1)+.5,color='r')
    plt.colorbar(cols,ax=ax)
    ax.set_title(title)
    
    
def Vis_Saccade_Angles(self,stim,subj='all',color='darkgreen',width= np.pi / 25,binsize=10):
    '''
    visualizes saccade angle for all or a given subject

    
     Positional arguments
    ----------
    stim : stimulus name.
    

     optional arguments
     ----------
    subj : subject numnber,. The default is 'all'.
    color : TYPE, optional
        DESCRIPTION. The default is 'darkgreen'.
    width : TYPE, optional
        DESCRIPTION. The default is np.pi / 25.
    binsize : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    '''
    if hasattr(self,'saccadeangles')==False:
        self.GetSaccades()
    stimn=int(np.nonzero(self.stimuli==stim)[0])
    binss=np.arange(0,360+binsize,binsize)
    sacarray=np.array([])
    for cs in range(self.ns):
        if np.isnan(self.nsac[cs,stimn])==False:
            saccarray=np.concatenate((sacarray,self.saccadeangles[cs,stimn]))  #self.stimuli==stim]

    bincounts, edges=np.histogram(saccarray,bins=binss)

    ax=plt.subplot(projection='polar')
    ax.bar(np.deg2rad(edges[:-1]),bincounts,width=width,bottom=0.0,color=color)
  
    ax.set_title(stim)
    
def VisSaccadedat(self, stimn, ax=None, alpha=0.5, allS=True, scan_path_col='salmon', fixation_col='blue', visFix=False,center=False):
    ''' 
    Description: Visualize scan path for a given stimulus.
    Positional Argument:
      -------
      stimn: stimulus index.
      
    Optional Arguments:
     -------    
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
    if hasattr(self,'images'):
         ax.imshow(self.images[self.stimuli[stimn]])

    idx=self.data.Stimulus==self.stimuli[stimn]
    x=ax.plot([self.data.start_x[idx],self.data.end_x[idx]],[self.data.start_y[idx],self.data.end_y[idx]],color='k',alpha=alpha)
    ax.set_title(self.stimuli[stimn])
    ax.set_xlim([0, self.x_size])
    ax.set_ylim([self.y_size, 0])
    ax.set_xticks([])
    ax.set_yticks([])
# %%
