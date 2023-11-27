
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd



def AngleCalc(self,ycm,viewD):
    ''' calculate visual angle from vertical screen size (cm), viewing distance (cm) and resolution (pixel)  
    since y pixel size is already provided at initialization, does not have to be provided here'''
    self.pixdeg=degrees(atan2(.5*ycm, viewD)) / (.5*self.y_size)
    return self.pixdeg
def AngtoPix(self,Deg):
    ''' angle to pixel transform '''
    if hasattr(self, 'pixdeg')==False:
        print('please provide ycm (vertical screen size), and viewD, viewing distance for AngleCalc first')
    return  Deg / self.pixdeg

def PixdoDeg(self,pix):
	''' pixel to degrees of visual angle transform '''
	if hasattr(self, 'pixdeg')==False:
	    print('please provide ycm (vertical screen size), and viewD, viewing distance for AngleCalc first')
	return self.pixdeg*pix

def Entropy(self,BinnedCount,base=None):
    ''' from binned 2d fixation counts calculate entropy,  
    default natural log based calculation, this can be changed by base= optional arguments
    output 1: entorpy
    output 2: maximum possibe entropy for number of bins -- from uniform probability distribution'''
    assert len(np.shape(BinnedCount))==2,'2d data input expected'
    size=np.shape(BinnedCount)[0]*np.shape(BinnedCount)[1]
    entrMax=stats.entropy(1/size*np.ones(size),base=base)
    EntrBinned=stats.entropy(BinnedCount.flatten(),base=base)
    return EntrBinned,entrMax

def FixDurProg(self,nfixmax=10,Stim=0,Vis=1):
    ''' within trial fixation duration progression
    nfixmax controls the first n fixations to compare'''
    self.durprog=np.zeros((self.ns,self.np,nfixmax))
    self.durprog[:]=np.NAN
    for cs,s in enumerate(self.subjects):
        for cp,p in enumerate(self.stimuli):      
            Durs=self.GetDurations(s,p)
            if len(Durs)<nfixmax:
                self.durprog[cs,cp,0:len(Durs)]=Durs
            else:
                self.durprog[cs,cp,:]=Durs[0:nfixmax]
  
    if Stim==0:
        Y=np.nanmean(np.nanmean(self.durprog,1),0)
        Err=stats.sem(np.nanmean(self.durprog,1),axis=0,nan_policy='omit')
        if Vis:
            PlotDurProg(nfixmax,Y,Err)
            plt.title('All stimuli')
        
    else:
        Y=np.nanmean(self.durprog[:,self.stimuli==Stim,:],0).flatten()
   
        Err=stats.sem(self.durprog[:,self.stimuli==Stim,:],axis=0,nan_policy='omit').flatten()

        if Vis: 
            PlotDurProg(nfixmax,Y,Err)
            plt.title(Stim)
        
    return None

def BinnedCount(self,Fixcounts,Stim,fixs=1,binsize_h=50,binsize_v=None):
    ''' makes a grid of binsize_h*binsize_v pixels, and counts the num of fixies for each
    fixs==1 : used the full screen size   
    fixs==0, use infered bounds '''
    
    assert len(np.shape(Fixcounts))==2, '2d input expected'
    if binsize_v==None:
        binsize_v=binsize_h
        
    if fixs==1:
        x_size=self.x_size
        y_size=self.y_size
        x_size_start=0
        y_size_start=0
    else: 
        x_size_start=np.intp(self.bounds['BoundX1'][self.bounds['Stimulus']==Stim])
        x_size=np.intp(self.bounds['BoundX2'][self.bounds['Stimulus']==Stim])
        y_size_start=np.intp(self.bounds['BoundY1'][self.bounds['Stimulus']==Stim])
        y_size=np.intp(self.bounds['BoundY2'][self.bounds['Stimulus']==Stim])

    assert binsize_h>=2,'binsize_h must be at least 2'
    assert binsize_v>=2,'binsize_v must be at least 2'
    assert binsize_h<(x_size-x_size_start)/2,'too large horizontal bin, must be below screen widht/2'
    assert binsize_v<(y_size-y_size_start)/2,'too large vertical bin, must be below screen height/2'

    BinsH=np.arange(binsize_h+x_size_start,x_size,binsize_h) 
    BinsV=np.arange(binsize_v+y_size_start,y_size,binsize_v) 
    BinnedCount=np.zeros((len(BinsV),len(BinsH)))
    for cx,x in enumerate(BinsH):
        for cy,y in enumerate(BinsV):
            BinnedCount[cy,cx]=np.sum(Fixcounts[int(y_size_start+cy*binsize_v):int(y),int(x_size_start+cx*binsize_h):int(x)])
    return BinnedCount

    
    
