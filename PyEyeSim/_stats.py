
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim ,CheckCoor
from .statshelper import SaliencyMapFilt,SaccadesTrial,ScanpathL,StatEntropy



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


def CalcStatPs(self,nHor,nVer,MinFix=20,InferS=1):
    ''' for a dataset, return number of fixation and static probability matrix, for given divisions
    returns StatPMat: nsubject*nstimulus*nvertical*nhorizontal '''
   
    statPMat=np.zeros((((self.ns,self.np,nVer,nHor))))
    statEntropyMat=np.zeros((self.ns,self.np,))
    
    for cs,s in enumerate(self.subjects):
        for cp,p in enumerate(self.stimuli):      
            FixTrialX,FixTrialY=self.GetFixationData(s,p)  
            
            if self.nfixations[cs,cp]>MinFix:
                NFixy,StatPtrial,StatNtrial=self.AOIFix(cp,FixTrialX,FixTrialY,nHor,nVer,InferS=InferS)
                statPMat[cs,cp,:,:]=StatPtrial.reshape(nVer,nHor)
                statEntropyMat[cs,cp]=StatEntropy(statPMat[cs,cp,:,:].reshape(-1,1))
            else:
                statEntropyMat[cs,cp]=np.NAN
                statPMat[cs,cp,:,:]=np.NAN
            
    return statPMat,statEntropyMat

    
def StatPDiffInd1(self,statPMat):
    StatIndDiff=np.zeros(((self.np,self.ns,self.ns)))
    for cp,p in enumerate(self.stimuli):   
        for cs1,s1 in enumerate(self.subjects):
            for cs2,s2 in enumerate(self.subjects):
                  StatIndDiff[cp,cs1,cs2]=np.nansum((statPMat[cs1,cp,:,:]-statPMat[cs2,cp,:,:])**2)
    return StatIndDiff
 
def StatPDiffInd2(self,BindAll):
    StatIndDiff=np.zeros(((self.np,self.ns,self.ns)))
    for cp,p in enumerate(self.stimuli):   
        for cs1,s1 in enumerate(self.subjects):
            for cs2,s2 in enumerate(self.subjects):
                 StatIndDiff[cp,cs1,cs2]=np.nansum((BindAll[cp][cs1,:,:]-BindAll[cp][cs2,:,:])**2)
    return StatIndDiff


def GetInddiff(self,nHor,nVer,Vis=0,zscore=0,InferS=1):
    ''' N DIVISION BASED. calculate individual similarity between all pairs of participants for all stimuli, for a given division'''
    statPMat,statEntropyMat=self.CalcStatPs(nHor,nVer,InferS=InferS)
 
    Inddiff=self.StatPDiffInd1(statPMat)
    Indmean=np.nanmean(Inddiff,2)
    SD=np.nanstd(Indmean,1)
    Indmean=np.nanmean(Indmean,1)
    if Vis:
        fig,ax=plt.subplots(figsize=(self.ns/4,4))
        if zscore:
            ax.scatter(np.arange(self.np),(Indmean-np.mean(Indmean))/np.std(Indmean),marker='o')
        else:
            ax.scatter(np.arange(self.np),Indmean,marker='o')
        ax.set_xticks(np.arange(self.np),self.stimuli,rotation=80,fontsize=12)
        ax.set_xlabel('Stimuli',fontsize=14)
        if zscore==1:
            ax.set_ylabel('fixation map relative difference',fontsize=14)
        else:
            ax.set_ylabel('fixation map difference',fontsize=14)
    return Indmean
  

def GetInddiff_v2(self,size=50,Vis=0,fixs=0):
    ''' PIXEl; NUMBER BASED; calculate individual similarity between all pairs of participants for all stimuli, for a given division'''
    statPMat=self.GetBinnedStimFixS(size=size,fixs=fixs)
    Inddiff=self.StatPDiffInd2(statPMat)
    Indmean=np.nanmean(Inddiff,2)
    SD=np.nanstd(Indmean,1)
    Indmean=np.nanmean(Indmean,1)
    if Vis:
        fig,ax=plt.subplots(figsize=(self.ns/4,4))

        #plt.errorbar(np.arange(self.np),Indmean,SD,marker='o',linestyle='none')
        ax.scatter(np.arange(self.np),Indmean,marker='o')
        ax.set_xticks(np.arange(self.np),self.stimuli,rotation=80,fontsize=12)
        ax.set_xlabel('Stimuli',fontsize=14)
        ax.set_ylabel('fixation map difference',fontsize=14)
    return Indmean
  

def GetBinnedStimFixS(self,size=50,fixs=1):
    ''' fixs=1: use full stimulus area
    fixs=0: use active area with 99% fixations '''
    BindAll=[]
    for cp,p in enumerate(self.stimuli):
        Fixcounts=self.FixCountCalc(p,CutAct=0)
        print('array size',np.round((Fixcounts.nbytes/1024)/1024,2),'MB')
        binIndC=self.BinnedCount(Fixcounts[0],p,fixs=fixs,binsize_h=size)
        BinDims=np.shape(binIndC)
       # print(cp,BinDims)
        BindAll.append(np.zeros(((self.ns,BinDims[0],BinDims[1]))))
        for cs,s in enumerate(self.subjects):
            BindAll[cp][cs,:,:]=self.BinnedCount(Fixcounts[cs],p,fixs=fixs,binsize_h=size)    
            BindAll[cp][cs,:,:]/=np.sum(BindAll[cp][cs,:,:])
    return BindAll


def RunDiffDivs(self,mindiv,maxdiv,Vis=1):
    ''' run grid based fixation map comparison from 
    mindiv*mindiv 
    to maxdiv *maxdiv number of divisions
    vis=1: visualized mean similarity'''
    if Vis:
        fig,ax=plt.subplots()
       # plt.figure()
    DiffsRaw=np.zeros((self.np,maxdiv-mindiv))
    DiffsZscore=np.zeros((self.np,maxdiv-mindiv))
    for cdiv,divs in enumerate(np.arange(mindiv,maxdiv)):
        DiffsRaw[:,cdiv]=self.GetInddiff(divs,divs,Vis=Vis,zscore=1)
        DiffsZscore[:,cdiv]=(DiffsRaw[:,cdiv]-np.mean(DiffsRaw[:,cdiv]))/np.std(DiffsRaw[:,cdiv])
    if Vis:
        ax.errorbar(np.arange(self.np),np.mean(DiffsZscore,1),np.std(DiffsZscore,1),linestyle='none',color='k',marker='o',markersize=5)
    return DiffsZscore,DiffsRaw



