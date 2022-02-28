#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:44:02 2022

@author: jarato
"""

import numpy as np
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
import pickle
import xarray as xr

#%%


def ProcessDat(Data,StimName='Stimulus',SubjName='subjectID',mean_x='mean_x',mean_y='mean_y',FixDuration=0):
    ''' the library expects column names Stimulus, subjectID, mean_x and mean_y, if you data is not in this format, this function will rename your columns accordingly 
     optionally, with FixDuration you can name your column of fixations lengths, which will be called duration afterwards'''
   # print(type(FixDuration))
    if type(FixDuration)!='int':
        DataN=Data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y',FixDuration: 'duration'})
    else:
        DataN=Data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y'})

    return DataN

def GetFixationData(s,p,Dat):
    """get X,Y fixation sequence for a participant and stimulus"""
    SubjIdx=np.nonzero(Dat['subjectID'].to_numpy()==s)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(Dat['Stimulus'].to_numpy()==p),SubjIdx) # idx for subject and painting
    FixTrialX=np.array(Dat['mean_x'].iloc[TrialSubIdx]) # get x data for trial
    FixTrialY=np.array(Dat['mean_y'].iloc[TrialSubIdx]) # get y data for trial
    return FixTrialX,FixTrialY

def GetDurations(s,p,Dat):
    """get X,Y fixation sequence for a participant and stimulus"""
    SubjIdx=np.nonzero(Dat['subjectID'].to_numpy()==s)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(Dat['Stimulus'].to_numpy()==p),SubjIdx) # idx for subject and painting
     # get x data for trial
    durations=np.array(Dat['duration'].iloc[TrialSubIdx]) # get y data for trial
    return durations


def AOIbounds(start,end,nDiv):  
    return np.linspace(start,end,nDiv+1)  

def GetParams(Data):
    """ Get stimulus and subject info of dataset """  
    Subjects=np.unique(Data['subjectID'].to_numpy())
    Stimuli=np.unique(Data['Stimulus'].to_numpy())
    return Subjects,Stimuli

def InferSize(Data,Stimuli,Interval=99):
    ''' Infer stimulus size as central Interval % fixations data'''
    BoundsX=np.zeros((len(Stimuli),2))
    BoundsY=np.zeros((len(Stimuli),2))
    for cp,p in enumerate(Stimuli):
        Idx=np.nonzero(Data['Stimulus'].to_numpy()==p)[0]
        BoundsX[cp,:]=np.percentile(Data['mean_x'].to_numpy()[Idx],[(100-Interval)/2,Interval+(100-Interval)/2])
        BoundsY[cp,:]=np.percentile(Data['mean_y'].to_numpy()[Idx],[(100-Interval)/2,Interval+(100-Interval)/2])
    return BoundsX,BoundsY

def MeanPlot(N,Y,yLab=0,xtickL=0,newfig=1,color='darkred',label=None):
    ''' expects data, row: subjects columbn: stimuli '''
    if newfig:
        plt.figure(figsize=(N/2,5))
    plt.errorbar(np.arange(N),np.mean(Y,0),stats.sem(Y,0)*2,linestyle='None',marker='o',color=color,label=label)
    if type(xtickL)!=int:
        plt.xticks(np.arange(N),xtickL,fontsize=10,rotation=60)
    plt.xlabel('Stimulus',fontsize=14)
    plt.ylabel(yLab,fontsize=14)
    return None

def HistPlot(Y,xtickL=0,newfig=1):
    ''' expects data, row: subjects columbn: stimuli '''
    assert len(np.shape(Y))==2, '2d data is expected: observer*stimulus'
    if newfig:
       plt.figure()
    plt.hist(np.mean(Y,1),color='darkred')
    plt.xlabel(xtickL,fontsize=14)
    plt.ylabel('Num observers',fontsize=13)
    return None


def RunDescriptiveFix(Data,Visual=0,duration=0):
    ''' for a dataset, return number of fixation, inferred stim boundaries and mean and SD of fixation locatios '''
    
    Subjects,Stimuli=GetParams(Data)
    #Subjects,Stimuli=GetParams(Data,StimName=StimName,SubjName=SubjName)

    BoundsX,BoundsY=InferSize(Data,Stimuli,Interval=99)
    print('Data for ',len(Subjects),'observers and ', len(Stimuli),' stimuli.')
    NS,NP=len(Subjects),len(Stimuli)
    NFixations=np.zeros((NS,NP))
    MeanFixXY=np.zeros(((NS,NP,2)))
    SDFixXY=np.zeros(((NS,NP,2)))
    if duration:
        Durations=np.zeros((NS,NP))
        
    for cs,s in enumerate(Subjects):
        for cp,p in enumerate(Stimuli):      
            FixTrialX,FixTrialY=GetFixationData(s,p,Data)
            if len(FixTrialX)>0:
                NFixations[cs,cp]=len(FixTrialX)
                MeanFixXY[cs,cp,0],MeanFixXY[cs,cp,1]=np.mean(FixTrialX),np.mean(FixTrialY)
                SDFixXY[cs,cp,0],SDFixXY[cs,cp,1]=np.std(FixTrialX),np.std(FixTrialY)
                Durations[cs,cp]=np.mean(GetDurations(s,p,Data))
            else:
                MeanFixXY[cs,cp,:],SDFixXY[cs,cp,:], Durations[cs,cp]=np.NAN,np.NAN,np.NAN
    print('Mean fixation number: ',np.round(np.mean(np.mean(NFixations,1)),2),' +/- ',np.round(np.std(np.mean(NFixations,1)),2))
    print('Mean fixation duration: ',np.round(np.mean(np.mean(Durations,1)),1),' +/- ',np.round(np.std(np.mean(Durations,1)),1),'msec')
    
    print('Num of trials with zero fixations:', np.sum(NFixations==0) )
    print('Num valid trials ',np.sum(NFixations>0))
    print('Mean X location: ',np.round(np.mean(np.nanmean(MeanFixXY[:,:,0],1)),1),' +/- ',np.round(np.std(np.nanmean(MeanFixXY[:,:,0],1)),1),' pixels')
    print('Mean Y location: ',np.round(np.mean(np.nanmean(MeanFixXY[:,:,1],1)),1),' +/- ',np.round(np.std(np.nanmean(MeanFixXY[:,:,1],1)),1),' pixels')
    
    if Visual:
        MeanPlot(NP,NFixations,yLab='Num Fixations',xtickL=Stimuli)
        HistPlot(NFixations,xtickL='Avergage Num Fixations')
    Bounds=pd.DataFrame(columns=['Stimulus'],data=Stimuli)
    Bounds['BoundX1']=BoundsX[:,0]
    Bounds['BoundX2']=BoundsX[:,1]
    Bounds['BoundY1']=BoundsY[:,0]
    Bounds['BoundY2']=BoundsY[:,1]    
    NFix = xr.DataArray(NFixations, dims=('subjectID','Stimulus'), coords={'subjectID':Subjects,'Stimulus': Stimuli})
    MeanFixXY = xr.DataArray(MeanFixXY, dims=('subjectID','Stimulus','XY'), coords={'subjectID':Subjects,'Stimulus': Stimuli, 'XY':['X','Y']})
    SDFixXY = xr.DataArray(SDFixXY, dims=('subjectID','Stimulus','XY'), coords={'subjectID':Subjects,'Stimulus': Stimuli, 'XY':['X','Y']})

    return NFix,Stimuli,Subjects,MeanFixXY,SDFixXY,Bounds


def DescripitiveGroups(Data,cond):
    ''' expects a categorical column name in cond '''
    Conds=np.unique(Data[cond])
    Cols=['salmon','darkgreen','b','orange','olive','r','m','c']
    assert len(Conds)>1, 'you need more than 1 group'
    assert len(Conds)<=len(Cols), 'too many groups, max is 8'
    plt.figure()
    for cc,c in enumerate(Conds):
        print('Group ',cc+1,c)
        Dat=Data[Data[cond]==c]
        NFix,Stimuli,Subjects,MeanFixXY,SDFixXY,Bounds=RunDescriptiveFix(Dat,Visual=0)
        NS,NP=len(Subjects),len(Stimuli)
        MeanPlot(NP,NFix,yLab='Num Fixations',xtickL=Stimuli,color=Cols[cc],newfig=0,label=c)
        print(' ')
    plt.legend()
    return None

def SaliencyPlot(SalMap,newfig=1):
    ''' expects data, row: subjects columbn: stimuli '''
    if newfig:
        plt.figure()
    plt.imshow(SalMap)
    return None




def FixCountCalc(Dat,Stim,x_size,y_size):
    ''' Pixelwise fixation count for each participant, but for single stimulus  (Stim) '''
    assert np.sum(Dat['Stimulus']==Stim)>0, 'stimulus not found'
    Subjs,Stimuli=GetParams(Dat)
    FixCountInd=np.zeros(((len(Subjs),y_size,x_size)))
    for cs,s in enumerate(Subjs):
        x,y=np.intp(GetFixationData(s,Stim,Dat))
        Valid=np.nonzero((x<x_size)&(y<y_size))[0]
        X,Y=x[Valid],y[Valid]
        FixCountInd[cs,Y,X]+=1
    return FixCountInd


def SaliencyMapFilt(Fixies,SD=25,Ind=0):
    ''' Gaussian filter of fixations counts, Ind=1 for individual, Ind=0 for group '''
    if Ind==0:
        Smap=ndimage.filters.gaussian_filter(np.mean(Fixies,0),SD)
    else:
        Smap=ndimage.filters.gaussian_filter(Fixies,SD)
    return Smap

def SaliencyMap(Dat,Stim,x_size,y_size,SD=25,Ind=0,Vis=0):
    ''' Pipeline for saliency map calculation'''
    FixCountIndie=FixCountCalc(Dat,Stim,x_size,y_size)
    assert np.sum(FixCountIndie)>0,'!!no fixations found'
    if np.sum(FixCountIndie)<10:
        print('WARNING NUM FIX FOUND: ',np.sum(FixCountIndie))
    if Ind==0:
        smap=SaliencyMapFilt(FixCountIndie,SD=SD,Ind=0)
    else:
        smap=np.zeros_like(FixCountIndie)
        Subjs,Stimuli=GetParams(Dat)
        for cs,s in enumerate(Subjs):
            smap[cs,:,:]=SaliencyMapFilt(FixCountIndie[cs,:,:],SD=SD,Ind=1)       
    if Vis:
        plt.imshow(smap)
        plt.xticks([])
        plt.yticks([])
    return smap


def BinnedCount(Fixcounts,x_size,y_size,x_size_start=0,y_size_start=0,binsize_h=50,binsize_v=None):
    ''' makes a grid of binsize_h*binsize_v pixels, and counts the num of fixies for each'''
    assert len(np.shape(Fixcounts))==2, '2d input expected'
    if binsize_v==None:
        binsize_v=binsize_h
    assert binsize_h>=2,'binsize_h must be at least 2'
    assert binsize_v>=2,'binsize_v must be at least 2'
    assert binsize_h<(x_size-x_size_start)/2,'too large horizontal bin, must be below screen widht/2'
    assert binsize_v<(y_size-y_size_start)/2,'too large vertical bin, must be below screen height/2'
    
    BinsH=np.arange(binsize_h+x_size_start,x_size,binsize_h) 
    BinsV=np.arange(binsize_v+y_size_start,y_size,binsize_v) 
    
    BinnedCount=np.zeros((len(BinsV),len(BinsH)))
    for cx,x in enumerate(BinsH):
        for cy,y in enumerate(BinsV):
            BinnedCount[cy,cx]=np.sum(Fixcounts[y_size_start+(cy*binsize_v):y,x_size_start+(cx*binsize_h):x])
    return BinnedCount


def Entropy(BinnedCount,base=None):
    ''' based on binned  2d fixation counts return entropy and relative entropy, default natural log'''
    assert len(np.shape(BinnedCount))==2,'2d data input expected'
    size=np.shape(BinnedCount)[0]*np.shape(BinnedCount)[1]
    entrMax=stats.entropy(1/size*np.ones(size),base=base)
    EntrBinned=stats.entropy(BinnedCount.flatten(),base=base)
    return EntrBinned,EntrBinned/entrMax



