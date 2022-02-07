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


def GetFixationData(s,p,Dat,StimName='Stimulus',SubjName='subjectID'):
    """get X,Y fixation sequence for a participant and stimulus"""
    SubjIdx=np.nonzero(Dat[SubjName].to_numpy()==s)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(Dat[StimName].to_numpy()==p),SubjIdx) # idx for subject and painting
    FixTrialX=np.array(Dat['mean_x'].iloc[TrialSubIdx]) # get x data for trial
    FixTrialY=np.array(Dat['mean_y'].iloc[TrialSubIdx]) # get y data for trial
    return FixTrialX,FixTrialY


def AOIbounds(starts,end,nDiv):  
    return np.linspace(starts,end,nDiv+1)  

def GetParams(Data,StimName='Stimulus',SubjName='subjectID'):
    """ Get stimulus and subject info of dataset """  
    Subjects=np.unique(Data[SubjName].to_numpy())
    Stimuli=np.unique(Data[StimName].to_numpy())
    return Subjects,Stimuli

def InferSize(Data,Stimuli,StimName='Stimulus',SubjName='subjectID'):
    ''' Infer stimulus size as 99% central fixations data'''
    BoundsX=np.zeros((len(Stimuli),2))
    BoundsY=np.zeros((len(Stimuli),2))
    for cp,p in enumerate(Stimuli):
        Idx=np.nonzero(Data[StimName].to_numpy()==p)[0]
        BoundsX[cp,:]=np.percentile(Data['mean_x'].to_numpy()[Idx],[.5,99.5])
        BoundsY[cp,:]=np.percentile(Data['mean_y'].to_numpy()[Idx],[.5,99.5])
    return BoundsX,BoundsY

def MeanPlot(N,Y,yLab=0,xtickL=0,newfig=1):
    ''' expects data, row: subjects columbn: stimuli '''
    if newfig:
        plt.figure(figsize=(N/2,5))
    plt.errorbar(np.arange(N),np.mean(Y,0),stats.sem(Y,0)*2,linestyle='None',marker='o',color='darkred')
    if type(xtickL)!=int:
        plt.xticks(np.arange(N),xtickL,fontsize=10,rotation=60)
    plt.xlabel('Stimulus',fontsize=14)
    plt.ylabel(yLab,fontsize=14)
    return None

def HistPlot(Y,xtickL=0,newfig=1):
    ''' expects data, row: subjects columbn: stimuli '''
    if newfig:
       plt.figure()#figsize=(N/2,5))
    plt.hist(np.mean(Y,1),color='darkred')
    plt.xlabel(xtickL,fontsize=14)
    plt.ylabel('Num observers',fontsize=13)
    return None


def RunDescriptiveFix(Data,StimName='Stimulus',SubjName='subjectID',Visual=0):
    ''' for a dataset, return number of fixation and static probability matrix, for given divisions'''
    Subjects,Stimuli=GetParams(Data,StimName=StimName,SubjName=SubjName)
    print('Data for ',len(Subjects),'observers and ', len(Stimuli),' stimuli.')
    NS,NP=len(Subjects),len(Stimuli)
    NFixations=np.zeros((NS,NP))
    MeanFixXY=np.zeros(((NS,NP,2)))
    SDFixXY=np.zeros(((NS,NP,2)))
    for cs,s in enumerate(Subjects):
        for cp,p in enumerate(Stimuli):      
            FixTrialX,FixTrialY=GetFixationData(s,p,Data,StimName=StimName,SubjName=SubjName)
            if len(FixTrialX)>0:
                NFixations[cs,cp]=len(FixTrialX)
                MeanFixXY[cs,cp,0],MeanFixXY[cs,cp,1]=np.mean(FixTrialX),np.mean(FixTrialY)
                SDFixXY[cs,cp,0],MeanFixXY[cs,cp,1]=np.std(FixTrialX),np.std(FixTrialY)
            else:
                MeanFixXY[cs,cp,:],SDFixXY[cs,cp,:]=np.NAN,np.NAN
    print('Mean fix Num: ',np.round(np.mean(NFixations),2),' +/- ',np.round(np.std(NFixations),2))
    print('Num of trials with zero fixations:', np.sum(NFixations==0) )
    print('Num valid trials ',np.sum(NFixations>0))

    if Visual:
        MeanPlot(NP,NFixations,yLab='Num Fixations',xtickL=Stimuli)
        HistPlot(NFixations,xtickL='Avergage Num Fixations')
    NFix = xr.DataArray(NFixations, dims=(SubjName,StimName), coords={SubjName:Subjects,StimName: Stimuli})
    return NFix,Stimuli,Subjects,MeanFixXY,SDFixXY

def SaliencyPlot(SalMap,newfig=1):
    ''' expects data, row: subjects columbn: stimuli '''
    if newfig:
        plt.figure()
    plt.imshow(SalMap)
    return None




def FixCountCalc(Dat,Stim,x_size,y_size,StimName='Stimulus',SubjName='subjectID'):
    ''' Pixelwise fixation count for each participant, but for single stimulus  (Stim) '''
    Subjs,Stimuli=GetParams(Dat,StimName=StimName,SubjName=SubjName)
    FixCountInd=np.zeros(((len(Subjs),y_size,x_size)))
    for cs,s in enumerate(Subjs):
        x,y=np.intp(GetFixationData(s,Stim,Dat,StimName=StimName))
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

def SaliencyMap(Dat,Stim,x_size,y_size,StimName='Stimulus',SubjName='subjectID',SD=25,Ind=0):
    ''' Pipeline for saliency map calculation'''
    FixCountIndie=FixCountCalc(Dat,Stim,x_size,y_size,StimName=StimName,SubjName=SubjName)
    if Ind==0:
        smap=SaliencyMapFilt(FixCountIndie,SD=SD,Ind=0)
    else:
        smap=np.zeros_like(FixCountIndie)
        Subjs,Stimuli=GetParams(Dat,StimName=StimName,SubjName=SubjName)
        for cs,s in enumerate(Subjs):
            smap[cs,:,:]=SaliencyMapFilt(FixCountIndie[cs,:,:],SD=SD,Ind=1)               
    return smap






