#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:58:03 2022

@author: jarato
"""

import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
import pickle
import xarray as xr
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
import platform
#% import  library helper functions

from .visualhelper import VisBinnedProg,PlotDurProg,JointBinnedPlot,MeanPlot,draw_ellipse,HistPlot
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim, CheckCoor
from .statshelper import SaliencyMapFilt,SaccadesTrial,ScanpathL,StatEntropy


class EyeData:
	
    from ._visuals import VisScanPath,MySaccadeVis,VisLOOHMM,VisHMM,MyTrainTestVis
    from ._dataproc import GetParams,GetStimuli,GetFixationData,GetDurations,GetGroups,GetCats,GetSaccades,SaccadeSel,GetEntropies,InferSize,Heatmap,FixCountCalc
    from ._stats import AngleCalc,AngtoPix,PixdoDeg,Entropy,FixDurProg,BinnedCount,GetInddiff,GetInddiff_v2,RunDiffDivs,GetBinnedStimFixS,StatPDiffInd2,StatPDiffInd1,CalcStatPs,CalcRets,CalcImmRets,BinnedDescriptives
    from ._comparegroups import CompareGroupsFix,CompareGroupsHeatmap,CompareWithinGroupsFix,FixDurProgGroups,BinnedDescriptivesGroups,CompareGroupsMat
    from ._scanpathsim import AOIFix,SacSimPipeline,SacSim1Group

    try: 
    	from ._hmm import DataArrayHmm,MyTrainTest,FitLOOHMM,FitVisHMM,FitVisHMMGroups,HMMSimPipeline
    except:
    	print('hmmlearn not found, hidden markov model functionality will not work')

    def __init__(self, name, design,data,x_size,y_size,fixdata=1):
        ''' 
        Description: initalizing eye-tracking data object.
        
        Arguments:
        name (str): A name associated with the eye-tracking data.
        design (str): Information about the study design.
        data (pandas.DataFrame): The eye-tracking data.
        x_size (int): Screen size in pixels (width).
        y_size (int): Screen size in pixels (height).
        fixdata (int, optional): Indicator for fixation data (1) or saccade data (0). Default is 1 as fixation data are expected for most functionalities. 
        '''
        self.name = name
        self.design = design
        self.data=data
        self.x_size=x_size
        self.y_size=y_size
        self.fixdata=fixdata


        if fixdata:
            print('Fixation dataset',self.name)
        else:
            print('Saccade dataset',self.name)
            print(' Expected saccade columns:  begin_x,begin_y,end_x,end_y')

        print('dataset size: ',np.shape(self.data))
        print('study design: ',self.design)
        print('presentation size:  x=',self.x_size,'pixels y=',self.y_size,' pixels')
        print('presentation size:  x=',self.x_size,'pixels y=',self.y_size,' pixels')
        if fixdata:   # if fixation data
            DefColumns={'Stimulus':'Stimulus','subjectID':'subjectID','mean_x':'mean_x','mean_y':'mean_y'}
        else:   # if saccade data
            DefColumns={'Stimulus':'Stimulus','subjectID':'subjectID','begin_x':'begin_x', 'begin_y':'begin_y', 'end_x':'end_x','end_y':'end_y'}

        for df in DefColumns:
            try:
                data[DefColumns[df]]
                print('column found: ', df,' default: ',DefColumns[df])
            except:
                print(df," not found !!, provide column as", df,"=YourColumn , default: ",DefColumns[df])
    def info(self):
        ''' 
         Description: prints screen information, dataset name and study design. 
        '''
        print('screen x_size',self.x_size)
        print('screen y_size',self.y_size)
        print(self.name)
        print(self.design,'design')

    def data(self):
        ''' 
        Description: shows dataset.
        '''
        return self.data
    
    
    
    def DataInfo(self,Stimulus='Stimulus',subjectID='subjectID',mean_x='mean_x',mean_y='mean_y',FixDuration=0,StimPath=0,StimExt='.jpg',infersubpath=False):
        ''' 
        Description: Provide information about amount of stimuli and subjects.
        Arguments:
        Stimulus (str): Column name for stimulus information in the eye-tracking data.
        subjectID (str): Column name for subject ID information in the eye-tracking data.
        mean_x (str): Column name for mean x-coordinate of fixations in the eye-tracking data.
        mean_y (str): Column name for mean y-coordinate of fixations in the eye-tracking data.
        FixDuration (int or str): Column name or integers for fixation duration in the eye-tracking data.
            If an integer, fixation duration column is assumed absent. It will be renamed "duration" afterwards
        StimPath (str): Path to stimuli. Set to 0 if not provided.
        StimExt (str): File extension of stimuli (default: '.jpg').
        infersubpath (bool): Flag to infer stimulus subpaths based on subject IDs (default: False).
        '''
       # print(type(FixDuration))
       
        if self.fixdata:
            if type(FixDuration)!='int':
                self.data=self.data.rename(columns={Stimulus:'Stimulus',subjectID:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y',FixDuration: 'duration'})
            else:
                self.data=self.data.rename(columns={Stimulus:'Stimulus',subjectID:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y'})
        else:
            if type(FixDuration)!='int':
                self.data=self.data.rename(columns={Stimulus:'Stimulus',subjectID:'subjectID',FixDuration: 'duration'})
            else:
                self.data=self.data.rename(columns={Stimulus:'Stimulus',subjectID:'subjectID'})
        
        try:
            subjs,stims=self.GetParams()
            print('info found for '+ str(len(subjs))+' subjects, and '+str(len(stims))+' stimuli')
            
        except:
            print('stimulus and subject info not found')
            
        if StimPath==0:
            print('Stim path not provided')
        else:
         #  try: 
            self.GetStimuli(StimExt,StimPath,infersubpath=infersubpath)
            print('stimuli loaded succesfully, access as self.images')
          # except:   
           #    print('stimuli not found')
        pass
  
    
    
    
   
    

    def RunDescriptiveFix(self,Visual=0,duration=0):
        '''
        Description:  Calculate descriptive statistics for fixation data in dataset.

        Arguments:
        Visual (int): Flag indicating whether to generate visual plots (default: 0). Use 1 to show plots.
        duration (int): Flag indicating whether fixation duration data is present (default: 0). Use one if fixation duration is present.  
        
        Returns: Mean fixation number, Number of valid fixations, inferred stim boundaries and mean and SD of fixation locations, mean Saccade amplitude, mean scanpath length.
        '''
        
        Subjects,Stimuli=self.GetParams()
        print('Data for ',len(self.subjects),'observers and ', len(self.stimuli),' stimuli.')
        self.boundsX,self.boundsY=self.InferSize(Interval=99)
        self.actsize=(self.boundsX[:,1]-self.boundsX[:,0])*(self.boundsY[:,1]-self.boundsY[:,0])
        self.nfixations=np.zeros((self.ns,self.np))
        self.nfixations[:]=np.NAN
        self.sacc_ampl=np.zeros((self.ns,self.np))
        self.len_scanpath=np.zeros((self.ns,self.np))

        MeanFixXY=np.zeros(((self.ns,self.np,2)))
        SDFixXY=np.zeros(((self.ns,self.np,2)))
        if duration:
            self.durations=np.zeros((self.ns,self.np))
         
            
        for cs,s in enumerate(self.subjects):
            for cp,p in enumerate(self.stimuli):      
                FixTrialX,FixTrialY=self.GetFixationData(s,p)
               
                if len(FixTrialX)>0:
                    self.nfixations[cs,cp]=len(FixTrialX)
                    self.sacc_ampl[cs,cp],self.len_scanpath[cs,cp]=ScanpathL(FixTrialX,FixTrialY)
                    MeanFixXY[cs,cp,0],MeanFixXY[cs,cp,1]=np.mean(FixTrialX),np.mean(FixTrialY)
                    SDFixXY[cs,cp,0],SDFixXY[cs,cp,1]=np.std(FixTrialX),np.std(FixTrialY)
                    if duration:
                        self.durations[cs,cp]=np.mean(self.GetDurations(s,p))     
                else:
                    MeanFixXY[cs,cp,:],SDFixXY[cs,cp,:]=np.NAN,np.NAN
                    self.sacc_ampl[cs,cp],self.len_scanpath[cs,cp]=np.NAN,np.NAN
                    if duration:
                        self.durations[cs,cp]=np.NAN
                        
        print('Mean fixation number: ',np.round(np.nanmean(np.nanmean(self.nfixations,1)),2),' +/- ',np.round(np.nanstd(np.nanmean(self.nfixations,1)),2))
        if duration:
            print('Mean fixation duration: ',np.round(np.nanmean(np.nanmean(self.durations,1)),1),' +/- ',np.round(np.nanstd(np.nanmean(self.durations,1)),1),'msec')
        else:
            print('fixation duration not asked for')
        print('Num of trials with zero fixations:', np.sum(self.nfixations==0) )
        print('Num valid trials ',np.sum(self.nfixations>0))
        print('Mean X location: ',np.round(np.mean(np.nanmean(MeanFixXY[:,:,0],1)),1),' +/- ',np.round(np.std(np.nanmean(MeanFixXY[:,:,0],1)),1),' pixels')
        print('Mean Y location: ',np.round(np.mean(np.nanmean(MeanFixXY[:,:,1],1)),1),' +/- ',np.round(np.std(np.nanmean(MeanFixXY[:,:,1],1)),1),' pixels')
        print('Mean saccade  amplitude: ',np.round(np.mean(np.nanmean(self.sacc_ampl,1)),1),' +/- ',np.round(np.std(np.nanmean(self.sacc_ampl,1)),1),' pixels')
        print('Mean scanpath  length: ',np.round(np.mean(np.nanmean(self.len_scanpath,1)),1),' +/- ',np.round(np.std(np.nanmean(self.len_scanpath,1)),1),' pixels')

        if Visual:
            MeanPlot(self.np,self.nfixations,yLab='num fixations',xtickL=Stimuli)
            MeanPlot(self.np,self.len_scanpath,yLab=' total scanpath length (pixels)',xtickL=Stimuli)

            HistPlot(self.nfixations,xtickL='Average Num Fixations')
        Bounds=pd.DataFrame(columns=['Stimulus'],data=Stimuli)
        Bounds['BoundX1']=self.boundsX[:,0]
        Bounds['BoundX2']=self.boundsX[:,1]
        Bounds['BoundY1']=self.boundsY[:,0]
        Bounds['BoundY2']=self.boundsY[:,1] 
        if duration:
            self.durs=xr.DataArray(self.durations, dims=('subjectID','Stimulus'), coords={'subjectID':Subjects,'Stimulus': Stimuli})
        self.nfix = xr.DataArray(self.nfixations, dims=('subjectID','Stimulus'), coords={'subjectID':Subjects,'Stimulus': Stimuli})
        self.meanfix_xy = xr.DataArray(MeanFixXY, dims=('subjectID','Stimulus','XY'), coords={'subjectID':Subjects,'Stimulus': Stimuli, 'XY':['X','Y']})
        self.sdfix_xy = xr.DataArray(SDFixXY, dims=('subjectID','Stimulus','XY'), coords={'subjectID':Subjects,'Stimulus': Stimuli, 'XY':['X','Y']})
        self.bounds=Bounds
        return Stimuli,Subjects
    
  
  
  
    
   



#  class ends here    


