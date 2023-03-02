# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:40:42 2022

@author: aratoj87
"""
import numpy as np
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
import pickle
import xarray as xr
import matplotlib.ticker as ticker
from math import atan2, degrees
import hmmlearn.hmm  as hmm
from matplotlib.patches import Ellipse

#%%


class EyeData:
    def __init__(self, name, design,data,x_size,y_size,fixdata=1):
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
                print(df," not found !!, provide column as", df,"=YourColumn default",DefColumns[df])
        

    def info(self):
        return self.name,self.design

    def data(self):
        return self.data
    
    def GetParams(self):
        """ Get stimulus and subject info of dataset """  
        self.subjects=np.unique(self.data['subjectID'].to_numpy())
        self.stimuli=np.unique(self.data['Stimulus'].to_numpy())

        self.ns,self.np=len(self.subjects),len(self.stimuli)
        return  self.subjects,self.stimuli
    
    
    def DataInfo(self,StimName='Stimulus',SubjName='subjectID',mean_x='mean_x',mean_y='mean_y',FixDuration=0,StimPath=0,StimExt='.jpg'):
        ''' the library expects column names Stimulus, subjectID, mean_x and mean_y, if you data is not in this format, this function will rename your columns accordingly 
         optionally, with FixDuration you can name your column of fixations lengths, which will be called duration afterwards'''
       # print(type(FixDuration))
       
        if self.fixdata:
            if type(FixDuration)!='int':
                self.data=self.data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y',FixDuration: 'duration'})
            else:
                self.data=self.data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y'})
        else:
            if type(FixDuration)!='int':
                self.data=self.data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',FixDuration: 'duration'})
            else:
                self.data=self.data.rename(columns={StimName:'Stimulus',SubjName:'subjectID'})
        
        try:
            subjs,stims=self.GetParams()
            print('info found for '+ str(len(subjs))+' subjects, and '+str(len(stims))+' stimuli')
            
        except:
            print('stimulus and subject info not found')
            
        if StimPath==0:
            print('Stim path not provided')
        else:
         #  try: 
            self.GetStimuli(StimExt,StimPath)
            print('stimuli loaded succesfully, access as self.images')
          # except:   
           #    print('stimuli not found')
        pass
  
    
    def GetStimuli(self,extension,path=0):
        ''' load stimuulus files from path'''
        self.images={}
        if path=='infer':
            if 'category' in self.data:
                self.data.rename(columns={'category':'Category'},inplace=True)
            print('infer path from database categeory')

        for cs,s in enumerate(self.stimuli):
            if path=='infer':
                cat=int(np.unique(self.data['Category'][self.data['Stimulus']==s])[0])
                p=str(cat)+'\\'
                print(cs,s,p)
        
            if path!='infer':
                print(path+s+extension)
                Stim=plt.imread(path+s+extension)
            else:
                if type(s)!=str:
                    print(p+str(int(s))+extension)
                    Stim=plt.imread(p+str(int(s))+extension)
                    
            Res=np.shape(Stim)
            if Res[0] != self.y_size:
                print("!y size incosistency warning expected",self.y_size,'vs actual', Res)
            if Res[1] != self.x_size:
                print("!x size incosistency warning, expected",self.x_size,'vs actual', Res)
            
            self.images[s]=Stim
        pass 
 
    def GetFixationData(self,s,p):
        """get X,Y fixation sequence for a participant and stimulus
        output 1: array of pixel x for sequence of fixations
        output 2: array of pixel y for sequence of fixations"""
        SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==s)  #idx for subject
        TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==p),SubjIdx) # idx for subject and painting
        FixTrialX=np.array(self.data['mean_x'].iloc[TrialSubIdx]) # get x data for trial
        FixTrialY=np.array(self.data['mean_y'].iloc[TrialSubIdx]) # get y data for trial
        return FixTrialX,FixTrialY
    
    def GetDurations(self,s,p):
        """get fixations durations for a trials
        output: array of fixation durations """
        SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==s)  #idx for subject
        TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==p),SubjIdx) # idx for subject and painting
         # get x data for trial
        durations=np.array(self.data['duration'].iloc[TrialSubIdx]) # get y data for trial
        return durations
    
    
    def InferSize(self,Interval=99):
        ''' Infer stimulus size as central Interval % fixations data'''
        BoundsX=np.zeros((len(self.stimuli),2))
        BoundsY=np.zeros((len(self.stimuli),2))
        for cp,p in enumerate(self.stimuli):
            Idx=np.nonzero(self.data['Stimulus'].to_numpy()==p)[0]
            BoundsX[cp,:]=np.percentile(self.data['mean_x'].to_numpy()[Idx],[(100-Interval)/2,Interval+(100-Interval)/2])
            BoundsY[cp,:]=np.percentile(self.data['mean_y'].to_numpy()[Idx],[(100-Interval)/2,Interval+(100-Interval)/2])
            
            if BoundsX[cp,0]<0:
                BoundsX[cp,0]=0
                print(p,' Bound below zero X found indicating out of stimulus area fixations-replaced with 0')
            if BoundsY[cp,0]<0:
                BoundsY[cp,0]=0
                print(p,' Bound below zeroY found indicating out of stimulus area fixations-replaced with 0')    
            if BoundsX[cp,1]>self.x_size:
                BoundsX[cp,1]=self.x_size
                print(p,' Bound over x_size found indicating out of stimulus area fixations-replaced with', self.x_size)
            if BoundsY[cp,1]>self.y_size:
                BoundsY[cp,1]=self.y_size
                print(p,' Bound over y_size found indicating out of stimulus area fixations-replaced with',self.y_size)    
        BoundsX=np.intp(np.round(BoundsX))
        BoundsY=np.intp(np.round(BoundsY))
        return BoundsX,BoundsY
    
    

    def RunDescriptiveFix(self,Visual=0,duration=0):
        ''' for a dataset, return number of fixation, inferred stim boundaries and mean and SD of fixation locatios '''
        
        Subjects,Stimuli=self.GetParams()
        print('Data for ',len(self.subjects),'observers and ', len(self.stimuli),' stimuli.')
        self.boundsX,self.boundsY=self.InferSize(Interval=99)
        self.actsize=(self.boundsX[:,1]-self.boundsX[:,0])*(self.boundsY[:,1]-self.boundsY[:,0])
        self.nfixations=np.zeros((self.ns,self.np))
        
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
                        
        print('Mean fixation number: ',np.round(np.mean(np.mean(self.nfixations,1)),2),' +/- ',np.round(np.std(np.mean(self.nfixations,1)),2))
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
    
    
    def FixCountCalc(self,Stim,CutAct=1):
        ''' Pixelwise fixation count for each participant, but for single stimulus  (Stim) 
        output: subjects*y*x --> num of fixaiton for each pixel
        if CutAct==1 in the end, only the within bounds areas is returned for further calculations'''
        assert np.sum(self.data['Stimulus']==Stim)>0, 'stimulus not found'
       
        stimn=np.nonzero(self.stimuli==Stim)[0]
        FixCountInd=np.zeros(((self.ns,self.y_size,self.x_size)))
       # sizy=round(self.boundsY[stimn,1]-self.boundsY[stimn,0])
       # sizx=round(self.boundsX[stimn,1]-self.boundsX[stimn,0])

       # FixCountInd=np.zeros(((self.ns,sizy,sizx)))
        
        for cs,s in enumerate(self.subjects):
            x,y=np.intp(self.GetFixationData(s,Stim))
            Valid=np.nonzero((x<self.boundsX[stimn,1])&(x>self.boundsX[stimn,0])&(y>self.boundsY[stimn,0])&(y<self.boundsY[stimn,1]))[0]
            X,Y=x[Valid],y[Valid]
            FixCountInd[cs,Y,X]+=1
       # self.boundsX[stimn,0]:self.boundsX[stimn,1]
        if CutAct:
            FixCountInd=FixCountInd[:,:,int(np.round(self.boundsX[stimn,0])):int(np.round(self.boundsX[stimn,1]))]  # cut X
            FixCountInd=FixCountInd[:,int(np.round(self.boundsY[stimn,0])):int(np.round(self.boundsY[stimn,1])),:] # cut Y
        return FixCountInd
   
    
    def Heatmap(self,Stim,SD=25,Ind=0,Vis=0,FixCounts=0,cutoff='median',CutArea=0):
        ''' Pipeline for  heatmap calculation, FixCounts are calculated for stimulus, or passed pre-calcualted as optional parameter
        output: heatmap for a stimulus
        cutarea option: 1 only use active area (99% percentile of fixations), 0- use all of the area 
        cutoff=median: median cutoff, otherwise percetile of values to replace with nans, goal--> clear visualization'''
      #  if hasattr(self,'fixcounts'):
       #     FixCountIndie=self.fixcounts['Stim']
        #else:    
        stimn=np.nonzero(self.stimuli==Stim)[0]

        if type(FixCounts)==int:
            if CutArea:
                FixCounts=self.FixCountCalc(Stim,CutAct=1) 
            else:
                FixCounts=self.FixCountCalc(Stim,CutAct=0) 
        assert np.sum(FixCounts)>0,'!!no fixations found'
 
        if np.sum(FixCounts)<10:
            print('WARNING NUM FIX FOUND: ',np.sum(FixCounts))
        if Ind==0:
            smap=SaliencyMapFilt(FixCounts,SD=SD,Ind=0)
            if cutoff=='median':
                 cutThr=np.median(smap)
            elif cutoff>0:
                 cutThr=np.percentile(smap,cutoff) 
            else:
                cutThr=0
            if CutArea:
                smapall=np.zeros((self.y_size,self.x_size))
                smapall[int(self.boundsY[stimn,0]):int(self.boundsY[stimn,1]),int(self.boundsX[stimn,0]):int(self.boundsX[stimn,1])]=smap
            else:
                smapall=np.copy(smap)
        else:
            smap=np.zeros_like(FixCounts)
            for cs,s in enumerate(self.subjects):
                smap[cs,:,:]=SaliencyMapFilt(FixCounts[cs,:,:],SD=SD,Ind=1)       
        if Vis:
            smapall[smapall<cutThr]=np.NAN  # replacing below threshold with NAN
            plt.imshow(self.images[Stim])
            plt.imshow(smapall,alpha=.5)        
            plt.xticks([])
            plt.yticks([])
        return smapall
    
  
    
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
    
    
    def GetEntropies(self,fixsize=0,binsize_h=50):
        ''' calcualte grid based entropy for all stimuli 
        if fixsize=0, bounds are inferred from range of fixations
        output 1: entropy for stimulus across partcipants
        output 2: max possible entropy for each stimulus-- assuming different stimulus sizes
        output 3: individual entropies for each stimlus (2d array: subjects*stimuli)
        
        '''
        self.entropies=np.zeros(self.np)
        self.entropmax=np.zeros(self.np)
        self.entropies_ind=np.zeros((self.ns,self.np))
        # self.fixcounts={}
        # for ci,i in enumerate(self.stimuli):
        #     self.fixcounts[i]=[]
        
        for cp,p in enumerate(self.stimuli):
            FixCountInd=self.FixCountCalc(p)
           # self.fixcounts[p]=FixCountInd
            binnedcount=self.BinnedCount(np.sum(FixCountInd,0),p,fixs=fixsize,binsize_h=binsize_h)
            self.entropies[cp],self.entropmax[cp]=self.Entropy(binnedcount)
            for cs,s in enumerate(self.subjects):
                binnedc_ind=self.BinnedCount(FixCountInd[cs,:,:],p,fixs=fixsize)
                self.entropies_ind[cs,cp],EntroMax=self.Entropy(binnedc_ind)
            
            print(cp,p,np.round(self.entropies[cp],2),'maximum entropy',np.round(self.entropmax[cp],2))
        return self.entropies,self.entropmax,self.entropies_ind
    

    def GetGroups(self,betwcond):
        ''' Between group comparison- 2 groups expected
        get conditions from between group column, check if mapping of participants to conditions is unique'''
        self.Conds=np.unique(self.data[betwcond])
        print('Conditions',self.Conds)
       
      #  assert len(self.Conds)==2, 'you need 2 groups'
        WhichC=np.zeros(self.ns)
        WhichCN=[]
        for cs,s in enumerate(self.subjects):
            for cc,c in enumerate(self.Conds):
                PPc=np.unique(self.data[betwcond][self.data['subjectID']==s])
                assert len(PPc)==1,'participant condition mapping not unique'
                if PPc==self.Conds[cc]:
                    WhichC[cs]=cc
                    WhichCN.append(c)
        self.whichC=WhichC
        return WhichC,np.array(WhichCN)

    def GetCats(self,condColumn):
        ''' Between group comparison- 2 groups expected
        get conditions from between group column, check if mapping of participants to conditions is unique'''
        self.WithinConds=np.unique(self.data[condColumn])
        print('Conditions',self.WithinConds)
        WhichCat=[]# np.zeros(self.np)

        for cp,p in enumerate(self.stimuli):
            AssignCat=np.unique(self.data[condColumn][self.data['Stimulus']==p])
           # print(cp,p,AssignCat)
            #assert len(AssignCat)==1, ' category mapping not unique for a stimulus'
            WhichCat.append(AssignCat)
        WhichCat=np.array(WhichCat)
        assert len(np.unique(WhichCat))==len(np.unique(self.WithinConds)), 'stimulus category mapping problem'
        return WhichCat
    


    def CompareGroupsFix(self,betwcond):
        '''run set of between group fixation comparisons, makes plots and prints descriptive stats'''
        
        WhichC,WhichCN=self.GetGroups(betwcond)
        if hasattr(self,'entropies')==False:   # check if entropy has already been calculated
            print('Calculating entropy')
            Entropies,self.entropmax,self.entropies_ind=self.GetEntropies()
        Cols=['darkred','cornflowerblue']
        plt.figure(figsize=(8,8))
        
        Entrs=[]
        Fixies=[]
        ScanpLs=[]
        SaccAmpls=[] 
        for cc,c in enumerate(self.Conds):
            Idx=np.nonzero(WhichC==cc)[0]
            FixGr=np.array(self.nfix[Idx,:])
            EntrGr=self.entropies_ind[Idx,:]
            Entrs.append(np.nanmean(EntrGr,1))
            Fixies.append(np.nanmean(FixGr,1))
            ScanpLs.append(np.nanmean(self.len_scanpath[Idx,:],1))
            SaccAmpls.append(np.nanmean(self.sacc_ampl[Idx,:],1))
         
            
            print(cc,c,'Num fix= ',np.round(np.mean(np.nanmean(FixGr,1)),2),'+/-',np.round(np.std(np.nanmean(FixGr,1)),2))
            print(cc,c,'Entropy= ',np.round(np.mean(np.nanmean(EntrGr,1)),2),'+/-',np.round(np.std(np.nanmean(EntrGr,1)),2))
            print(cc,c,'tot scanpath len = ',np.round(np.mean(np.nanmean(self.len_scanpath[Idx,:],1)),2),'+/-',np.round(np.std(np.nanmean(self.len_scanpath[Idx,:],1)),2),'pix')
            print(cc,c,'saccade amplitude = ',np.round(np.mean(np.nanmean(self.sacc_ampl[Idx,:],1)),2),'+/-',np.round(np.std(np.nanmean(self.sacc_ampl[Idx,:],1)),2),'pix')

            plt.subplot(2,2,1)
            MeanPlot(self.np,FixGr,yLab='Num Fixations',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc])
            plt.subplot(2,2,2)
            MeanPlot(self.np,EntrGr,yLab='Entropy',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc])
            plt.subplot(2,2,3)
            MeanPlot(self.np,self.len_scanpath[Idx,:],yLab='tot scanpath len (pix)',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc])
            plt.subplot(2,2,4)
            MeanPlot(self.np,self.sacc_ampl[Idx,:],yLab='saccade amplitude (pix)',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc])
            
            
        t,p=stats.ttest_ind(Entrs[0],Entrs[1])
        print(' ')
        print('Overall group differences: ')
        print('Entropy t=',np.round(t,4),', p=',np.round(p,4))
        #if pglib:
         #   pg.ttest(Fixies[0],Fixies[1],paired=False)
        #else:
        t,p=stats.ttest_ind(Fixies[0],Fixies[1])
        print('Num Fix t=',np.round(t,4),', p= ',np.round(p,4))
        t,p=stats.ttest_ind(ScanpLs[0],ScanpLs[1])
        

        print('Scanpath lengths t=',np.round(t,4),', p=',np.round(p,4))
        t,p=stats.ttest_ind(SaccAmpls[0],SaccAmpls[1])

        print('Saccade amplitudes t=',np.round(t,4),', p=',np.round(p,4))


        plt.legend()
        plt.tight_layout()
        return 
    
    
    def CompareGroupsHeatmap(self,Stim,betwcond,StimPath='',SD=25,CutArea=0,Conds=0):
        ''' visualize group heatmap, along with heatmap difference 
        SD optional parameter of heatmap smoothness, in pixels!
        CutArea==1: use central area only with 99% of fixations
        Conds==0: use automatically detected conditions conditions, as provided in betweencond column
        othewise Conds=['MyCond1' MyCond2'], if we want to specify the order of access for betweencond column '''
        WhichC,WhichCN=self.GetGroups(betwcond)
        if hasattr(self,'subjects')==0:
            self.GetParams()    
        #Cols=['darkred','cornflowerblue']
        plt.figure(figsize=(10,5))
       # FixCounts=self.FixCountCalc(Stim)
        
        if CutArea:
            FixCounts=self.FixCountCalc(Stim,CutAct=1) 
        else:
            FixCounts=self.FixCountCalc(Stim,CutAct=0) 
        assert np.sum(FixCounts)>0,'!!no fixations found'
        hmaps=[]
        
        if type(Conds)==int:    
            Conditions=np.copy(self.Conds)
        else:
            print('use provided conditions: ' ,Conds)
            Conditions=np.copy(Conds)
        for cc,c in enumerate(Conditions):
            Idx=np.nonzero(WhichCN==c)[0]   
            plt.subplot(2,2,cc+1)
            hmap=self.Heatmap(Stim,SD=SD,Ind=0,Vis=1,FixCounts=FixCounts[Idx,:,:],CutArea=CutArea)
            plt.title(c)
            plt.colorbar()
            hmaps.append(hmap)
        plt.subplot(2,2,3)
        if hasattr(self,'images'):
            plt.imshow( self.images[Stim])

        Diff=hmaps[0]-hmaps[1]
        #plt.imshow(Diff,cmap='RdBu',alpha=.5)
        
        plt.imshow(Diff,cmap='RdBu', vmin=-np.nanmax(np.abs(Diff)), vmax=np.nanmax(np.abs(Diff)),alpha=.5)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(Conditions[0])+' - '+str(Conditions[1]))
        cbar=plt.colorbar()
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(str(Conditions[0])+'<---->'+str(Conditions[1]), rotation=270)
        plt.subplot(2,2,4)
        if hasattr(self,'images'):
            plt.imshow( self.images[Stim])
        plt.imshow(np.abs(Diff), vmin=0, vmax=np.nanmax(np.abs(Diff)),alpha=.5)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('Absolute diff: '+str(np.round(np.nansum(np.abs(Diff)),3)))
        plt.tight_layout()
        return 
    
    
    
    def CompareWithinGroupsFix(self,withinColName):
        '''run set of between group fixation comparisons, makes plots and prints descriptive stats'''
        
        WhichC=self.GetCats(withinColName)

        if hasattr(self,'entropies')==False:   # check if entropy has already been calculated
            print('Calculating entropy')
            Entropies,self.entropmax,self.entropies_ind=self.GetEntropies()
        #Cols=['darkred','cornflowerblue']
#        plt.figure(figsize=(8,8))
        
        Entrs=[]
        Fixies=[]
        ScanpLs=[]
        SaccAmpls=[] 
        for cc,c in enumerate(self.WithinConds):
            print(cc,'Category',c)
            Idx=np.nonzero(WhichC==c)[0]
            FixGr=np.array(self.nfix[:,Idx])
            EntrGr=self.entropies_ind[:,Idx]
            Entrs.append(np.nanmean(EntrGr,1))
            Fixies.append(np.nanmean(FixGr,1))
            ScanpLs.append(np.nanmean(self.len_scanpath[:,Idx],1))
            SaccAmpls.append(np.nanmean(self.sacc_ampl[:,Idx],1))
         
            
            print(cc,c,'Num fix= ',np.round(np.mean(np.nanmean(FixGr,1)),2),'+/-',np.round(np.std(np.nanmean(FixGr,1)),2))
            print(cc,c,'Entropy= ',np.round(np.mean(np.nanmean(EntrGr,1)),2),'+/-',np.round(np.std(np.nanmean(EntrGr,1)),2))
            print(cc,c,'tot scanpath len = ',np.round(np.mean(np.nanmean(self.len_scanpath[:,Idx],1)),2),'+/-',np.round(np.std(np.nanmean(self.len_scanpath[:,Idx],1)),2),'pix')
            print(cc,c,'saccade amplitude = ',np.round(np.mean(np.nanmean(self.sacc_ampl[:,Idx],1)),2),'+/-',np.round(np.std(np.nanmean(self.sacc_ampl[:,Idx],1)),2),'pix')
            print('')
        return
     
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
    
    
    
    def FixDurProgGroups(self,withinColName,nfixmax=10):
        self.FixDurProg(nfixmax=nfixmax,Stim=0,Vis=0)
        WhichC=self.GetCats(withinColName)
        for cc,c in enumerate(self.WithinConds):
            Idx=np.nonzero(WhichC==c)[0]
            Y=np.nanmean(np.nanmean(self.durprog[:,Idx],1),0)
            Err=stats.sem(np.nanmean(self.durprog[:,Idx],1),axis=0,nan_policy='omit')
            PlotDurProg(nfixmax,Y,Err,c)
        plt.legend()



    
    def AOIFix(self,p,FixTrialX,FixTrialY,nDivH,nDivV,InferS=1):
        """ given a sequence of X,Y fixation data and AOI divisions, calculate static N and p matrix) """ 
        nAOI=nDivH*nDivV
        AOInums=np.arange(nAOI).reshape(nDivV,nDivH)
        NFix=len(FixTrialX)  # num of data points
        # set AOI bounds
       # print(p,SizeGendX)
        if InferS==0:
            AOIboundsH=AOIbounds(0, self.x_size,nDivH)       
            AOIboundsV=AOIbounds(0, self.y_size,nDivV)  
        else:
            AOIboundsH=AOIbounds(self.boundsX[p,0], self.boundsX[p,1],nDivH)       
            AOIboundsV=AOIbounds(self.boundsY[p,0], self.boundsY[p,1],nDivV)   
       
        # set parameters & arrays to store data
        StatPtrial=np.zeros(nAOI) # static probabilities.
        StatNtrial=np.zeros(nAOI) # static counts.
    
       
        WhichAOIH=np.zeros(NFix)
        WhichAOIV=np.zeros(NFix)
        for x in range(NFix):
            WhichAOIH[x]=CheckCor(AOIboundsH,FixTrialX[x]) # store which horizontal AOI each fixation is
            WhichAOIV[x]=CheckCor(AOIboundsV,FixTrialY[x]) # store which vertical AOI each fixation is
    
        WhichAOI=np.zeros(NFix)
        WhichAOI[:]=np.NAN
        for x in range(NFix):
            if WhichAOIV[x]>-1 and WhichAOIH[x]>-1:   # only use valid idx
                WhichAOI[x]=AOInums[np.intp(WhichAOIV[x]),np.intp(WhichAOIH[x])]  # get combined vertival and horizontal
        for st in range(nAOI): # gaze transition start
            StatNtrial[st]=np.sum(WhichAOI==st)  # get count in AOI
            StatPtrial[st]=np.sum(WhichAOI==st)/np.sum(np.isfinite(WhichAOI)) # calculate stationary P for each AOI    
        return NFix,StatPtrial,StatNtrial
    
    
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
            #plt.errorbar(np.arange(self.np),Indmean,SD,marker='o',linestyle='none')
            if zscore:
                plt.scatter(np.arange(self.np),(Indmean-np.mean(Indmean))/np.std(Indmean),marker='o')
            else:
                plt.scatter(np.arange(self.np),Indmean,marker='o')
            plt.xticks(np.arange(self.np),self.stimuli,rotation=80,fontsize=12)
            plt.xlabel('Stimuli',fontsize=14)
            if zscore==1:
                plt.ylabel('fixation map relative difference',fontsize=14)
            else:
                plt.ylabel('fixation map difference',fontsize=14)
        return Indmean
      
    
    def GetInddiff_v2(self,size=50,Vis=0,fixs=0):
        ''' PIXE; NUMBER BASED; calculate individual similarity between all pairs of participants for all stimuli, for a given division'''
        statPMat=self.GetBinnedStimFixS(size=size,fixs=fixs)
        Inddiff=self.StatPDiffInd2(statPMat)
        Indmean=np.nanmean(Inddiff,2)
        SD=np.nanstd(Indmean,1)
        Indmean=np.nanmean(Indmean,1)
        if Vis:
            #plt.errorbar(np.arange(self.np),Indmean,SD,marker='o',linestyle='none')
            plt.scatter(np.arange(self.np),Indmean,marker='o')
            plt.xticks(np.arange(self.np),self.stimuli,rotation=80,fontsize=12)
            plt.xlabel('Stimuli',fontsize=14)
            plt.ylabel('fixation map difference',fontsize=14)
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
            plt.figure()
        DiffsRaw=np.zeros((self.np,maxdiv-mindiv))
        DiffsZscore=np.zeros((self.np,maxdiv-mindiv))
        for cdiv,divs in enumerate(np.arange(mindiv,maxdiv)):
            DiffsRaw[:,cdiv]=self.GetInddiff(divs,divs,Vis=Vis,zscore=1)
            DiffsZscore[:,cdiv]=(DiffsRaw[:,cdiv]-np.mean(DiffsRaw[:,cdiv]))/np.std(DiffsRaw[:,cdiv])
        if Vis:
            plt.errorbar(np.arange(self.np),np.mean(DiffsZscore,1),np.std(DiffsZscore,1),linestyle='none',color='k',marker='o',markersize=5)
        return DiffsZscore,DiffsRaw
    
    
    
    
    
    def BinnedDescriptives(self,length,binsize,timecol,durcol,startime=0):
        ''' time-binned within trial descriptive progression
        INPUTS
        length: maximum trial length of interest in ms
        binsize: length of timebin 
        timecol: name of column with time length information
        durcol: name of column with fixation duration information '''
        Bins=np.arange(startime,length+binsize,binsize)
        print(f'Bins {Bins}')
        self.tbins=Bins
        self.binFixL=np.zeros((self.ns,self.np,len(Bins)-1))
        self.saccadeAmp=np.zeros((self.ns,self.np,len(Bins)-1))
        self.totLscanpath=np.zeros((self.ns,self.np,len(Bins)-1))
        cb=0
        for bs,be in zip(Bins[0:-1],Bins[1:]):
            BindIdx=(self.data[timecol]>bs) & (self.data[timecol]<be)
            print(f'from {bs} to {be} found: ',np.sum(BindIdx))
            for cs,s in enumerate(self.subjects):
                 SubjIdx=self.data['subjectID']==s
                 for cp,p in enumerate(self.stimuli):
                     Idx=((self.data['Stimulus']==p) & BindIdx & SubjIdx)
                     self.binFixL[cs,cp,cb]=np.mean(self.data[durcol][Idx])
                     self.saccadeAmp[cs,cp,cb],self.totLscanpath[cs,cp,cb]=ScanpathL(self.data['mean_x'][Idx].to_numpy(), self.data['mean_y'][Idx].to_numpy())
            cb+=1
        
        plt.figure()
        VisBinnedProg(Bins,np.nanmean(self.binFixL,1),'fixation duration (ms)')  
        plt.figure()
        VisBinnedProg(Bins,np.nanmean(self.saccadeAmp,1),'saccade ampl (pixel)')  
        plt.figure()
        VisBinnedProg(Bins,np.nanmean(self.totLscanpath,1),'scanpath length (pixel)')  
        return self.binFixL,self.saccadeAmp,self.totLscanpath

    def BinnedDescStim(self,stimuli):
        if hasattr(self,'binFixL')==False: 
            print('run BinnedDescriptives first, than call this function fo r')
                    
       
    def BinnedDescriptivesGroups(self,withinColName):
        ''' time-binned within trial descriptive progression, groups of stimuli'''
        if hasattr(self,'binFixL')==False: 
            print('run BinnedDescriptives first, than call this function fo r')
        WhichC=self.GetCats(withinColName)
        Colors=['navy','salmon','olive','orange','gray']
        fig,ax=plt.subplots(nrows=3,ncols=1,figsize=(4,10))
        for cc,c in enumerate(self.WithinConds):
            Idx=np.nonzero(WhichC==c)[0]
            axout=VisBinnedProg(self.tbins,np.nanmean(self.binFixL[:,Idx,:],1),'fixation duration (ms)',col=Colors[cc],label=c,axin=ax[0])
            axout=VisBinnedProg(self.tbins,np.nanmean(self.saccadeAmp[:,Idx,:],1),'saccade ampl (pixel)',col=Colors[cc],label=c,axin=ax[1])
            axout=VisBinnedProg(self.tbins,np.nanmean(self.totLscanpath[:,Idx,:],1),'scanpath length (pixel)',col=Colors[cc],label=c,axin=ax[2])

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.tight_layout()      
        
        
        
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

     # hmm related functions start here
    def DataArrayHmm(self,stim,group=-1,tolerance=20):
        ''' HMM data arrangement, for the format required by hmmlearn
        tolarance control the numbers of pixels, where out of stimulus fixations are still accepted
        participants with invalid fixations are removed'''
        
        XX=np.array([])
        YY=np.array([])
        Lengths=np.array([],dtype=int)
        self.suseHMM=np.array([],dtype=int)
        for cs,s in enumerate(self.subjects):
            if group!=-1:
                if self.whichC[cs]==group:
                    useS=True
                else:
                    useS=False
            else:
                useS=True
            if useS:
                fixX,fixY=self.GetFixationData(s,stim)
              #  print(cs,s,fixX)
                if any(fixX<-tolerance) or any(fixX>self.x_size+tolerance) or any(fixY<-tolerance)or any(fixY>self.y_size+tolerance):
                    print('invalid fixation location for subj', s)
                else:
                    if len(fixX)>2:
                        XX=np.append(XX,fixX)
                        YY=np.append(YY,fixY)
                        Lengths=np.append(Lengths,len(fixX))
                        self.suseHMM=np.append(self.suseHMM,s)
                    else:
                        print('not enough fixations for subj', s)

        return XX,YY,Lengths

    
    def MyTrainTest(self,Dat,Lengths,ntest,vis=0,rand=1,totest=0):
        ''' separate hidden markov model dataset, into training and test set'''
        if rand:
            totest=np.random.choice(np.arange(len(Lengths)),size=ntest,replace=False)
        else:
            totest=np.array([totest],dtype=int)
        Idxs=np.cumsum(Lengths)
        lenTrain=np.array([],dtype=int)
        lenTest=Lengths[totest]
        DatTest=np.zeros((0,2))
        DatTr=np.zeros((0,2)) 
        for ci in range(len(Lengths)):
            if ci==0:
                start=0
            else:
                start=Idxs[ci-1]
            if ci in totest:
                DatTest=np.vstack((DatTest,Dat[start:Idxs[ci],:]))
            else:
                DatTr=np.vstack((DatTr,Dat[start:Idxs[ci],:]))
                lenTrain=np.append(lenTrain,Lengths[ci])
        if vis:
            self.MyTrainTestVis(DatTr,DatTest,lenTrain,lenTest,totest)
        return DatTr,DatTest,lenTrain,lenTest   
    
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
            

    def FitLOOHMM(self,ncomp,stim,covar='full'):
        ''' fit HMM, N subject times, leaving out once each time
        ncomp: number of components
        stim: stimulus code 
        covar: covariance type 'full' or  'tied' '''
        NTest=1
        xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80)
        Dat=np.column_stack((xx,yy))
        ScoresLOO=np.zeros(len(self.suseHMM))
        for cs,s in enumerate(self.suseHMM):
            DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=0,totest=cs)
            HMMfitted,sctr,scte=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)
            ScoresLOO[cs]=scte
        return Dat,lengths,ScoresLOO
    
    def VisLOOHMM(self,Dat,lengths,ScoresLOO,nshow=3,title='',showim=False,stimname=0):
        ''' visualize least and most typical sequences, from above fitted HMM'''
        Sorted=np.argsort(ScoresLOO)
        fig,ax=plt.subplots(ncols=nshow,nrows=2,figsize=(10,7))
        for a in range(nshow):
            if showim==True:
                ax[0,a].imshow(self.images[stimname])
                ax[1,a].imshow(self.images[stimname])
            DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,5,vis=0,rand=0,totest=Sorted[a])
            self.MySaccadeVis(ax[0,a],DatTest,lenTest,title='max'+str(a)+' logL: '+str(np.round(ScoresLOO[Sorted[a]],2)))
            DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,5,vis=0,rand=0,totest=Sorted[-a-1])
            self.MySaccadeVis(ax[1,a],DatTest,lenTest,title='min'+str(a)+' logL: '+str(np.round(ScoresLOO[Sorted[-a-1]],2)))
        plt.suptitle(title)
        plt.tight_layout() 
        
        
    def VisHMM(self,dat,hmmfitted,ax=0,showim=1,stim=0):
        if type(ax)==int:
           fig,ax= plt.subplots()
        if showim:
            ax.imshow(self.images[stim])
            alph=.5
        else:
            alph=.2
        ax.scatter(dat[:,0],dat[:,1],color='k',alpha=alph)
        ax.scatter(hmmfitted.means_[:,0],hmmfitted.means_[:,1],color='darkred',s=50)
        for c1 in range(hmmfitted.n_components):
            draw_ellipse((hmmfitted.means_[c1,0],hmmfitted.means_[c1,1]),hmmfitted.covars_[c1],ax=ax,facecolor='none',edgecolor='olive',linewidth=2)
            for c2 in range(hmmfitted.n_components):
                if c1!=c2:
                    ax.plot([hmmfitted.means_[c1,0],hmmfitted.means_[c2,0]],[hmmfitted.means_[c1,1],hmmfitted.means_[c2,1]],linewidth=hmmfitted.transmat_[c1,c2]*5,color='r')
        ax.set_ylim([self.y_size,0])
        ax.set_xlim([0,self.x_size])
        ax.set_yticks([])
        ax.set_xticks([])   
        
        
    def FitVisHMM(self,stim,ncomp=3,covar='full',ax=0,ax2=0,NTest=5,showim=False):
        ''' fit and visualize HMM -- beta version
        different random train - test split for each iteration-- noisy results'''
        xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80)
        Dat=np.column_stack((xx,yy))
        
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=1)


        HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)

        if type(ax)==int:
            fig,ax=plt.subplots()
        if type(ax2)==int:
            fig,ax2=plt.subplots()
        self.VisHMM(DatTr,HMMfitted,ax=ax,showim=showim,stim=stim)
        ax.set_title('n: '+str(ncomp)+' train ll: '+str(np.round(meanscore,2))+' test ll: '+str(np.round(meanscoreTe,2)),fontsize=9)
        ax2.scatter(ncomp,meanscore,color='g')
        ax2.scatter(ncomp,meanscoreTe,color='r')
        ax2.set_xlabel('num components')
        ax2.set_ylabel('log likelihood')
      
        return meanscore,meanscoreTe
        
    def FitVisHMMGroups(self,stim,betwcond,ncomp=3,covar='full',ax=0,ax2=0,NTest=3,showim=False,Rep=1,groupnames=0):
        ''' fit and visualize HMM -- beta version
        different random train - test split for each iteration-- noisy results'''
        self.GetGroups(betwcond)
        Grs=np.unique(self.data[betwcond])
        
        fig,ax=plt.subplots(ncols=len(Grs),figsize=(12,5))
        fig,ax2=plt.subplots(ncols=2,figsize=(7,3))

        # data arrangement for groups
      
          
        
        ScoresTrain=np.zeros((Rep,len(Grs),len(Grs)))
        ScoresTest=np.zeros((Rep,len(Grs),len(Grs)))
       
       
        for rep in range(Rep):  
            XXTrain=[]
            LengthsTrain=[]
            XXTest=[]
            LengthsTest=[]
            for cgr,gr in enumerate(Grs):
                xx,yy,Lengths=self.DataArrayHmm(stim,group=gr,tolerance=50)
                Dat=np.column_stack((xx,yy))
                DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,Lengths,1,vis=0,rand=1,totest=NTest)
                XXTrain.append(DatTr)
                XXTest.append(DatTest)
                LengthsTrain.append(lenTrain)
                LengthsTest.append(lenTest)
            for cgr,gr in enumerate(Grs):
                HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,XXTrain[cgr],XXTest[cgr],LengthsTrain[cgr],LengthsTest[cgr],covar=covar)
                if rep==0:
                    self.VisHMM(XXTrain[cgr],HMMfitted,ax=ax[cgr],showim=showim,stim=stim)
                    ax[cgr].set_title(groupnames[cgr])
                for cgr2,gr2 in enumerate(Grs):
                    ScoresTrain[rep,cgr2,cgr]=HMMfitted.score(XXTrain[cgr2],LengthsTrain[cgr2])/np.sum(LengthsTrain[cgr2])
                    ScoresTest[rep,cgr2,cgr]=HMMfitted.score(XXTest[cgr2],LengthsTest[cgr2])/np.sum(LengthsTest[cgr2])

        im=ax2[0].pcolor(np.mean(ScoresTrain,0))
        ax2[0].set_title('training')
        plt.colorbar(im)
        im=ax2[1].pcolor(np.mean(ScoresTest,0))
        plt.colorbar(im)
        ax2[1].set_title('test')
        for pl in range(2):
            ax2[pl].set_xlabel('fitted')
            ax2[pl].set_ylabel('tested')
            ax2[pl].set_xticks(np.arange(len(Grs))+.5)
            if type(groupnames)==int:
                ax2[pl].set_xticklabels(Grs)
                ax2[pl].set_yticklabels(Grs)
            else:
                ax2[pl].set_xticklabels(groupnames)
                ax2[pl].set_yticklabels(groupnames)
            ax2[pl].set_yticks(np.arange(len(Grs))+.5)

        plt.tight_layout()


        # xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80)
      #  for cgr,gr in enumerate(np.unique(groups))):

        # Dat=np.column_stack((xx,yy))
        
        # DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=1)

        # HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)

        # if type(ax)==int:
        #     fig,ax=plt.subplots()
        # if type(ax2)==int:
        #     fig,ax2=plt.subplots()

        # if showim:
        #     ax.imshow(self.images[stim])
        #     alph=.5
        # else:
        #     alph=.2
        # ax.scatter(Dat[:,0],Dat[:,1],color='k',alpha=alph)
        # ax.scatter(HMMfitted.means_[:,0],HMMfitted.means_[:,1],color='darkred',s=50)
        # for c1 in range(ncomp):
        #     draw_ellipse((HMMfitted.means_[c1,0],HMMfitted.means_[c1,1]),HMMfitted.covars_[c1],ax=ax,facecolor='none',edgecolor='olive',linewidth=2)
        #     for c2 in range(ncomp):
        #         if c1!=c2:
        #             ax.plot([HMMfitted.means_[c1,0],HMMfitted.means_[c2,0]],[HMMfitted.means_[c1,1],HMMfitted.means_[c2,1]],linewidth=HMMfitted.transmat_[c1,c2]*5,color='r')
    
        # ax.set_ylim([self.y_size,0])
        # ax.set_xlim([0,self.x_size])
        # ax.set_yticks([])
        # ax.set_xticks([])
    
        # ax.set_title('n: '+str(ncomp)+' train'+str(np.round(meanscore,2))+' test'+str(np.round(meanscoreTe,2)),fontsize=9)
        
     
        # ax2.scatter(ncomp,meanscore,color='g')
        # ax2.scatter(ncomp,meanscoreTe,color='r')
        # ax2.set_xlabel('num components')
        # ax2.set_ylabel('log likelihood')
      
        return meanscore   
            
#  class ends here    



    

def FitScoreHMMGauss(ncomp,xx,xxt,lenx,lenxxt,covar='full'):
    HMM=hmm.GaussianHMM(n_components=ncomp, covariance_type=covar)
    HMM.fit(xx,lenx)
    sctr=HMM.score(xx,lenx)/np.sum(lenx)
    scte=HMM.score(xxt,lenxxt)/np.sum(lenxxt)
    return HMM,sctr,scte

def MeanPlot(N,Y,yLab=0,xtickL=0,newfig=1,color='darkred',label=None):
    ''' expects data, row: subjects columbn: stimuli '''
    if newfig:
        plt.figure(figsize=(N/2,5))
    plt.errorbar(np.arange(N),np.nanmean(Y,0),stats.sem(Y,0,nan_policy="omit")*2,linestyle='None',marker='o',color=color,label=label)
    if type(xtickL)!=int:
        plt.xticks(np.arange(N),xtickL,fontsize=9,rotation=60)
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


def AOIbounds(start,end,nDiv):  
    ''' calcuale AOI bounds, linearly spaced from: start to: end, for nDiv number of divisions'''
    return np.linspace(start,end,nDiv+1)  


def SaliencyMapFilt(Fixies,SD=25,Ind=0):
    ''' Gaussian filter of fixations counts, Ind=1 for individual, Ind=0 for group '''
    if Ind==0:
        Smap=ndimage.filters.gaussian_filter(np.mean(Fixies,0),SD)
    else:
        Smap=ndimage.filters.gaussian_filter(Fixies,SD)
    return Smap
    
def ScanpathL(x,y):
    ''' input 2 arrays for x and y ordered fixations
    output 1: average amplitude of  saccacdes
    output 2: total  length of scanpath'''
    x1=x[0:-1]
    x2=x[1:]
    y1=y[0:-1]
    y2=y[1:] 
    lengths=np.sqrt((x2-x1)**2+(y2-y1)**2)
    return np.mean(lengths),np.sum(lengths)

    
def CheckCor(AOIs,FixLoc):
    """ to check if fixation coordinates are within AOI """  
    for coor in range(len(AOIs)-1):
        if FixLoc>AOIs[coor] and FixLoc<=AOIs[coor+1]:
            AOI=coor
            break
        else: # if gaze out of screen
            AOI=np.NAN                      
    return AOI 


def StatEntropy(StatP): 
    """Calculate entropy of probability distribution
    without nans, result should be the same as scipy.stats.entropy with base=2"""
    LogP=np.log2(StatP)   
    LogP[np.isfinite(LogP)==0]=0   # replace nans with zeros    
    return -np.sum(StatP*LogP)


def PlotDurProg(nmax,Y,error,label=''):
    plt.fill_between(np.arange(1,nmax+1),Y-error,Y+error,alpha=.5)
    plt.plot(np.arange(1,nmax+1),Y,label=label)
    plt.xticks(np.intp(np.linspace(1,nmax,5)),np.intp(np.linspace(1,nmax,5)))
    plt.xlabel('fixation number',fontsize=14)
    plt.ylabel('fixation duration (ms)',fontsize=14)
    return 


def VisBinnedProg(bins,Y,ylabel,col='navy',label='',axin=0):
    if type(axin)==int:
        fig,axin=plt.subplots()
    axin.errorbar((bins[0:-1]+bins[1:])/2,np.nanmean(Y,0),stats.sem(Y,0,nan_policy='omit'),color=col,linewidth=2,label=label)
    axin.set_xlabel('time (ms)')
    axin.yaxis.set_major_locator(ticker.MaxNLocator(5))
    axin.set_ylabel(ylabel)
    return axin

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance
    source:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html """
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 2):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
