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

#%%


class EyeData:
    def __init__(self, name, design,data,x_size,y_size):
        self.name = name
        self.design = design
        self.data=data
        self.x_size=x_size
        self.y_size=y_size
        print('Fixation dataset',self.name)
        print('dataset size: ',np.shape(self.data))
        print('study design: ',self.design)
        print('presentation size:  x=',self.x_size,'pixels y=',self.y_size,' pixels')
        print('presentation size:  x=',self.x_size,'pixels y=',self.y_size,' pixels')
        DefColumns={'StimName':'Stimulus','SubjName':'subjectID','mean_x':'mean_x','mean_y':'mean_y'}
        for df in DefColumns:
            try:
                data[DefColumns[df]]
                print('column found: ', df,' default: ',DefColumns[df])
            except:
                print(df," not found !!, provide column as .DataInfo(StimName='YourColumn') default",DefColumns[df])
        

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
        if type(FixDuration)!='int':
            self.data=self.data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y',FixDuration: 'duration'})
        else:
            self.data=self.data.rename(columns={StimName:'Stimulus',SubjName:'subjectID',mean_x: 'mean_x',mean_y: 'mean_y'})
        
        try:
            subjs,stims=self.GetParams()
            print('info found for '+ str(len(subjs))+' subjects, and '+str(len(stims))+' stimuli')
            
        except:
            print('stimulus and subject info not found')
            
        if StimPath==0:
            print('Stim path not provided')
        else:
            try: 
                self.GetStimuli(StimPath,StimExt)
                print('stimuli loaded succesfully, access as self.images')
            except:
                
                print('stimuli not found')
        pass
  
    
    def GetStimuli(self,path,extension):
        ''' load stimuulus files from path'''
        self.images={}
        for cs,s in enumerate(self.stimuli):
            Stim=plt.imread(path+s+extension)
            self.images[s]=Stim
        pass 
        
    def GetFixationData(self,s,p):
        """get X,Y fixation sequence for a participant and stimulus"""
        SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==s)  #idx for subject
        TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==p),SubjIdx) # idx for subject and painting
        FixTrialX=np.array(self.data['mean_x'].iloc[TrialSubIdx]) # get x data for trial
        FixTrialY=np.array(self.data['mean_y'].iloc[TrialSubIdx]) # get y data for trial
        return FixTrialX,FixTrialY
    
    def GetDurations(self,s,p):
        """get X,Y fixation sequence for a participant and stimulus"""
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
        return BoundsX,BoundsY
    
    

    def RunDescriptiveFix(self,Visual=0,duration=0):
        ''' for a dataset, return number of fixation, inferred stim boundaries and mean and SD of fixation locatios '''
        
        Subjects,Stimuli=self.GetParams()
        print('Data for ',len(self.subjects),'observers and ', len(self.stimuli),' stimuli.')
        self.boundsX,self.boundsY=self.InferSize(Interval=99)
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
        self.nfix = xr.DataArray(self.nfixations, dims=('subjectID','Stimulus'), coords={'subjectID':Subjects,'Stimulus': Stimuli})
        self.meanfix_xy = xr.DataArray(MeanFixXY, dims=('subjectID','Stimulus','XY'), coords={'subjectID':Subjects,'Stimulus': Stimuli, 'XY':['X','Y']})
        self.sdfix_xy = xr.DataArray(SDFixXY, dims=('subjectID','Stimulus','XY'), coords={'subjectID':Subjects,'Stimulus': Stimuli, 'XY':['X','Y']})
        self.bounds=Bounds
        return Stimuli,Subjects
    
    
    def FixCountCalc(self,Stim):
        ''' Pixelwise fixation count for each participant, but for single stimulus  (Stim) '''
        assert np.sum(self.data['Stimulus']==Stim)>0, 'stimulus not found'
       
        FixCountInd=np.zeros(((self.ns,self.y_size,self.x_size)))
        for cs,s in enumerate(self.subjects):
            x,y=np.intp(self.GetFixationData(s,Stim))
            Valid=np.nonzero((x<self.x_size)&(y<self.y_size))[0]
            X,Y=x[Valid],y[Valid]
            FixCountInd[cs,Y,X]+=1
        return FixCountInd
    
    def FixDurProg(self,nfixmax=10,Stim=0):
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
            plt.figure()
            plt.fill_between(np.arange(nfixmax),Y-Err,Y+Err,alpha=.5)
            plt.plot(np.arange(nfixmax),Y,color='k')
            plt.xlabel('Fixation number')
            plt.ylabel('Fixation duration')
            plt.title('All stimuli')
            
        else:
            Y=np.nanmean(self.durprog[:,self.stimuli==Stim,:],0).flatten()
           # print(np.shape(Y))
            Err=stats.sem(self.durprog[:,self.stimuli==Stim,:],axis=0,nan_policy='omit').flatten()
           # print(np.shape(Err))
           # print(Y+Err)
            plt.figure()
            plt.fill_between(np.arange(nfixmax),Y-Err,Y+Err,alpha=.5)
            plt.plot(np.arange(nfixmax),Y,color='k')
            plt.xlabel('Fixation number')
            plt.ylabel('Fixation duration')
            plt.title(Stim)
            
        return None
    
    def Heatmap(self,Stim,SD=25,Ind=0,Vis=0,Idx=0,FixCounts=0):
        ''' Pipeline for saliency map calculation, FixCounts are calculated for stimulus, or passed pre-calcualted as optional parameter'''
      #  if hasattr(self,'fixcounts'):
       #     FixCountIndie=self.fixcounts['Stim']
        #else:    
        if type(FixCounts)==int:
            FixCounts=self.FixCountCalc(Stim)
            
        assert np.sum(FixCounts)>0,'!!no fixations found'
        if np.sum(FixCounts)<10:
            print('WARNING NUM FIX FOUND: ',np.sum(FixCounts))
        if Ind==0:
            smap=SaliencyMapFilt(FixCounts,SD=SD,Ind=0)
        else:
            smap=np.zeros_like(FixCounts)
            for cs,s in enumerate(self.subjects):
                smap[cs,:,:]=SaliencyMapFilt(FixCounts[cs,:,:],SD=SD,Ind=1)       
        if Vis:
            smap[smap<np.median(smap)]=np.NAN
            plt.imshow( self.images[Stim])
            plt.imshow(smap,alpha=.5)
            plt.xticks([])
            plt.yticks([])
        return smap
    
  
    
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
        ''' based on binned  2d fixation counts return entropy and relative entropy, default natural log'''
        assert len(np.shape(BinnedCount))==2,'2d data input expected'
        size=np.shape(BinnedCount)[0]*np.shape(BinnedCount)[1]
        entrMax=stats.entropy(1/size*np.ones(size),base=base)
        EntrBinned=stats.entropy(BinnedCount.flatten(),base=base)
        return EntrBinned,entrMax
    
    
    def GetEntropies(self,fixsize=0,binsize_h=50):
        ''' calcualte grid based entropy for all stimuli 
        if fixsize=0, bounds are inferred from range of fixations'''
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
        ''' Between group comparison- 2 groups expected'''
        self.Conds=np.unique(self.data[betwcond])
        print('Conditions',self.Conds)
       
        assert len(self.Conds)==2, 'you need 2 groups'
        WhichC=np.zeros(self.ns)
        WhichCN=[]
        for cs,s in enumerate(self.subjects):
            for cc,c in enumerate(self.Conds):
                PPc=np.unique(self.data[betwcond][self.data['subjectID']==s])
                assert len(PPc)==1,'participant condition mapping not unique'
                if PPc==self.Conds[cc]:
                    WhichC[cs]=cc
                    WhichCN.append(c)
        return WhichC,np.array(WhichCN)

    def CompareGroupsFix(self,betwcond):
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

        
        ScanpLs

        plt.legend()
        plt.tight_layout()
        return 
    
    
    def CompareGroupsHeatMap(self,Stim,betwcond,StimPath='',SD=25):
        ''' visualize group heatmap, along with heatmap difference 
        SD optional parameter of heatmap smoothness, in pixels!'''
        WhichC,WhichCN=self.GetGroups(betwcond)
        if hasattr(self,'subjects')==0:
            self.GetParams()    
        #Cols=['darkred','cornflowerblue']
        plt.figure(figsize=(10,5))
        FixCounts=self.FixCountCalc(Stim)
        hmaps=[]
        for cc,c in enumerate(self.Conds):
            Idx=np.nonzero(WhichC==cc)[0]
            plt.subplot(2,2,cc+1)
            if hasattr(self,'images'):
                plt.imshow( self.images[Stim])

            hmap=self.Heatmap(Stim,SD=SD,Ind=0,Vis=1,FixCounts=FixCounts[Idx,:,:])
            plt.title(c)
            plt.colorbar()

            hmaps.append(hmap)
        plt.subplot(2,2,3)
        if hasattr(self,'images'):
            plt.imshow( self.images[Stim])

        Diff=hmaps[0]-hmaps[1]
        plt.imshow(Diff,cmap='RdBu',alpha=.5)

       # plt.imshow(Diff,cmap='RdBu', vmin=-np.max(np.abs(Diff)), vmax=np.max(np.abs(Diff)),alpha=.5)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(self.Conds[0])+' - '+str(self.Conds[1]))
        cbar=plt.colorbar()
        cbar.ax.get_yaxis().set_ticks([])
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(str(self.Conds[0])+'<---->'+str(self.Conds[1]), rotation=270)
        plt.subplot(2,2,4)
        if hasattr(self,'images'):
            plt.imshow( self.images[Stim])

        plt.imshow(np.abs(hmaps[0]-hmaps[1]),alpha=.5)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('Absolute difference')
        plt.tight_layout()
        return 
    

        
    pass
  
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
    output 1 average length of  saccacdes
    output 2 total  length of scanpath'''
    x1=x[0:-1]
    x2=x[1:]
    y1=y[0:-1]
    y2=y[1:] 
    lengths=np.sqrt((x2-x1)**2+(y2-y1)**2)
    return np.mean(lengths),np.sum(lengths)
