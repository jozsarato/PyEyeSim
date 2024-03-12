
import numpy as np
from numpy import matlib
#from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
from .statshelper import SaliencyMapFilt,SaccadesTrial
from .scanpathshelpdebug import CreatAoiRects,SaccadeLine
import platform


def GetParams(self):
    """ Get stimulus and subject info of dataset """  
    assert  'subjectID' in self.data.columns , 'subjectID column not found- DataInfo(subjectID=Your Column)'
    assert  'Stimulus' in self.data.columns, 'Stimulus column not found- DataInfo(Stimulus=Your Column'

    self.subjects=np.unique(self.data['subjectID'].to_numpy())
    self.stimuli=np.unique(self.data['Stimulus'].to_numpy())

    self.ns,self.np=len(self.subjects),len(self.stimuli)
    return  self.subjects,self.stimuli

def InferSize(self,Interval=99):
    ''' Infer stimulus size as central Interval % fixations data'''
    BoundsX=np.zeros((len(self.stimuli),2))
    BoundsY=np.zeros((len(self.stimuli),2))
    for cp,p in enumerate(self.stimuli):
        Idx=np.nonzero(self.data['Stimulus'].to_numpy()==p)[0]
        BoundsX[cp,:]=np.percentile(self.data['mean_x'].to_numpy()[Idx],[(100-Interval)/2,Interval+(100-Interval)/2])
        BoundsY[cp,:]=np.percentile(self.data['mean_y'].to_numpy()[Idx],[(100-Interval)/2,Interval+(100-Interval)/2])
        
        if BoundsX[cp,0]<0:  
            BoundsX[cp,0]=0. ## out of area bounds are replaced with screen size
        if BoundsY[cp,0]<0:
            BoundsY[cp,0]=0  ## out of area bounds are replaced with screen size
        if BoundsX[cp,1]>self.x_size:
            BoundsX[cp,1]=self.x_size  ## out of area bounds are replaced with screen size
        if BoundsY[cp,1]>self.y_size:
            BoundsY[cp,1]=self.y_size  ## out of area bounds are replaced with screen size
    BoundsX=np.intp(np.round(BoundsX))
    BoundsY=np.intp(np.round(BoundsY))
    #self.boundsX=BoundsX
    #self.boundsY=BoundsY
    return BoundsX,BoundsY

def GetStimuli(self,extension,path=0,infersubpath=False):
    ''' load stimulus files from path'''
    #assert 'Stimulus' in self.data.columns, 'stimulus column not found'
    assert len(self.stimuli)>0, '!stimuli not loaded!  provide: DataInfo(Stimulus=Your Column)'

    self.images={}
    if infersubpath==True:
        if 'category' in self.data:
            self.data.rename(columns={'category':'Category'},inplace=True)
        print('infer path from database categeory')
    for cs,s in enumerate(self.stimuli):
        
        if infersubpath==True:
            cat=int(np.unique(self.data['Category'][self.data['Stimulus']==s])[0])
            if platform.platform().find('mac')>-1:
                p=str(cat)+'/'
            else:
                p=str(cat)+'\\'
            print(cs,s,p)
            Stim=plt.imread(path+p+str(int(s))+extension)

        else:
            if type(s)==str:
              #  print(path+s+extension)
                try: 
                    Stim=plt.imread(path+s+extension)
                except:  # because sometimes stimulus name already contains the extension
                    Stim=plt.imread(path+s)

            else:   
               # print(path+str(int(s))+extension)
                try:
                    Stim=plt.imread(path+str(int(s))+extension)
                except:  # because sometimes stimulus name already contains the extension
                    Stim=plt.imread(path+str(int(s)))


       # else:
        #    if type(s)!=str:
         #       print(p+str(int(s))+extension)
          #      Stim=plt.imread(p+str(int(s))+extension)
                
        Res=np.shape(Stim)
        if Res[0] != self.y_size:
            print("!y size incosistency warning expected",self.y_size,'vs actual', Res)
        if Res[1] != self.x_size:
            print("!x size incosistency warning, expected",self.x_size,'vs actual', Res)
        
        self.images[s]=Stim
    pass 
 

  
def FixCountCalc(self,Stim,CutAct=1,substring=False):
    ''' Pixelwise fixation count for each participant, but for single stimulus  (Stim) 
    output: subjects*y*x --> num of fixaiton for each pixel
    if CutAct==1 in the end, only the within bounds areas is returned for further calculations
    optional parameter, substring for stimulus names containing the same substring'''
    if substring==False:
        assert np.sum(self.data['Stimulus']==Stim)>0, 'stimulus not found'
        stimn=np.nonzero(self.stimuli==Stim)[0]
        print('stimns found:',stimn,Stim)

    elif substring==True:  
        self.stimuli=self.stimuli.astype('str')
        stimn=np.char.find(self.stimuli,Stim)
        Stims=self.stimuli[stimn>-1]
        stimn=np.nonzero(stimn>-1)[0]
        print('stimns found:',stimn,Stims)

    FixCountInd=np.zeros(((self.ns,self.y_size,self.x_size)))
 
    
    for cs,s in enumerate(self.subjects):
        if substring:
            x,y=np.intp(self.GetFixationData(s,Stims[0]))
            if len(x)==0: # if length of first match is zero
                x,y=np.intp(self.GetFixationData(s,Stims[1])) # get second match
                stimIdx=1
            else:
                stimIdx=0

            Valid=np.nonzero((x<self.boundsX[stimn[stimIdx],1])&(x>self.boundsX[stimn[stimIdx],0])&(y>self.boundsY[stimn[stimIdx],0])&(y<self.boundsY[stimn[stimIdx],1]))[0]

        else:
            x,y=np.intp(self.GetFixationData(s,Stim))
            Valid=np.nonzero((x<self.boundsX[stimn,1])&(x>self.boundsX[stimn,0])&(y>self.boundsY[stimn,0])&(y<self.boundsY[stimn,1]))[0]
        X,Y=x[Valid],y[Valid]
        FixCountInd[cs,Y,X]+=1
   # self.boundsX[stimn,0]:self.boundsX[stimn,1]
    if CutAct:
        if len(stimn)>0:
            stimn=stimn[0] # cut based on the bounds of the first if there are more matching stimuli
        FixCountInd=FixCountInd[:,:,int(np.round(self.boundsX[stimn,0])):int(np.round(self.boundsX[stimn,1]))]  # cut X
        FixCountInd=FixCountInd[:,int(np.round(self.boundsY[stimn,0])):int(np.round(self.boundsY[stimn,1])),:] # cut Y
    return FixCountInd



def GetFixationData(self,subj,stim,timemin=0,timemax=np.inf,timecol=0):
    """get X,Y fixation sequence for a subject and stimulus
    output 1: array of pixel x for sequence of fixations
    output 2: array of pixel y for sequence of fixations"""
    SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==subj)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==stim),SubjIdx) # idx for subject and painting
    if type(timecol)!=int:
        TimeIdx=np.nonzero((self.data[timecol]>timemin)&(self.data[timecol]<timemax))[0]
        TrialSubIdx=np.intersect1d(TrialSubIdx, TimeIdx)
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

def GetSaccades(self):
    ''' from fixations, make approximate saccades, and store it as saccade objects'''
    SaccadeObj=[]
    
    self.nsac=np.zeros((self.ns,self.np))
    self.saccadelenghts=np.zeros((self.ns,self.np))
    for cs,s in enumerate(self.subjects):
            SaccadeObj.append([])        
            for cp,p in enumerate(self.stimuli):
                SaccadeObj[cs].append([])
                FixTrialX,FixTrialY=self.GetFixationData(s,p)
                StartTrialX,StartTrialY,EndTrialX,EndTrialY=SaccadesTrial(FixTrialX,FixTrialY)
                SaccadesSubj=np.column_stack((StartTrialX,StartTrialY,EndTrialX,EndTrialY)) 
                csac=0
                for sac in range(len(StartTrialX)):
                    if np.isfinite(SaccadesSubj[sac,0])==True:
                        SaccadeObj[cs][cp].append(SaccadeLine(SaccadesSubj[sac,:]))  # store saccades as list of  objects 
                        self.saccadelenghts[cs,cp]+=SaccadeObj[cs][cp][-1].length()   
                        csac+=1
                self.nsac[cs,cp]=csac  # num of saccades for each participant and painting
                if csac>0:
                    self.saccadelenghts[cs,cp]/=csac
            
                else:
                    self.saccadelenghts[cs,cp]=np.NAN
    return SaccadeObj


def GetEntropies(self,fixsize=0,binsize_h=50):
    ''' calcualte grid based entropy for all stimuli 
    if fixsize=0, bounds are inferred from range of fixations
    output 1: entropy for stimulus across partcipants
    output 2: max possible entropy for each stimulus-- assuming different stimulus sizes
    output 3: individual entropies for each stimlus (2d array: subjects*stimuli)
    
    '''
    if hasattr(self,'boundsX')==False:
            print('runnnig descriptives to get bounds')
            self.RunDescriptiveFix()  
    self.entropies=np.zeros(self.np)
    self.entropmax=np.zeros(self.np)
    self.entropies_ind=np.zeros((self.ns,self.np))
    # self.fixcounts={}
    # for ci,i in enumerate(self.stimuli):
    #     self.fixcounts[i]=[]
    
    for cp,p in enumerate(self.stimuli):
        FixCountInd=self.FixCountCalc(p)
       # self.fixcounts[p]=FixCountInd
        binnedcount=self.BinnedCount(np.nansum(FixCountInd,0),p,fixs=fixsize,binsize_h=binsize_h)
        self.entropies[cp],self.entropmax[cp]=self.Entropy(binnedcount)
        for cs,s in enumerate(self.subjects):
            binnedc_ind=self.BinnedCount(FixCountInd[cs,:,:],p,fixs=fixsize)
            self.entropies_ind[cs,cp],EntroMax=self.Entropy(binnedc_ind)
        
        print(cp,p,np.round(self.entropies[cp],2),'maximum entropy',np.round(self.entropmax[cp],2))
    return self.entropies,self.entropmax,self.entropies_ind


  
def Heatmap(self,Stim,SD=25,Ind=0,Vis=0,FixCounts=0,cutoff='median',CutArea=0,ax=False,center=0,substring=False,cmap='plasma',alpha=.5):
    ''' Pipeline for  heatmap calculation, FixCounts are calculated for stimulus, or passed pre-calcualted as optional parameter
    output: heatmap for a stimulus
    
    Vis:  if 1 Heatmap visual shows up- otherwise no visualization, but returns the heatmap values
    
    cutarea option: 1 only use active area (99% percentile of fixations), 0- use all of the area - set it to 1, if stimulus does not cover the whole screen
    cutoff=median: median cutoff, otherwise percetile of values to replace with nans, goal--> clear visualization
    center: if set to 1, if pixel coordinates dont match, painting presented centrally, but gaze coors are zero based
    substring: use part of file name (expected for mathcing paired files)
    cmap=colormap (see matplotlib colormaps for options: https://matplotlib.org/stable/users/explain/colors/colormaps.html)
    alpha= transparency- 0-1 higher values less transparent
    '''
  #  if hasattr(self,'fixcounts'):
   #     FixCountIndie=self.fixcounts['Stim']
    #else:    
    if substring==False:
        stimn=np.nonzero(self.stimuli==Stim)[0]
        stimShow=Stim
    else:
        self.stimuli=self.stimuli.astype('str')
        stimn=np.char.find(self.stimuli,Stim)
        Stims=self.stimuli[stimn>-1]
        stimn=np.nonzero(stimn>-1)[0]
        stimShow=Stims[0]


    if hasattr(self,'boundsX')==False:
        print('run RunDescriptiveFix first- without visuals')
        self.RunDescriptiveFix()
    if type(FixCounts)==int:
        if CutArea:
            FixCounts=self.FixCountCalc(Stim,CutAct=1,substring=substring) 
        else:
            FixCounts=self.FixCountCalc(Stim,CutAct=0,substring=substring) 
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
            if len(stimn)>0:
                stimn=stimn[0]
            smapall[int(self.boundsY[stimn,0]):int(self.boundsY[stimn,1]),int(self.boundsX[stimn,0]):int(self.boundsX[stimn,1])]=smap
        else:
            smapall=np.copy(smap)
    else:
        smap=np.zeros_like(FixCounts)
        for cs,s in enumerate(self.subjects):
            smap[cs,:,:]=SaliencyMapFilt(FixCounts[cs,:,:],SD=SD,Ind=1)       
    if Vis:
        smapall[smapall<cutThr]=np.NAN  # replacing below threshold with NAN
        xs1=(self.x_size-np.shape(self.images[stimShow])[1])/2
        xs2=self.x_size-xs1
        ys1=(self.y_size-np.shape(self.images[stimShow])[0])/2
        ys2=self.y_size-ys1
        if ax==False:
            fig,ax=plt.subplots()
        if center:
            ax.imshow(self.images[stimShow],extent=[xs1,xs2,ys2,ys1])
        else:
            ax.imshow(self.images[stimShow])
        ax.imshow(smapall,alpha=alpha,cmap=cmap) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xs1,xs2])
        ax.set_ylim([ys2,ys1])
            
    return smapall

