
import numpy as np
from numpy import matlib
#from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
from .statshelper import SaliencyMapFilt,SaccadesTrial,calculate_angle
from .scanpathshelpdebug import CreatAoiRects,SaccadeLine
import platform
import warnings


def GetParams(self):
    """ Get stimulus and subject info of dataset """  
    if 'subjectID' not in self.data.columns:
        raise ValueError('subjectID column not found. Please set DataInfo(subjectID=Your Column)')

    if 'Stimulus' not in self.data.columns:
        raise ValueError('Stimulus column not found. Please set DataInfo(Stimulus=Your Column)')


    self.subjects=np.unique(self.data['subjectID'].to_numpy())
    self.stimuli=np.unique(self.data['Stimulus'].to_numpy())

    self.ns,self.np=len(self.subjects),len(self.stimuli)
    return  self.subjects,self.stimuli

def GetSize(self,infersize=False,Interval=99):
    '''
    get stimulus specific resolution.
    
    if infersize==False, and images are loaded, use the resolution of each image
    if infersize==True size inferred as central Interval percentile of fixations data, 
    if infersize==False and image is not found, use full screen resolution for each image

    Parameters
    ----------
    infersize : type bool, optional
        DESCRIPTION. The default is False.
    Interval: type percentile to use to infer image area from eye movement data, only relevant in infersize=True The default is 99.

    Returns
    -------
    BoundsX: x pixel bounds for each stimulus (num stimuli, for start and end X coordinate)
    BoundsY : y pixel bounds for each stimulus (num stimuli, for start and end Y coordinate)

    '''
    BoundsX=np.zeros((len(self.stimuli),2))
    BoundsY=np.zeros((len(self.stimuli),2))
    
    if hasattr(self,'images') and infersize==False:
        for cim,im in enumerate(self.images):
            BoundsX[cim,1]=np.shape(self.images[im])[1]
            BoundsY[cim,1]=np.shape(self.images[im])[0]     
    elif infersize==True:    
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
    else:
        BoundsX[:,1]=self.x_size
        BoundsY[:,1]=self.y_size

    BoundsX=np.intp(np.round(BoundsX))
    BoundsY=np.intp(np.round(BoundsY))
    return BoundsX,BoundsY

def GetStimuli(self,extension,path=0,infersubpath=False,sizecorrect=True):
    ''' load stimulus files from path'''
    #assert 'Stimulus' in self.data.columns, 'stimulus column not found'
    if len(self.stimuli) <= 0:
        raise ValueError('No stimuli loaded. Please provide: DataInfo(Stimulus=Your Column)')


    self.images={}
    if infersubpath==True:
        if 'category' in self.data:
            self.data.rename(columns={'category':'Category'},inplace=True)
        print('infer path from database categeory')
    for cs,s in enumerate(self.stimuli):
        print(s)
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
                
        self.images[s]=Stim

        Res=np.shape(Stim)
        
        ## implement correction for difference between screen and stimulus/image resolution!
        # and and y values are shifted, so that stimuli start at 0
        if sizecorrect:

            if Res[0] != self.y_size:   
                print("!y size incosistency warning, expected:",self.y_size,'vs actual:', Res[0])
                ys1=(self.y_size-np.shape(self.images[s])[0])/2        
    #            ys2=self.y_size-ys1
                if self.saccadedat:
                    self.data.loc[self.data.Stimulus==s, 'start_y']= self.data.start_y[self.data.Stimulus==s]-ys1
                    self.data.loc[self.data.Stimulus==s, 'end_y']= self.data.end_y[self.data.Stimulus==s]-ys1


                else:
                    self.data.loc[self.data.Stimulus==s, 'mean_y']= self.data.mean_y[self.data.Stimulus==s]-ys1
            else:
                print('stimulus size in y is full screen')
            if Res[1] != self.x_size:
                print("!x size incosistency warning, expected",self.x_size,'vs actual', Res[1])
                xs1=(self.x_size-np.shape(self.images[s])[1])/2
               # self.data['mean_x'][self.data.Stimulus==s] -= xs1
                if self.saccadedat:
                    self.data.loc[self.data.Stimulus==s, 'start_x']= self.data.start_x[self.data.Stimulus==s]-xs1
                    self.data.loc[self.data.Stimulus==s, 'end_x']= self.data.end_x[self.data.Stimulus==s]-xs1

                else:
                    self.data.loc[self.data.Stimulus==s, 'mean_x']= self.data.mean_x[self.data.Stimulus==s]-xs1

                print('correction applied, assuming central stimulus presentation')
            else:
                print('stimulus size in x full screen')

       
        print(' ')
        
    pass 
 

  
def FixCountCalc(self,Stim,CutAct=False):
    ''' Pixelwise fixation count for each participant, but for single stimulus  (Stim) 
    output: subjects*y*x --> num of fixaiton for each pixel
    if CutAct==1 in the end, only the within bounds areas is returned for further calculations
    optional parameter, substring for stimulus names containing the same substring'''
 
    
    if hasattr(self,'images')==False:
         idims=np.array([self.y_size,self.x_size])
    else:
         idims=np.shape(self.images[Stim])
    stimn=np.nonzero(self.stimuli==Stim)[0]
    yimsize,ximsize=idims[0],idims[1]
    print('resolution x =', ximsize, ' y =',yimsize)

    FixCountInd=np.zeros(((self.ns,yimsize,ximsize)))
    
    
    for cs,s in enumerate(self.subjects):
      
        x,y=np.intp(self.GetFixationData(s,Stim))
        if len(x)>0:
            Valid=np.nonzero((x<ximsize)&(x>=0)&(y>=0)&(y<yimsize))[0]
            X,Y=x[Valid],y[Valid]
            for xx,yy in zip(X,Y):
                FixCountInd[cs,yy,xx]+=1
        #if np.sum(FixCountInd[cs,:,:])==0:
            #print('no fixations for', s ,'with', Stim)
      
    if CutAct:
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

def GetSaccadeData(self,subj,stim,timemin=0,timemax=np.inf,timecol=0):
    """get X,Y start and sequence for a subject and stimulus saccades
    this function is working with Saccade data format, with start_x, start_y,end_x,end_y
    output 1: array of pixel x for sequence of saccade start y
    output 2: array of pixel y for sequence of saccade start y
    output 3: array of pixel x for sequence of saccade end x
    output 4: array of pixel y for sequence of saccade end y
    
    """
    SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==subj)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==stim),SubjIdx) # idx for subject and painting
    if type(timecol)!=int:
        TimeIdx=np.nonzero((self.data[timecol]>timemin)&(self.data[timecol]<timemax))[0]
        TrialSubIdx=np.intersect1d(TrialSubIdx, TimeIdx)
    StartTrialX=np.array(self.data['start_x'].iloc[TrialSubIdx]) # get x data for trial
    StartTrialY=np.array(self.data['start_y'].iloc[TrialSubIdx]) # get y data for trial
    EndTrialX=np.array(self.data['end_x'].iloc[TrialSubIdx]) # get y data for trial
    EndTrialY=np.array(self.data['end_y'].iloc[TrialSubIdx]) # get y data for trial
    return StartTrialX,StartTrialY,EndTrialX,EndTrialY


def GetDurations(self,s,p):
    """get fixations durations for a trials
    output: array of fixation durations """
    SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==s)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==p),SubjIdx) # idx for subject and painting
     # get x data for trial
    durations=np.array(self.data['duration'].iloc[TrialSubIdx]) # get y data for trial
    return durations

def GetGroups(self,betwcond,stim=False):
    ''' Between group comparison- 2 groups expected
    get conditions from between group column, check if mapping of participants to conditions is unique
    if participant is not in either of the groups NAN in the output'''
    self.Conds=np.unique(self.data[betwcond])
    print('Conditions',self.Conds)
   
  #  assert len(self.Conds)==2, 'you need 2 groups'
    WhichC=np.zeros(self.ns)
    WhichC[:]=np.nan
    WhichCN=np.zeros(self.ns,dtype='object')
    WhichCN[:]=np.nan

    for cs,s in enumerate(self.subjects):
        for cc,c in enumerate(self.Conds):
            if stim==False:
                PPc=np.unique(self.data[betwcond][self.data['subjectID']==s])
            else:
                dat=self.data[self.data['Stimulus']==stim]
                PPc=np.unique(dat[betwcond][dat['subjectID']==s])
            if len(PPc) > 1:
                raise ValueError('Participant condition mapping not unique')
                
            if PPc==self.Conds[cc]:
                WhichC[cs]=cc
                WhichCN[cs]=c
    self.whichC=WhichC
    return WhichC,np.array(WhichCN)

def GetCats(self,condColumn):
    ''' Within group comparison- 
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
    if len(np.unique(WhichCat)) != len(np.unique(self.WithinConds)):
        raise ValueError('Stimulus category mapping problem')

    return WhichCat

def GetStimSubjMap(self,Stims):
    StimIdxs=[[],[]]  # check which participant saw which stimulus
    for cs,s in enumerate(self.subjects):
        x1,y1=np.intp(self.GetFixationData(s,Stims[0]))
        x2,y2=np.intp(self.GetFixationData(s,Stims[1]))
        if len(x1)>0 and len(x2)>0:
            warnings.warn('non unique stimulus - subject mapping error')
        if len(x1)>0:
            StimIdxs[0].append(cs)
        elif len(x2)>0:
            StimIdxs[1].append(cs)
           
    StimIdxs[0]=np.intp(np.array(StimIdxs[0]))    
    StimIdxs[1]=np.intp(np.array(StimIdxs[1]))    
    return StimIdxs
    


def GetSaccades(self):
    ''' from fixations, make approximate saccades, and store it as saccade objects'''
    SaccadeObj=[]
    
    self.nsac=np.zeros((self.ns,self.np))
    self.saccadelenghts=np.zeros((self.ns,self.np),dtype=object)
    self.saccadeangles=np.zeros((self.ns,self.np),dtype=object) 

    for cs,s in enumerate(self.subjects):
            SaccadeObj.append([])        
            for cp,p in enumerate(self.stimuli):
                
                    
                SaccadeObj[cs].append([])
                if self.saccadedat==True: ##  if already in saccade format
                    StartTrialX,StartTrialY,EndTrialX,EndTrialY=self.GetSaccadeData(s,p)
                else: # if transformed from fixation format
                    FixTrialX,FixTrialY=self.GetFixationData(s,p)
                    StartTrialX,StartTrialY,EndTrialX,EndTrialY=SaccadesTrial(FixTrialX,FixTrialY)
                if len(StartTrialX)>0:
                    SaccadesSubj=np.column_stack((StartTrialX,StartTrialY,EndTrialX,EndTrialY)) 
                    csac=0
                    self.saccadeangles[cs,cp]=calculate_angle(StartTrialX,StartTrialY,EndTrialX,EndTrialY)
                    saccadelens_list=[]
                    for sac in range(len(StartTrialX)):
                        if np.isfinite(SaccadesSubj[sac,0])==True:
                            SaccadeObj[cs][cp].append(SaccadeLine(SaccadesSubj[sac,:]))  # store saccades as list of  objects 
                            saccadelens_list.append(SaccadeObj[cs][cp][-1].length())
                            csac+=1
                    self.nsac[cs,cp]=csac  # num of saccades for each participant and painting
                    self.saccadelenghts[cs,cp]=np.array([saccadelens_list])
                else:
                    self.saccadeangles[cs,cp]=np.NAN
                    self.nsac[cs,cp]=np.NAN  # num of saccades for each participant and painting
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


  
def Heatmap(self,Stim,SD=25,Ind=0,Vis=0,FixCounts=0,cutoff='median',CutArea=0,ax=False,substring=False,cmap='plasma',alpha=.5):
    ''' Pipeline for  heatmap calculation, FixCounts are calculated for stimulus, or passed pre-calcualted as optional parameter
    output: heatmap for a stimulus
    
    Vis:  if 1 Heatmap visual shows up- otherwise no visualization, but returns the heatmap values
    Ind: independent heatmap for each participant
    cutarea option: 1 only use active area (99% percentile of fixations), 0- use all of the area - set it to 1, if stimulus does not cover the whole screen
    cutoff=median: median cutoff, otherwise percetile of values to replace with nans, goal--> clear visualization
    substring: use part of file name (expected for mathcing paired files)
    cmap=colormap (see matplotlib colormaps for options: https://matplotlib.org/stable/users/explain/colors/colormaps.html)
    alpha= transparency- 0-1 higher values less transparent
    '''
  #  if hasattr(self,'fixcounts'):
   #     FixCountIndie=self.fixcounts['Stim']
    #else:    
    stimn=np.nonzero(self.stimuli==Stim)[0]
    if hasattr(self,'images')==False:
        idims=np.array([self.y_size,self.x_size])
    else:
        idims=np.shape(self.images[Stim])
        
    yimsize,ximsize=idims[0],idims[1]

    if type(FixCounts)==int:
        FixCounts=self.FixCountCalc(Stim,CutAct=CutArea) 
    if np.sum(FixCounts) <= 0:
        raise ValueError('No fixations found')


    if np.sum(FixCounts)<10:
        print('WARNING NUM FIX FOUND: ',np.sum(FixCounts))
    if Ind==0:
        smap=SaliencyMapFilt(FixCounts,SD=SD,Ind=0)
        if CutArea:
            smapall=np.zeros((yimsize,ximsize))
            smapall[:]=np.nan
            if len(stimn)>0:
                stimn=stimn[0]
            smapall[int(self.boundsY[stimn,0]):int(self.boundsY[stimn,1]),int(self.boundsX[stimn,0]):int(self.boundsX[stimn,1])]=smap
        else:
            smapall=np.copy(smap)
    else:
        smap=np.zeros_like(FixCounts)
        for cs,s in enumerate(self.subjects):
            smap[cs,:,:]=SaliencyMapFilt(FixCounts[cs,:,:],SD=SD,Ind=1)
        smapall=np.nanmean(smap,0)
       
    
    if Vis:
        self.VisHeatmap(Stim,smapall,ax=ax,cutoff=cutoff,cmap=cmap,alpha=alpha)
    return smapall

