
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
from .statshelper import SaliencyMapFilt,SaccadesTrial,ScanpathL,StatEntropy
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim ,CheckCorr
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
                Stim=plt.imread(path+s+extension)
            else:   
               # print(path+str(int(s))+extension)
                Stim=plt.imread(path+str(int(s))+extension)

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
 
def GetFixationData(self,subj,stim):
    """get X,Y fixation sequence for a subject and stimulus
    output 1: array of pixel x for sequence of fixations
    output 2: array of pixel y for sequence of fixations"""
    SubjIdx=np.nonzero(self.data['subjectID'].to_numpy()==subj)  #idx for subject
    TrialSubIdx=np.intersect1d(np.nonzero(self.data['Stimulus'].to_numpy()==stim),SubjIdx) # idx for subject and painting
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
    
def SaccadeSel(self,SaccadeObj,nDiv): 
    ''' select saccades for angle comparison method'''
    nH,nV=nDiv,nDiv
    SaccadeAOIAngles=[]
    SaccadeAOIAnglesCross=[]
    
    AOIRects=CreatAoiRects(nH,nV,self.boundsX,self.boundsY)
    Saccades=np.zeros((((self.ns,self.np,nH,nV))),dtype=np.ndarray)  # store an array of saccades that cross the cell, for each AOI rectangle of each trial for each partiicpant
    for s in np.arange(self.ns):
        SaccadeAOIAngles.append([])
        SaccadeAOIAnglesCross.append([])
        for p in range(self.np):
            SaccadeAOIAngles[s].append(np.zeros(((int(self.nsac[s,p]),nH,nV))))
           # print(s,p,NSac[s,p])
            SaccadeAOIAngles[s][p][:]=np.NAN
            SaccadeAOIAnglesCross[s].append(np.zeros(((int(self.nsac[s,p]),nH,nV))))
            SaccadeAOIAnglesCross[s][p][:]=np.NAN
            for sac in range(len(SaccadeObj[s][p])):
                SaccadeDots=SaccadeObj[s][p][sac].LinePoints()
                
                
                for h in range(nH):
                    for v in range(nV):
                       # print(h,v)
                        if AOIRects[p][h][v].Cross(SaccadeDots)==True:
                          #  print(h,v,SaccadeObj[s][p][sac].Angle())
                            SaccadeAOIAngles[s][p][sac,h,v]=SaccadeObj[s][p][sac].Angle()  # get the angle of the sacccade

                if np.sum(SaccadeAOIAngles[s][p][sac,:,:]>0)>1:  # select saccaded that use multiple cells
                    #print('CrossSel',SaccadeAOIAngles[s][p][sac,:,:])
                    SaccadeAOIAnglesCross[s][p][sac,:,:]=SaccadeAOIAngles[s][p][sac,:,:]

            for h in range(nH):
                for v in range(nV):
                    if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][p][:,h,v]))>0:
                        Saccades[s,p,h,v]=np.array(SaccadeAOIAnglesCross[s][p][~np.isnan(SaccadeAOIAnglesCross[s][p][:,h,v]),h,v])
                    else:
                        Saccades[s,p,h,v]=np.array([])
    return Saccades


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


  
def Heatmap(self,Stim,SD=25,Ind=0,Vis=0,FixCounts=0,cutoff='median',CutArea=0,ax=False,alpha=.5,center=0):
    ''' Pipeline for  heatmap calculation, FixCounts are calculated for stimulus, or passed pre-calcualted as optional parameter
    output: heatmap for a stimulus
    cutarea option: 1 only use active area (99% percentile of fixations), 0- use all of the area 
    cutoff=median: median cutoff, otherwise percetile of values to replace with nans, goal--> clear visualization
    center, if pixel coordinates dont match, painting presented centrally, but gaze coors are zero based'''
  #  if hasattr(self,'fixcounts'):
   #     FixCountIndie=self.fixcounts['Stim']
    #else:    
    stimn=np.nonzero(self.stimuli==Stim)[0]
    if hasattr(self,'boundsX')==False:
        print('run RunDescriptiveFix first- without visuals')
        self.RunDescriptiveFix()
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
        xs1=(self.x_size-np.shape(self.images[Stim])[1])/2
        xs2=self.x_size-xs1
        ys1=(self.y_size-np.shape(self.images[Stim])[0])/2
        ys2=self.y_size-ys1
        if ax==False:
            fig,ax=plt.subplots()
        if center:
            ax.imshow(self.images[Stim],extent=[xs1,xs2,ys2,ys1])
        else:
            ax.imshow(self.images[Stim])
        ax.imshow(smapall,alpha=.5) 
        ax.set_xticks([])
        ax.set_yticks([])

            
    return smapall

