
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd


def GetParams(self):
    """ Get stimulus and subject info of dataset """  
    self.subjects=np.unique(self.data['subjectID'].to_numpy())
    self.stimuli=np.unique(self.data['Stimulus'].to_numpy())

    self.ns,self.np=len(self.subjects),len(self.stimuli)
    return  self.subjects,self.stimuli


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
            if platform.platform().find('mac')>-1:
                p=str(cat)+'/'
            else:
                p=str(cat)+'\\'
            print(cs,s,p)
            Stim=plt.imread(p+str(int(s))+extension)

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



