
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


