# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:54:38 2024

@author: aratoj87
"""

import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import time
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim, CheckCoor,CalcSimAlt,angle_difference_power



def AOIFix(self,p,FixTrialX,FixTrialY,nDivH,nDivV):
    '''
    given a sequence of X,Y fixation data and AOI divisions, calculate static N and p matrix)

    Positional arguments
   ----------
    p : stimulus index
    FixTrialX : np array of fixation x positions
    FixTrialY : np array of fixation y positions
    nDivH : int, num of horizontal divisions
    nDivV : int, num of vertical divisions
     
    
    Returns
    -------
    NFix : number of fixations.
    StatPtrial : statitic fixation probability distribution across divisions
    StatNtrial : static fixation counts

    '''
    nAOI=nDivH*nDivV
    AOInums=np.arange(nAOI).reshape(nDivV,nDivH)
    NFix=len(FixTrialX)  # num of data points
    # set AOI bounds
   # print(p,SizeGendX)

    AOIboundsH=AOIbounds(self.boundsX[p,0], self.boundsX[p,1],nDivH)       
    AOIboundsV=AOIbounds(self.boundsY[p,0], self.boundsY[p,1],nDivV)  
     
   
    # set parameters & arrays to store data
    StatPtrial=np.zeros(nAOI) # static probabilities.
    StatNtrial=np.zeros(nAOI) # static counts.

   
    WhichAOIH=np.zeros(NFix)
    WhichAOIV=np.zeros(NFix)
    for x in range(NFix):
        WhichAOIH[x]=CheckCoor(AOIboundsH,FixTrialX[x]) # store which horizontal AOI each fixation is
        WhichAOIV[x]=CheckCoor(AOIboundsV,FixTrialY[x]) # store which vertical AOI each fixation is

    WhichAOI=np.zeros(NFix)
    WhichAOI[:]=np.nan
    for x in range(NFix):
        if WhichAOIV[x]>-1 and WhichAOIH[x]>-1:   # only use valid idx
            WhichAOI[x]=AOInums[np.intp(WhichAOIV[x]),np.intp(WhichAOIH[x])]  # get combined vertival and horizontal
    for st in range(nAOI): # gaze transition start
        StatNtrial[st]=np.sum(WhichAOI==st)  # get count in AOI
        StatPtrial[st]=np.sum(WhichAOI==st)/np.sum(np.isfinite(WhichAOI)) # calculate stationary P for each AOI    
    return NFix,StatPtrial,StatNtrial

    
def SaccadeSel(self,SaccadeObj,nHor,nVer=0): 
    '''
    select saccades for angle comparison method, return array of saccade angles for each participant, stimulus and cell for a given hor by ver division
    

    Positional arguments
   ----------
    SaccadeObj : saccade object as input (as defined in scanpathsimhelper.py)
    nHor : num horizontal divisons
     
     Optional arguments
     ----------
    nVer : num vertical divisions (if zero, equals to horizontal)

    Returns
    -------
    Saccades :  num observers * num stimuli * num vertical * num horizontal matrix, each entry is an array of saccade angles for that cell

    '''
    if nVer==0:
        nVer=nHor  # if number of vertical divisions not provided -- use same as the number of horizontal
    SaccadeAOIAngles=[]
    SaccadeAOIAnglesCross=[]
    
    AOIRects=CreatAoiRects(nHor,nVer,self.boundsX,self.boundsY)
    

    Saccades=np.zeros((((self.ns,self.np,nVer,nHor))),dtype=np.ndarray)  # store an array of saccades that cross the cell, for each AOI rectangle of each trial for each partiicpant
    for s in np.arange(self.ns):
        SaccadeAOIAngles.append([])
        SaccadeAOIAnglesCross.append([])
        for p in range(self.np):
            SaccadeAOIAngles[s].append(np.zeros(((int(self.nsac[s,p]),nVer,nHor))))
           # print(s,p,NSac[s,p])
            SaccadeAOIAngles[s][p][:]=np.nan
            SaccadeAOIAnglesCross[s].append(np.zeros(((int(self.nsac[s,p]),nVer,nHor))))
            SaccadeAOIAnglesCross[s][p][:]=np.nan
            for sac in range(len(SaccadeObj[s][p])):
                SaccadeDots=SaccadeObj[s][p][sac].LinePoints()
                
                
                for h in range(nHor):
                    for v in range(nVer):
                       # print(h,v)
                        if AOIRects[p][h][v].Cross(SaccadeDots)==True:
                          #  print(h,v,SaccadeObj[s][p][sac].Angle())
                            SaccadeAOIAngles[s][p][sac,v,h]=SaccadeObj[s][p][sac].Angle()  # get the angle of the sacccade

                if np.sum(SaccadeAOIAngles[s][p][sac,:,:]>0)>1:  # select saccaded that use multiple cells
                    #print('CrossSel',SaccadeAOIAngles[s][p][sac,:,:])
                    SaccadeAOIAnglesCross[s][p][sac,:,:]=SaccadeAOIAngles[s][p][sac,:,:]

            for h in range(nHor):
                for v in range(nVer):
                    if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][p][:,v,h]))>0:
                        Saccades[s,p,v,h]=np.array(SaccadeAOIAnglesCross[s][p][~np.isnan(SaccadeAOIAnglesCross[s][p][:,v,h]),v,h])
                    else:
                        Saccades[s,p,v,h]=np.array([])
    return Saccades


def SacSim1Group(self,Saccades,Thr=5,p='all',normalize='add',power=1,bothnot=False):
    ''' calculate saccade similarity for each stimulus, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input saccade matrix dimensions
    Thr=5: threshold for similarity in degree
    !! if Thr is 0, use power function for difference in angle, for now this is a difference score, not a similarity
    normalize, if provided must be add or mult 
    simcalc: True all angles transformed to below 180 before calculating similarity
    bothnot: if True cells where neither particpants have fixations, are calculated as similar : 1 
    cells where only 1 person has saccades, calculated as zero'''
    
    nVer=np.shape(Saccades)[2]
    nHor=np.shape(Saccades)[3]
        
    SimSacP=np.zeros((self.ns,self.ns,self.np,nVer,nHor))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            if s1!=s2:
                for p1 in range(self.np):
                    if self.nsac[s1,p1]>5 and self.nsac[s2,p1]>5:                    
                        for h in range(nHor):
                            for v in range(nVer):
                                if len(Saccades[s1,p1,v,h])>0 and len(Saccades[s2,p1,v,h])>0:
                                    
                                    if Thr==0:
                                        SimSacP[s1,s2,p1,v,h]=angle_difference_power(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],power=power)
                                        
                                    else:          
                                       
                                        simsacn=CalcSimAlt(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],Thr=Thr)                       
                                        if normalize=='add':
                                            SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p1,v,h]))
                                        elif normalize=='mult':
                                            SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])*len(Saccades[s2,p1,v,h]))
                                elif len(Saccades[s1,p1,v,h])==0 and len(Saccades[s2,p1,v,h])>0:
                                    if bothnot:
                                        SimSacP[s1,s2,p1,v,h]=0
                                elif len(Saccades[s1,p1,v,h])>0 and len(Saccades[s2,p1,v,h])==0:
                                    if bothnot:
                                        SimSacP[s1,s2,p1,v,h]=0
                                elif len(Saccades[s1,p1,v,h])==0 and len(Saccades[s2,p1,v,h])==0:
                                    if bothnot:
                                        SimSacP[s1,s2,p1,v,h]=1
    

    return SimSacP

  
def SacSim1GroupAll2All(self,Saccades,Thr=5,p='all',normalize='add',power=1):
    ''' calculate saccade similarity for each stimulus, and across all stimuli, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input
    Thr=5: threshold for similarity    
    !! if Thr is 0, use power function for difference in angle, for now this is a difference score, not a similarity

    normalize, if provided must be add or mult '''
    
    nVer=np.shape(Saccades)[2]
    nHor=np.shape(Saccades)[3]
        
    SimSacP=np.zeros((self.ns,self.ns,self.np,self.np,nVer,nHor))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            if s1!=s2:
                for p1 in range(self.np):
                    for p2 in range(self.np):
                        if self.nsac[s1,p1]>5 and self.nsac[s2,p2]>5:                    
                            for h in range(nHor):
                                for v in range(nVer):
                                    if len(Saccades[s1,p1,v,h])>0 and len(Saccades[s2,p2,v,h])>0:
                                        
                                        if Thr==0:
                                            SimSacP[s1,s2,p1,p2,v,h]=angle_difference_power(Saccades[s1,p1,v,h],Saccades[s2,p2,v,h],power=power)

                                        else:
                                            simsacn=CalcSimAlt(Saccades[s1,p1,v,h],Saccades[s2,p2,v,h],Thr=Thr)
                                            if normalize=='add':
                                                SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p2,v,h]))
                                            elif normalize=='mult':
                                                SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades[s1,p1,v,h])*len(Saccades[s2,p2,v,h]))
     
    return SimSacP

def SacSim2GroupAll2All(self,Saccades1,Saccades2,Thr=5,p='all',normalize='add',power=1):
    ''' calculate saccade similarity for each stimulus, from two different observations,  across all stimuli, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects divided into grid AOIs as input,
    vertical and horizontal dimensions are inferred from the input, have to match between the two datasets
    Thr=5: threshold for similarity    
    !! if Thr is 0, use power function for difference in angle, for now this is a difference score, not a similarity

    normalize, if provided must be add or mult '''
    
    nVer1=np.shape(Saccades1)[2]
    nHor1=np.shape(Saccades1)[3]
    nVer2=np.shape(Saccades2)[2]
    nHor2=np.shape(Saccades2)[3]
    assert nVer1==nVer2,'vertical grid division mismatch'
    assert nHor1==nHor2,'horizontal grid division mismatch'

    SimSacP=np.zeros((self.ns,self.ns,self.np,self.np,nVer1,nHor1))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            for p1 in range(self.np):
                for p2 in range(self.np):
                    for h in range(nHor1):
                        for v in range(nVer1):
                            if len(Saccades1[s1,p1,v,h])>0 and len(Saccades2[s2,p2,v,h])>0:                                 
                                if Thr==0:
                                    SimSacP[s1,s2,p1,p2,v,h]=angle_difference_power(Saccades1[s1,p1,v,h],Saccades2[s2,p2,v,h],power=power)

                                else:
                                    simsacn=CalcSimAlt(Saccades1[s1,p1,v,h],Saccades2[s2,p2,v,h],Thr=Thr)
                                    if normalize=='add':
                                        SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades1[s1,p1,v,h])+len(Saccades2[s2,p2,v,h]))
                                    elif normalize=='mult':
                                        SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades1[s1,p1,v,h])*len(Saccades2[s2,p2,v,h]))
 
    return SimSacP


def SacSimPipeline(self,divs=[4,5,7,9],Thr=5,normalize='add',power=1,bothnot=False):
    ''' if Thr>0, threshold based similarity ratio,
    if Thr=0, average saccadic angle difference 
    if Thr=0 and power>1, average saccadic angle difference on the value defined by power
    this pipeline compares observers within each stimulus
    '''
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np))
    SimsAll=[]
    for cd,ndiv in enumerate(divs):
        start_time = time.time()
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv)
        SimSacP=self.SacSim1Group(sacDivSel,Thr=Thr,normalize=normalize,power=power,bothnot=bothnot)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0)
        StimSims[cd,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0),0)
        SimsAll.append(SimSacP)
        end_time=time.time()
        print(f"calculating similarity with div {ndiv}*{ndiv} took {end_time - start_time:.3f} sec")

    return StimSims,np.nanmean(StimSimsInd,0),SimsAll

def SacSimPipelineAll2All(self,divs=[4,5,7,9],Thr=5,normalize='add',power=1):
    ''' if Thr>0, threshold based similarity ratio,
    if Thr=0, average saccadic angle difference 
    if Thr=0 and power>1, average saccadic angle difference on the value defined by power
    the all to all pipeline compares observers both within and also between stimuli, therefore has a longer runtime
    
    '''
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np,self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np,self.np))
    SimsAll=[]
    for cd,ndiv in enumerate(divs):
        start_time = time.time()
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv)
        SimSacP=self.SacSim1GroupAll2All(sacDivSel,Thr=Thr,normalize=normalize,power=power)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,5),4),0)
        StimSims[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,5),4),0),0)
        SimsAll.append(SimSacP)
        end_time=time.time()
        print(f"calculating all to all similarity with div {ndiv}*{ndiv} took {end_time - start_time:.3f} sec")
    return StimSims,np.nanmean(StimSimsInd,0),SimsAll




def ScanpathSim2Groups(self,stim,betwcond,nHor=5,nVer=0,Thr=5,normalize='add'):
    if hasattr(self,'subjects')==0:
        self.GetParams()  
    SaccadeObj=self.GetSaccades()
    if type(stim)==str:
        if stim=='all':
            stimn=np.arange(self.ns)  # all stimuli
        else:
            stimn=np.nonzero(self.stimuli==stim)[0]

    else:    
        stimn=np.nonzero(self.stimuli==stim)[0] 
    if nVer==0:
        nVer=nHor  #
    
    SaccadeDiv=self.SaccadeSel(SaccadeObj,nHor=nHor,nVer=nVer)    
    SimSacP=self.SacSim1Group(SaccadeDiv,Thr=Thr,normalize=normalize)
    WhichC,WhichCN=self.GetGroups(betwcond)
    Idxs=[]
   
    #Cols=['darkred','cornflowerblue']
    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,8))
                        
    for cc,cond in enumerate(self.Conds):
        Idxs.append(np.nonzero(WhichCN==cond)[0])
    SimVals=np.zeros((2,2))
    SimValsSD=np.zeros((2,2))

    for cgr1,gr1 in enumerate(self.Conds):
        for cgr2,gr2 in enumerate(self.Conds):
            Vals=np.nanmean(np.nanmean(SimSacP[Idxs[cgr1],:,stimn,:,:][:,Idxs[cgr2],:,:],0),0)  
            SimVals[cgr1,cgr2]=np.nanmean(Vals)
            SimValsSD[cgr1,cgr2]=np.nanstd(Vals)
            self.VisGrid(Vals,stim,ax=ax[cgr1,cgr2],cbar=True,alpha=.8)
            ax[cgr1,cgr2].set_title(str(gr1)+' '+str(gr2)+' mean= '+str(np.round(SimVals[cgr1,cgr2],3)))
    
    return SimVals,SimValsSD