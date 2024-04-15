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

from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim, CheckCoor







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
    elif InferS==1:
        AOIboundsH=AOIbounds(self.boundsX[p,0], self.boundsX[p,1],nDivH)       
        AOIboundsV=AOIbounds(self.boundsY[p,0], self.boundsY[p,1],nDivV)   
    elif InferS==2:
        ims=np.shape(self.images[self.stimuli[p]])
        
        AOIboundsH=AOIbounds(0, ims[1],nDivH)       
        AOIboundsV=AOIbounds(0,ims[0],nDivV)  

    # set parameters & arrays to store data
    StatPtrial=np.zeros(nAOI) # static probabilities.
    StatNtrial=np.zeros(nAOI) # static counts.

   
    WhichAOIH=np.zeros(NFix)
    WhichAOIV=np.zeros(NFix)
    for x in range(NFix):
        WhichAOIH[x]=CheckCoor(AOIboundsH,FixTrialX[x]) # store which horizontal AOI each fixation is
        WhichAOIV[x]=CheckCoor(AOIboundsV,FixTrialY[x]) # store which vertical AOI each fixation is

    WhichAOI=np.zeros(NFix)
    WhichAOI[:]=np.NAN
    for x in range(NFix):
        if WhichAOIV[x]>-1 and WhichAOIH[x]>-1:   # only use valid idx
            WhichAOI[x]=AOInums[np.intp(WhichAOIV[x]),np.intp(WhichAOIH[x])]  # get combined vertival and horizontal
    for st in range(nAOI): # gaze transition start
        StatNtrial[st]=np.sum(WhichAOI==st)  # get count in AOI
        StatPtrial[st]=np.sum(WhichAOI==st)/np.sum(np.isfinite(WhichAOI)) # calculate stationary P for each AOI    
    return NFix,StatPtrial,StatNtrial

    
def SaccadeSel(self,SaccadeObj,nHor,nVer=0,InferS=False): 
    ''' select saccades for angle comparison method'''
    if nVer==0:
        nVer=nHor  # if number of vertical divisions not provided -- use same as the number of horizontal
    SaccadeAOIAngles=[]
    SaccadeAOIAnglesCross=[]
    if InferS:
        if hasattr(self,'boundsX')==False:
            print('runnnig descriptives to get bounds')
            self.RunDescriptiveFix()  
        AOIRects=CreatAoiRects(nHor,nVer,self.boundsX,self.boundsY)
    else:
        AOIRects=CreatAoiRects(nHor,nVer,self.x_size,self.y_size)

    Saccades=np.zeros((((self.ns,self.np,nVer,nHor))),dtype=np.ndarray)  # store an array of saccades that cross the cell, for each AOI rectangle of each trial for each partiicpant
    for s in np.arange(self.ns):
        SaccadeAOIAngles.append([])
        SaccadeAOIAnglesCross.append([])
        for p in range(self.np):
            SaccadeAOIAngles[s].append(np.zeros(((int(self.nsac[s,p]),nVer,nHor))))
           # print(s,p,NSac[s,p])
            SaccadeAOIAngles[s][p][:]=np.NAN
            SaccadeAOIAnglesCross[s].append(np.zeros(((int(self.nsac[s,p]),nVer,nHor))))
            SaccadeAOIAnglesCross[s][p][:]=np.NAN
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


def SacSim1Group(self,Saccades,Thr=5):
    ''' calculate saccade similarity for each stimulus, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input
    Thr=5: threshold for similarity'''
    
    nVer=np.shape(Saccades)[2]
    nHor=np.shape(Saccades)[3]
        
    SimSacP=np.zeros((self.ns,self.ns,self.np,nVer,nHor))  
    SimSacP[:]=np.NaN
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            if s1!=s2:
                for p1 in range(self.np):
                    if self.nsac[s1,p1]>5 and self.nsac[s2,p1]>5:                    
                        for h in range(nHor):
                            for v in range(nVer):
                                if len(Saccades[s1,p1,v,h])>0 and len(Saccades[s2,p1,v,h])>0:
                                        
                                    simsacn=CalcSim(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],Thr=Thr)
                                    SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p1,v,h]))
    return SimSacP

  


def SacSimPipeline(self,divs=[4,5,7,9],Thr=5,InferS=True):
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np))

    for cd,ndiv in enumerate(divs):
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv,InferS=InferS)
        SimSacP=self.SacSim1Group(sacDivSel,ndiv,Thr=Thr)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0)
        StimSims[cd,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0),0)
    return StimSims,np.nanmean(StimSimsInd,0)
   
