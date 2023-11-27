
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 

import hmmlearn.hmm  as hmm


def DiffCompsHMM(datobj,stim=0,ncomps=np.arange(2,6),NRep=10,NTest=3,covar='full'):
    ''' fit and cross validate HMM for a number of different hidden state numbers, as defined by ncomps'''
    if len(ncomps)<7:
        fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(13,6))
    elif len(ncomps)<12:
        fig,ax=plt.subplots(ncols=4,nrows=3,figsize=(14,8))
  
    fig,ax2=plt.subplots()
    
    scoretrain,scoretest=np.zeros((NRep,len(ncomps))),np.zeros((NRep,len(ncomps)))
    for cc,nc in enumerate(ax.flat):
        if cc<len(ncomps):
            print('num comps: ',ncomps[cc],' num:', cc+1,'/', len(ncomps))
            for rep in range(NRep):
                if rep==NRep-1:
                    vis=True
                else:
                    vis=False
                hmm,scoretrain[rep,cc],scoretest[rep,cc]=datobj.FitVisHMM(datobj.stimuli[stim],ncomps[cc],covar=covar,ax=nc,ax2=ax2,vis=vis,NTest=NTest,verb=False)

    plt.legend(['train','test'])
    plt.tight_layout()
    
    fig,ax=plt.subplots()
    ax.errorbar(ncomps,np.mean(scoretrain,0),stats.sem(scoretrain,0),color='g',label='train',marker='o')
    ax.errorbar(ncomps,np.mean(scoretest,0),stats.sem(scoretest,0),color='r',label='test',marker='o')
    ax.set_xlabel('num of components')
    ax.set_ylabel('log(likelihood)')
    ax.legend()
    return 


    

def FitScoreHMMGauss(ncomp,xx,xxt,lenx,lenxxt,covar='full'):
    HMM=hmm.GaussianHMM(n_components=ncomp, covariance_type=covar)
    HMM.fit(xx,lenx)
    sctr=HMM.score(xx,lenx)/np.sum(lenx)
    scte=HMM.score(xxt,lenxxt)/np.sum(lenxxt)
    return HMM,sctr,scte

