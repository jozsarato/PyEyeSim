import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 
from .hmmhelper import DiffCompsHMM,FitScoreHMMGauss
from .visualhelper import draw_ellipse
import hmmlearn.hmm  as hmm
 # hmm related functions start here
def DataArrayHmm(self,stim,group=-1,tolerance=20,verb=True):
    ''' HMM data arrangement, for the format required by hmmlearn
    tolarance control the numbers of pixels, where out of stimulus fixations are still accepted, currently disabled as not yet adapted for changing bounds
    therefore, participants with invalid fixations are not yet removed
    
    verb-- verbose-- print missing participants, too much printing for leave one out cross validation'''
    
    XX=np.array([])
    YY=np.array([])
    Lengths=np.array([],dtype=int)
    self.suseHMM=np.array([],dtype=int)
    #print('org data for stim')
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
                if verb:
                    print('invalid fixation location for subj', s)
           # else:
            if len(fixX)>2:
                XX=np.append(XX,fixX)
                YY=np.append(YY,fixY)
                Lengths=np.append(Lengths,len(fixX))
                self.suseHMM=np.append(self.suseHMM,s)
            elif verb:
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
    lenTest=np.array([],dtype=int)
    DatTest=np.zeros((0,2))
    DatTr=np.zeros((0,2)) 
    for ci in range(len(Lengths)):
        if ci==0:
            start=0
        else:
            start=Idxs[ci-1]
        if ci in totest:
            DatTest=np.vstack((DatTest,Dat[start:Idxs[ci],:]))
            lenTest=np.append(lenTest,Lengths[ci])
        else:
            DatTr=np.vstack((DatTr,Dat[start:Idxs[ci],:]))
            lenTrain=np.append(lenTrain,Lengths[ci])
    if vis:
        self.MyTrainTestVis(DatTr,DatTest,lenTrain,lenTest,totest)
    return DatTr,DatTest,lenTrain,lenTest   



def FitLOOHMM(self,ncomp,stim,covar='full',verb=False):
    ''' fit HMM, N subject times, leaving out once each time
    ncomp: number of components
    stim: stimulus code 
    covar: covariance type 'full' or  'tied' '''
    NTest=1
    xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80,verb=verb)
    Dat=np.column_stack((xx,yy))
    ScoresLOO=np.zeros(len(self.suseHMM))
    print('num valid observers',len(ScoresLOO))
    for cs,s in enumerate(self.suseHMM):
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=0,totest=cs)
        HMMfitted,sctr,scte=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)
        ScoresLOO[cs]=scte
    return Dat,lengths,ScoresLOO
def FitVisHMM(self,stim,ncomp=3,covar='full',ax=0,ax2=0,NTest=5,showim=True,verb=False,incol=False,vis=True):
    ''' fit and visualize HMM -- beta version
    different random train - test split for each iteration-- noisy results'''
    xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80,verb=verb)
    Dat=np.column_stack((xx,yy))
    
    DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=1)


    HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)


    if vis:
        if type(ax)==int:
            fig,ax=plt.subplots()
        if type(ax2)==int:
            fig,ax2=plt.subplots()
        self.VisHMM(DatTr,HMMfitted,ax=ax,showim=showim,stim=stim,lengths=lenTrain,incol=incol)
        ax.set_title('n: '+str(ncomp)+' train ll: '+str(np.round(meanscore,2))+' test ll: '+str(np.round(meanscoreTe,2)),fontsize=9)
        ax2.scatter(ncomp,meanscore,color='g',label='training')
        ax2.scatter(ncomp,meanscoreTe,color='r',label='test')
        ax2.set_xlabel('num components')
        ax2.set_ylabel('log likelihood')
        ax2.legend()

  
    return HMMfitted,meanscore,meanscoreTe
    
def FitVisHMMGroups(self,stim,betwcond,ncomp=3,covar='full',ax=0,ax2=0,NTest=3,showim=False,Rep=1,groupnames=0):
    ''' fit and visualize HMM -- beta version
    different random train - test split for each iteration-- noisy results'''
    self.GetGroups(betwcond)
    Grs=np.unique(self.data[betwcond])
    
    fig,ax=plt.subplots(ncols=len(Grs),figsize=(12,5))
    fig2,ax2=plt.subplots(ncols=2) 

    # data arrangement for groups
    ScoresTrain=np.zeros((Rep,len(Grs),len(Grs)))
    ScoresTest=np.zeros((Rep,len(Grs),len(Grs)))
   
   
    for rep in range(Rep):  
        XXTrain=[]
        LengthsTrain=[]
        XXTest=[]
        LengthsTest=[]
        for cgr,gr in enumerate(Grs):
            xx,yy,Lengths=self.DataArrayHmm(stim,group=cgr,tolerance=50,verb=False)
            if np.sum(np.shape(xx))==0:
                print('data not found')
            Dat=np.column_stack((xx,yy))
            
            DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,Lengths,ntest=NTest,vis=0,rand=1)
            XXTrain.append(DatTr)
            XXTest.append(DatTest)
            LengthsTrain.append(lenTrain)
            LengthsTest.append(lenTest)
        for cgr,gr in enumerate(Grs):
            HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,XXTrain[cgr],XXTest[cgr],LengthsTrain[cgr],LengthsTest[cgr],covar=covar)
            if rep==0:
                self.VisHMM(XXTrain[cgr],HMMfitted,ax=ax[cgr],showim=showim,stim=stim,lengths=LengthsTrain[cgr])
                if type(groupnames)==int:
                    ax[cgr].set_title(cgr)

                else:
                    ax[cgr].set_title(groupnames[cgr])
            for cgr2,gr2 in enumerate(Grs):
                ScoresTrain[rep,cgr2,cgr]=HMMfitted.score(XXTrain[cgr2],LengthsTrain[cgr2])/np.sum(LengthsTrain[cgr2])
                ScoresTest[rep,cgr2,cgr]=HMMfitted.score(XXTest[cgr2],LengthsTest[cgr2])/np.sum(LengthsTest[cgr2])

    im=ax2[0].pcolor(np.mean(ScoresTrain,0))
    ax2[0].scatter(np.arange(len(Grs))+.5,np.argmax(np.mean(ScoresTrain,0),0)+.5,color='k')  # mark most likely for each group
    ax2[0].set_title('training')
#       plt.colorbar(im1)
    im=ax2[1].pcolor(np.mean(ScoresTest,0))
    ax2[1].scatter(np.arange(len(Grs))+.5,np.argmax(np.mean(ScoresTest,0),0)+.5,color='k')  # mark most likely for each group

#        plt.colorbar(im2)
    ax2[1].set_title('test')
    ax2[0].set_ylabel('tested')

    for pl in range(2):
        ax2[pl].set_xlabel('fitted')
        ax2[pl].set_xticks(np.arange(len(Grs))+.5)
        if type(groupnames)==int:
            ax2[pl].set_xticklabels(Grs)
            ax2[pl].set_yticklabels(Grs,rotation=90)
        else:
            ax2[pl].set_xticklabels(groupnames)
            ax2[pl].set_yticklabels(groupnames,rotation=90)
        ax2[pl].set_yticks(np.arange(len(Grs))+.5)
    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
    fig2.colorbar(im, cax=cbar_ax)
#    plt.tight_layout()
    plt.show()


    return ScoresTrain, ScoresTest


def HMMSimPipeline(self,ncomps=[4,6],verb=False):
    ''' fit l hidden markov model to data, with different number of components, each participants likelihood with leave-one-out cross validation
    can have a long run time with longer viewing time/lot of data 
    return the individual loo log likelihoods from the best model (highest log likelihood) for each stimulus 
    verb=True: print line for subjects with not enough fixations. - too much printing for many subjects wiht low number of fixations '''
    StimSimsHMM=np.zeros((len(ncomps),self.np))
    
    print(np.shape(StimSimsHMM))
    StimSimsHMMall=np.zeros((len(ncomps),self.ns,self.np))
    StimSimsHMMall[:]=np.NAN
    for cncomp, ncomp in enumerate(ncomps):
        print(f'fitting HMM with {ncomp} components')
        for cp in range(self.np):
            print(f'for stimulus {self.stimuli[cp]}')
            Dat,lengths,ScoresLOO=self.FitLOOHMM(ncomp,self.stimuli[cp],covar='tied',verb=verb)
            missS=np.setdiff1d(self.subjects,self.suseHMM)
            if len(missS)>0:
                idxs=np.array([],dtype=int)
                for cs,s in enumerate(self.subjects):
                    if s not in missS:
                        idxs=np.append(idxs,cs)            
                StimSimsHMMall[cncomp,idxs,cp]=ScoresLOO
            else:
                StimSimsHMMall[cncomp,:,cp]=ScoresLOO
            StimSimsHMM[cncomp,cp]=np.mean(ScoresLOO)
    return StimSimsHMM, StimSimsHMMall


