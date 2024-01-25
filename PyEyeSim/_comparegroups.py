  
# this file contains the EyeData methods for bewteen or within group comparisons.
# not all functions can do both ---- yet  

import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy

# import  library helper functions. 
from .statshelper import SaliencyMapFilt,SaccadesTrial,ScanpathL,StatEntropy
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim ,CheckCoor
from .visualhelper import VisBinnedProg,PlotDurProg,JointBinnedPlot,MeanPlot,draw_ellipse,HistPlot




def CompareGroupsFix(self,betwcond):
    '''
    Description: Run a set of between-group fixation comparisons, generate plots, and print descriptive statistics.
    should work for 2-4 groups
    calculates:
    - number of fixations
    - entropy of fixations (potentially long run time)
    - total scanpath length
    - saccade amplitude

    prints descriptive stats of the above
    print pairwise comparison of groups with the above measures
    
    Arguments: 
    betwcond (str): Name of the conditions for between-group fixation comparisons.
    '''
    
    WhichC,WhichCN=self.GetGroups(betwcond)
    if hasattr(self,'entropies')==False:   # check if entropy has already been calculated
        print('Calculating entropy')
        Entropies,self.entropmax,self.entropies_ind=self.GetEntropies()
    Cols=['darkred','cornflowerblue','orange','salmon']
    #plt.figure(figsize=(8,8))
    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,8))
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
     
        print(cc,c,'num participants: ',len(Idx))

        print(cc,c,'Num fix= ',np.round(np.mean(np.nanmean(FixGr,1)),2),'+/-',np.round(np.std(np.nanmean(FixGr,1)),2))
        print(cc,c,'Entropy= ',np.round(np.mean(np.nanmean(EntrGr,1)),2),'+/-',np.round(np.std(np.nanmean(EntrGr,1)),2))
        print(cc,c,'tot scanpath len = ',np.round(np.mean(np.nanmean(self.len_scanpath[Idx,:],1)),2),'+/-',np.round(np.std(np.nanmean(self.len_scanpath[Idx,:],1)),2),'pix')
        print(cc,c,'saccade amplitude = ',np.round(np.mean(np.nanmean(self.sacc_ampl[Idx,:],1)),2),'+/-',np.round(np.std(np.nanmean(self.sacc_ampl[Idx,:],1)),2),'pix')
        print('')
        MeanPlot(self.np,FixGr,yLab='Num Fixations',xtickL=self.stimuli,label=c,color=Cols[cc],ax=ax[0,0])
        MeanPlot(self.np,EntrGr,yLab='Entropy',xtickL=self.stimuli,label=c,color=Cols[cc],ax=ax[0,1])
        MeanPlot(self.np,self.len_scanpath[Idx,:],yLab='tot scanpath len (pix)',xtickL=self.stimuli,label=c,color=Cols[cc],ax=ax[1,0])
        MeanPlot(self.np,self.sacc_ampl[Idx,:],yLab='saccade amplitude (pix)',xtickL=self.stimuli,label=c,color=Cols[cc],ax=ax[1,1])
    
    
    plt.legend()
    plt.tight_layout()
    
    for gr1 in range(len(self.Conds)):
        for gr2 in range(len(self.Conds)):
            if gr1 <gr2:
                print()
                t,p=stats.ttest_ind(Entrs[gr1],Entrs[gr2])
                print(' ')
                print('Overall group differences: ',self.Conds[gr1],'vs',self.Conds[gr2] )
                print('Entropy t=',np.round(t,4),', p=',np.round(p,4))
                #if pglib:
                 #   pg.ttest(Fixies[0],Fixies[1],paired=False)
                #else:
                t,p=stats.ttest_ind(Fixies[gr1],Fixies[gr2])
                print('Num Fix t=',np.round(t,4),', p= ',np.round(p,4))
                t,p=stats.ttest_ind(ScanpLs[gr1],ScanpLs[gr2])
                
            
                print('Scanpath lengths t=',np.round(t,4),', p=',np.round(p,4))
                t,p=stats.ttest_ind(SaccAmpls[gr1],SaccAmpls[gr2])
            
                print('Saccade amplitudes t=',np.round(t,4),', p=',np.round(p,4))
                print(' ')

    
    return 

    
def CompareGroupsHeatmap(self,Stim,betwcond,StimPath='',SD=25,CutArea=0,Conds=0,Center=0,substring=False):
    ''' 
    Description: visualize group heatmap, along with heatmap difference.

    Arguments:
    Stim (str): The stimulus for which the heatmap is generated.
    betwcond (str): The condition for between-group heatmap comparison.
    StimPath (str, optional): Path to the stimulus. Default is an empty string.
    SD (int, optional): Optional parameter for heatmap smoothness, in pixels. Default is 25.
    CutArea (int, optional): Cut fixations. For example if you use '1', it shows 99% percentile of fixations. Default is 0.
    Conds (int or list, optional): use automatically detected conditions conditions, as provided in betweencond column
        othewise Conds=['MyCond1' MyCond2'], if we want to specify the order of access for betweencond column.
    center: if stimulus area does not start at pixel 0
    '''
    if substring:
        self.stimuli=self.stimuli.astype('str')
        stimn=np.char.find(self.stimuli,Stim)
        Stims=self.stimuli[stimn>-1]
        stimn=np.nonzero(stimn>-1)[0]
        stimShow=Stims[0]
        print('stimns found:',stimn,Stims)
    else:
        stimShow=copy.copy(Stim)

    WhichC,WhichCN=self.GetGroups(betwcond)
    if hasattr(self,'subjects')==0:
        self.GetParams()    
    #Cols=['darkred','cornflowerblue']
    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,8)) 
    
    if CutArea:
        FixCounts=self.FixCountCalc(Stim,CutAct=1,substring=substring) 
    else:
        FixCounts=self.FixCountCalc(Stim,CutAct=0,substring=substring) 
    assert np.sum(FixCounts)>0,'!!no fixations found'
    hmaps=[]
    
    if type(Conds)==int:    
        Conditions=np.copy(self.Conds)
    else:
        print('use provided conditions: ' ,Conds)
        Conditions=np.copy(Conds)
    for cc,c in enumerate(Conditions):
        Idx=np.nonzero(WhichCN==c)[0]   
        if substring:
            if np.sum(self.data.Stimulus[self.data['group']==self.Conds[cc]]==Stims[0])>0:
                stims=Stims[0] # select which of the two  image files to show (assuming two similar named image files)
            else:
                stims=Stims[1]
        else:
            stims=copy.copy(Stim)
        print(cc,c,stims)
        hmap=self.Heatmap(stims,SD=SD,Ind=0,Vis=1,FixCounts=FixCounts[Idx,:,:],CutArea=CutArea,center=Center,substring=False,ax=ax[0,cc])
        ax[0,cc].set_title(c)
       # ax[0,cc].colorbar()
        hmaps.append(hmap)
    if hasattr(self,'images'):
        if Center:
            xs1=(self.x_size-np.shape(self.images[stimShow])[1])/2
            xs2=self.x_size-xs1
            ys1=(self.y_size-np.shape(self.images[stimShow])[0])/2
            ys2=self.y_size-ys1
            ax[1,0].imshow(self.images[stimShow],extent=[xs1,xs2,ys2,ys1])
        else:
            ax[1,0].imshow( self.images[stimShow])

    Diff=hmaps[0]-hmaps[1]
    
    im=ax[1,0].imshow(Diff,cmap='RdBu', vmin=-np.nanmax(np.abs(Diff)), vmax=np.nanmax(np.abs(Diff)),alpha=.5)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title(str(Conditions[0])+' - '+str(Conditions[1]))
    cbar=plt.colorbar(im,ax=ax[1,0])
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(str(Conditions[0])+'<---->'+str(Conditions[1]), rotation=270)
    if hasattr(self,'images'):
        if Center:
            ax[1,1].imshow( self.images[stimShow],extent=[xs1,xs2,ys2,ys1])
        else:
            ax[1,1].imshow( self.images[stimShow])
    im=ax[1,1].imshow(np.abs(Diff), vmin=0, vmax=np.nanmax(np.abs(Diff)),alpha=.5)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    plt.colorbar(im,ax=ax[1,1])
    ax[1,1].set_title('Absolute diff: '+str(np.round(np.nansum(np.abs(Diff)),3)))
    plt.tight_layout()
    return 

    
    
def CompareWithinGroupsFix(self,withinColName):
    '''
    Description: Run fixation comparisons within groups defined by a category column. Makes plots and prints descriptive stats.
    
    Arguments:
    withinColName (str): The name of the categorical column defining groups for analysis.
    '''
    
    WhichC=self.GetCats(withinColName)

    if hasattr(self,'entropies')==False:   # check if entropy has already been calculated
        print('Calculating entropy')
        Entropies,self.entropmax,self.entropies_ind=self.GetEntropies()
    #Cols=['darkred','cornflowerblue']
#        plt.figure(figsize=(8,8))
    
    Entrs=[]
    Fixies=[]
    ScanpLs=[]
    SaccAmpls=[] 
    for cc,c in enumerate(self.WithinConds):
        print(cc,'Category',c)
        Idx=np.nonzero(WhichC==c)[0]
        FixGr=np.array(self.nfix[:,Idx])
        EntrGr=self.entropies_ind[:,Idx]
        Entrs.append(np.nanmean(EntrGr,1))
        Fixies.append(np.nanmean(FixGr,1))
        ScanpLs.append(np.nanmean(self.len_scanpath[:,Idx],1))
        SaccAmpls.append(np.nanmean(self.sacc_ampl[:,Idx],1))
     
        
        print(cc,c,'Num fix= ',np.round(np.mean(np.nanmean(FixGr,1)),2),'+/-',np.round(np.std(np.nanmean(FixGr,1)),2))
        print(cc,c,'Entropy= ',np.round(np.mean(np.nanmean(EntrGr,1)),2),'+/-',np.round(np.std(np.nanmean(EntrGr,1)),2))
        print(cc,c,'tot scanpath len = ',np.round(np.mean(np.nanmean(self.len_scanpath[:,Idx],1)),2),'+/-',np.round(np.std(np.nanmean(self.len_scanpath[:,Idx],1)),2),'pix')
        print(cc,c,'saccade amplitude = ',np.round(np.mean(np.nanmean(self.sacc_ampl[:,Idx],1)),2),'+/-',np.round(np.std(np.nanmean(self.sacc_ampl[:,Idx],1)),2),'pix')
        print('')
    return
 
   
    
    
def FixDurProgGroups(self,colName,nfixmax=10,between=0):
    ''' 
    Description: Calculate and visualize fixation duration progression within or between groups defined by a category column, for the first x number of fixations of each trial

    Arguments:
    colName (str): The name of the category column defining groups for analysis.
    nfixmax (int): The maximum number of fixations to consider in the progression (default: 10).
    between: if True between group comparison, by default expects within group comparison (for groups of stimuli)
    '''
    self.FixDurProg(nfixmax=nfixmax,Stim=0,Vis=0)
    print(np.shape(self.durprog))
    fig,ax=plt.subplots()
    
    if between:
        WhichC,WhichCN=self.GetGroups(colName)
    else:
        WhichC=self.GetCats(colName)
    for cc,c in enumerate(self.WithinConds):
        if between:
            Idx=np.nonzero(WhichC==cc)[0]
        else:
            Idx=np.nonzero(WhichC==c)[0]

        print('group',cc,c)
        if between:
            Y=np.nanmean(np.nanmean(self.durprog[Idx,:,:],1),0)
            Err=stats.sem(np.nanmean(self.durprog[Idx,:,:],1),0,nan_policy='omit')
        else:
            Y=np.nanmean(np.nanmean(self.durprog[:,Idx,:],1),0)
            Err=stats.sem(np.nanmean(self.durprog[:,Idx,:],1),0,nan_policy='omit')
        PlotDurProg(nfixmax,Y,Err,c,ax=ax)
    plt.legend()




def BinnedDescriptivesGroups(self,withinColName):
    ''' time-binned within trial descriptive progression, groups of stimuli'''
    if hasattr(self,'binFixL')==False: 
        print('run BinnedDescriptives first, than call this function for group wise visualization')
    WhichC=self.GetCats(withinColName)
    Colors=['navy','salmon','olive','orange','gray']
    fig,ax=plt.subplots(nrows=3,ncols=1,figsize=(4,10))
    for cc,c in enumerate(self.WithinConds):
        Idx=np.nonzero(WhichC==c)[0]
        axout=VisBinnedProg(self.tbins,np.nanmean(self.binFixL[:,Idx,:],1),'fixation duration (ms)',col=Colors[cc],label=c,axin=ax[0])
        axout=VisBinnedProg(self.tbins,np.nanmean(self.saccadeAmp[:,Idx,:],1),'saccade ampl (pixel)',col=Colors[cc],label=c,axin=ax[1])
        axout=VisBinnedProg(self.tbins,np.nanmean(self.totLscanpath[:,Idx,:],1),'scanpath length (pixel)',col=Colors[cc],label=c,axin=ax[2])
        
        ax1,ax2=JointBinnedPlot(self.tbins,np.nanmean(self.binFixL[:,Idx,:],1),np.nanmean(self.saccadeAmp[:,Idx,:],1),ylabel1='fixation duration (ms)',ylabel2='saccade ampl (pixel)')
        ax1.set_title(c)
        

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()      
    
    
         
    
def CompareGroupsMat(self,group,indsimmat):
    ''' 
    calculates  average within and between group values from inividual matrix differences
    group: expected column for between group comparison
    indsimmat: individual differences in the format (stimulus*subject*subject) '''
    groups,grarray=self.GetGroups(group)
    grs=np.unique(groups)
    print('groups: ',groups)
    Diffs=np.zeros((self.np,len(grs),len(grs)))
    for cg1,gr1 in enumerate(grs):
        for cg2,gr2 in enumerate(grs):
            for cs in range(self.np):
                Diffs[cs,cg1,cg2]=np.nanmean(indsimmat[cs,groups==cg1,:][:,groups==cg2])
    return Diffs
        
  
    