  
# this file contains the EyeData methods for bewteen or within group comparisons.
# not all functions can do both ---- yet  

import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import copy

# import  library helper functions. 
from .statshelper import SaliencyMapFilt,SaccadesTrial,ScanpathL,StatEntropy,BonfHolm,ReduceCounts
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
    
    print('!runnning between group comparison')
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
 
   
def CompareGroupsHeatmap(self,Stim,betwcond,SD=25,CutArea=0,Conds=0,cmap_ind='plasma',cmap_diff='RdYlBu',cmap_abs='Greens',alpha=.5,cutoff='median',downsample=8, Nrand=100):
    ''' 
    DESCRIPTION: visualize  heatmap fopr two groups, 
    subplot 1: group 1
    subplot 2: group 2
    subplot 3: difference heatmap (raw value)
    subplot 4: difference heatmap (absolute value)
    

    ARGUMENTS:
    
    Stim (str): The stimulus name
    betwcond (str): The condition for between-group heatmap comparison.

    
    OPTIONAL PARAMETERS
    
    StimPath (str, optional): Path to the stimulus. Default is an empty string. if stimuli loaded before, this is not necessary
    SD (int, optional): Optional parameter for heatmap smoothness, in pixels. Default is 25.
    CutArea (int, optional): Cut fixations. For example if you use '1', it shows 99% percentile of fixations. Default is 0. SEt to 1, if stimulus does not cover the screen size eg: for  portrait orientation
    Conds (int or list, optional): use automatically detected conditions conditions, as provided in betweencond column
        othewise Conds=['MyCond1' MyCond2'], if we want to specify the order of access for betweencond column.  (could be useful if there are more then 2 conditions, and we want to select 2 to contrast)
    center: if stimulus area does not start at pixel 0, shifts image display using the plt.imshow(image, extent=)
    cmap_ind=colormap for  each of the two group heatmaps (see matplotlib colormaps for options: https://matplotlib.org/stable/users/explain/colors/colormaps.html) (two top figures)
    cmap_diff: colormap of difference heatmap (raw) in two directions (bottom left figure)
    cmap_abs: colormap of absolute difference heatmap (bottom right figure)

    alpha= transparency- 0-1 higher values less transparent
    cutoff: shows areas below this threshold as blank
    twostim: compare two similar stimuli, either provided as list or  match stimuli based on part of string --- if we want to compare two differently named stimuli, with part of the stimulus name matching
    downsample: downampling size 8 reduced by a factor of 8*8 pixels for example (using skimage)
    Nrand: number of random permutations to compute, default 100, for actual stats long run time and at least 1000 permutations are recommended, if set to 0 random permutation comparisonno performed
    # optional paramter to control color of difference heatmap
    '''

    if hasattr(self,'subjects')==0:
        self.GetParams()   
        
  

    FixCounts=self.FixCountCalc(Stim,CutAct=CutArea) 
    if np.sum(fixcount) <= 0:
        raise ValueError('!!no fixations found')

    print('dimensions=',np.shape(FixCounts))
    Reduced=ReduceCounts(FixCounts,downsample)

    print('reduced dims',np.shape(Reduced))


    WhichC,WhichCN=self.GetGroups(betwcond)
    if type(Conds)==int:    
        Conditions=np.copy(self.Conds)
    else:
        print('use provided conditions: ' ,Conds) 
        Conditions=np.copy(Conds)
    N1=np.sum(WhichCN==Conditions[0])
    print(f'num observers in group 1: {N1}') 
    print(f'num observers in group 2: {np.sum(WhichCN==Conditions[1])}') 


    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,8)) 
    hmaps=[]
    hmapsred=[]## reduced heatmaps, downsampled with scikit image (mean based downsampling)
    for cc,c in enumerate(Conditions):
            
        Idx=np.nonzero(WhichCN==c)[0]   
        heatmap=SaliencyMapFilt(FixCounts[Idx,:,:],SD=SD,Ind=0)
        hmaps.append(heatmap)
        
        self.VisHeatmap(Stim,heatmap,ax=ax[0,cc],cutoff=cutoff,alpha=alpha,cmap=cmap_ind,title=c)
        
        
        redcount=ReduceCounts(FixCounts[Idx,:,:],downsample)
        hmap_red=SaliencyMapFilt(redcount,SD=8,Ind=0)
        hmapsred.append(hmap_red)
   
    # difference heatmp
    Diff=hmaps[0]-hmaps[1]


    
    colorbarlabel=str(Conditions[0])+'<---->'+str(Conditions[1])
   
    self.VisHeatmap(Stim,Diff,ax=ax[1,0],cutoff=cutoff,alpha=alpha,cmap=cmap_diff,cbar=True,cbarlabel=colorbarlabel,title='difference')
    
    self.VisHeatmap(Stim,np.abs(Diff),ax=ax[1,1],cutoff=cutoff,alpha=alpha,cmap=cmap_abs,cbar=True,cbarlabel=' ',title='absolute difference')
    
    
    
    DiffRed=hmapsred[0]-hmapsred[1]
   
    ### calculate permuted difference heatmaps
    DiffPerm=np.zeros(Nrand)
    print(f'{Nrand} permutations starting')
    if Nrand>0:
        for n in range(Nrand):
            Idxs=np.random.permutation(np.shape(Reduced)[0])         
            hmap1=SaliencyMapFilt(Reduced[Idxs[0:N1],:,:],SD=downsample,Ind=0)
            hmap2=SaliencyMapFilt(Reduced[Idxs[N1:],:,:],SD=downsample,Ind=0)
            DiffPerm[n]=np.nansum(np.abs(hmap1-hmap2))
        
   


    truereddiff=np.nansum(np.abs(DiffRed))
    # visualize permuted difference heatmap distribution
    
    if Nrand>0:
        
        HistPlot(DiffPerm,xtickL='group difference',ax=0,ylab='num random permutations',verline=truereddiff,title=f' {Stim} permuted {Nrand} vs true diff - p={np.sum(DiffPerm>truereddiff)/Nrand}',mean=False)
        
   

    return np.nansum(np.abs(Diff))#,truereddiff,DiffPerm

def CompareStimHeatmap(self,Stim,SD=25,CutArea=0,Conds=0,cmap_ind='plasma',cmap_diff='RdYlBu',cmap_abs='Greens',alpha=.5,cutoff='median',downsample=8, Nrand=100):
    ''' 
    DESCRIPTION: visualize  heatmap fopr two groups, 
    subplot 1: group 1
    subplot 2: group 2
    subplot 3: difference heatmap (raw value)
    subplot 4: difference heatmap (absolute value)
    

    ARGUMENTS:
    
    Stim (str): list of two stimuli for which the heatmap is generated, it can also find similarly named stimuli based on common substring
    
    
    
    OPTIONAL PARAMETERS
    
    
    SD (int, optional): Optional parameter for heatmap smoothness, in pixels. Default is 25.
    CutArea (int, optional): Cut fixations. For example if you use '1', it shows 99% percentile of fixations. Default is 0. SEt to 1, if stimulus does not cover the screen size eg: for  portrait orientation
    Conds (int or list, optional): use automatically detected conditions conditions, as provided in betweencond column
        othewise Conds=['MyCond1' MyCond2'], if we want to specify the order of access for betweencond column.  (could be useful if there are more then 2 conditions, and we want to select 2 to contrast)
    center: if stimulus area does not start at pixel 0, shifts image display using the plt.imshow(image, extent=)
    cmap_ind=colormap for  each of the two group heatmaps (see matplotlib colormaps for options: https://matplotlib.org/stable/users/explain/colors/colormaps.html) (two top figures)
    cmap_diff: colormap of difference heatmap (raw) in two directions (bottom left figure)
    cmap_abs: colormap of absolute difference heatmap (bottom right figure)

    alpha= transparency- 0-1 higher values less transparent
    cutoff: shows areas below this threshold as blank
    twostim: compare two similar stimuli, either provided as list or  match stimuli based on part of string --- if we want to compare two differently named stimuli, with part of the stimulus name matching
    downsample: downampling size 8 reduced by a factor of 8*8 pixels for example (using skimage)
    Nrand: number of random permutations to compute, default 100, for actual stats long run time and at least 1000 permutations are recommended, if set to 0 random permutation comparisonno performed
    # optional paramter to control color of difference heatmap
    '''

    if hasattr(self,'subjects')==0:
        self.GetParams()   
    
    if type(Stim)==list:
        if len(Stim) != 2:
            raise ValueError('Length 2 list expected')

        Stims=np.array(Stim)
        stimn=np.zeros(2)
        for cs,s in enumerate(Stims):
            stimn[cs]=np.nonzero(self.stimuli==s)[0]
            if stimn[cs] <= -1:
                raise ValueError('Stim not found')

    else:
        self.stimuli=self.stimuli.astype('str')
        stimn=np.char.find(self.stimuli,Stim)
        Stims=self.stimuli[stimn>-1]
        stimn=np.nonzero(stimn>-1)[0]
        print('stimns found:',stimn,Stims)
    stimn=np.intp(stimn)
    stimShow=Stims[0]  # for comparison figures, first figure is used
    StimIdxs=self.GetStimSubjMap(Stims)


  
    

    FixCounts=[]
    for s in Stims:
        fixcount=self.FixCountCalc(s,CutAct=0)
        if np.sum(fixcount) <= 0:
            raise ValueError('!!no fixations found')

        HasFixIdx=np.sum(np.sum(fixcount,2),1)>0
        fixcount=fixcount[HasFixIdx,:,:]
        FixCounts.append(fixcount)
        print(s,np.shape(FixCounts[-1]))
    Reduced1=ReduceCounts(FixCounts[0],downsample)
    Reduced2=ReduceCounts(FixCounts[1],downsample)
    Reduced=np.concatenate((Reduced1,Reduced2),axis=0)


    print('reduced dims',np.shape(Reduced))
    
    N1=len(StimIdxs[0])
    print(f'num observers seen stimulus {Stims[0]}: {N1}') 
    print(f'num observers seen stimulus {Stims[1]}: {len(StimIdxs[1])}') 



    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,8)) 
    hmaps=[]
    hmapsred=[]## reduced heatmaps, downsampled with scikit image (mean based downsampling)
    
    for cs,s in enumerate(stimn):
            
        
        heatmap=SaliencyMapFilt(FixCounts[cs],SD=SD,Ind=0)
        hmaps.append(heatmap)
        self.VisHeatmap(Stims[cs],heatmap,ax=ax[0,cs],cutoff=cutoff,alpha=alpha,cmap=cmap_ind,title=self.stimuli[s])
        
        
        redcount=ReduceCounts(FixCounts[cs],downsample)
        hmap_red=SaliencyMapFilt(redcount,SD=8,Ind=0)
        hmapsred.append(hmap_red)
   
    # difference heatmp
    Diff=hmaps[0]-hmaps[1]


    colorbarlabel=str(Stims[0])+'<---->'+str(Stims[1])

    self.VisHeatmap(stimShow,Diff,ax=ax[1,0],cutoff=cutoff,alpha=alpha,cmap=cmap_diff,cbar=True,cbarlabel=colorbarlabel,title='difference')
    
    self.VisHeatmap(stimShow,np.abs(Diff),ax=ax[1,1],cutoff=cutoff,alpha=alpha,cmap=cmap_abs,cbar=True,cbarlabel=' ',title='absolute difference')
    
    
    
    DiffRed=hmapsred[0]-hmapsred[1]
   
    ### calculate permuted difference heatmaps
    DiffPerm=np.zeros(Nrand)
    print(f'{Nrand} permutations starting')
    if Nrand>0:
        for n in range(Nrand):
            Idxs=np.random.permutation(np.shape(Reduced)[0])         
            hmap1=SaliencyMapFilt(Reduced[Idxs[0:N1],:,:],SD=downsample,Ind=0)
            hmap2=SaliencyMapFilt(Reduced[Idxs[N1:],:,:],SD=downsample,Ind=0)
            DiffPerm[n]=np.nansum(np.abs(hmap1-hmap2))
        
   


    truereddiff=np.nansum(np.abs(DiffRed))
    # visualize permuted difference heatmap distribution
    
    if Nrand>0:
        
        HistPlot(DiffPerm,xtickL='group difference',ax=0,ylab='num random permutations',verline=truereddiff,title=f' {Stim} permuted {Nrand} vs true diff - p={np.sum(DiffPerm>truereddiff)/Nrand}',mean=False)
        
      

    return np.nansum(np.abs(Diff))#,truereddiff,DiffPerm

    
 
    
 
    
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
        print('running between group comparison')
        WhichC,WhichCN=self.GetGroups(colName)
    else:
        print('running within group comparison, provide between=True for between group comparisons')

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




def BinnedDescriptivesGroups(self,colName,between=0, MeasureNames=['fixation duration (ms)','saccade ampl (pixel)','scanpath length (pixel)'],Colors=['navy','salmon','olive','orange','gray'],ylims=[False,False,False]):
    ''' time-binned within trial descriptive progression, groups of stimuli or between groups of participants'''
    if hasattr(self,'binFixL')==False: 
        
        warnings.warn('run BinnedDescriptives first to specify time columns and bins, than call this function for group wise visualization')
        self.BinnedDescriptives()
    if between:
         print('running between group comparison')

         WhichC,WhichCN=self.GetGroups(colName)
    else:
         print('running within group comparison, provide between=True for between group comparisons')

         WhichC=self.GetCats(colName)
 #   print(WhichC)
    
    
    Measures=[self.binFixL,self.saccadeAmp,self.totLscanpath]
   
    fig,ax=plt.subplots(nrows=3,ncols=1,figsize=(4,10))
    
    Dats=[{},{}]
   
    for cc,c in enumerate(self.WithinConds):
        if between:
            Idx=np.nonzero(WhichC==cc)[0]
            print(Idx)
            ax1,ax2=JointBinnedPlot(self.tbins,np.nanmean(self.binFixL[Idx,:,:],1),np.nanmean(self.saccadeAmp[Idx,:,:],1),ylabel1='fixation duration (ms)',ylabel2='saccade ampl (pixel)')
        else:
            Idx=np.nonzero(WhichC==c)[0]
            ax1,ax2=JointBinnedPlot(self.tbins,np.nanmean(self.binFixL[:,Idx,:],1),np.nanmean(self.saccadeAmp[:,Idx,:],1),ylabel1='fixation duration (ms)',ylabel2='saccade ampl (pixel)')

        for cm,m in enumerate(Measures):
   
            if between:
                dat=np.nanmean(m[Idx,:,:],1)
            else:
                dat=np.nanmean(m[:,Idx,:],1)
            axout=VisBinnedProg(self.tbins,dat,MeasureNames[cm],col=Colors[cc],label=c,axin=ax[cm],ylim=ylims[cm])
            Dats[cc][MeasureNames[cm]]=dat
        ax1.set_title(c)
    tvals={}
    pvals={}
    for cm,m in enumerate(Measures):
        if between:
            t,p=stats.ttest_ind(Dats[0][MeasureNames[cm]],Dats[1][MeasureNames[cm]])
        else:
            t,p=stats.ttest_rel(Dats[0][MeasureNames[cm]],Dats[1][MeasureNames[cm]])
        tvals[MeasureNames[cm]]=t
        pvals[MeasureNames[cm]]=p
        ypos=np.nanmean(np.concatenate((Dats[0][MeasureNames[cm]],Dats[1][MeasureNames[cm]])))
        for c,p_i in enumerate(p):
            xpos=(self.tbins[c]+self.tbins[c+1])/2
            if p_i<.01:
                ax[cm].scatter(xpos,ypos,marker='*', s=30,color='k')
            elif  p_i<.05:
                ax[cm].scatter(xpos,ypos,marker='*', s=17,color='k')


   #     print(cm,MeasureNames[cm],t,p)
            

    for a in range(3):
        ax[a].legend()
    plt.tight_layout()      
    return tvals, pvals
    
    
         
    
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
        


def CompareGroupsGridFix(self,Stim,betwcond,Conds=0,nhor=5,nver=5,cmap_ind='plasma',cmap_diff='RdYlBu',alpha=.5,t_abs=False,timemin=0, timemax=np.inf, timecol=0,useT=True,cutoff=-1): 
    ''' 

    Stim: stimulus name
    betwcond: between subject condition (if substring=True, this is not used)
    
    Conds: explicitly provide conditions (if there are more than 2, this is necessary)
    t_abs: default=False,  absolute t value vs raw t-values grid
    nhor: number of horizonal cells for the grid
    nver: number of vertical cells for the grid
    center (true): stimulus position correction (based on difference between stimulus and screen resolution), stimulus must be presented centrally!
    substring: if two different stimuli are compared,  can be paired stimuli have to found based on common part in stimulusID
    cmap_ind: colormap for the heatmap of each group, default: 'plasma'
    cmap_diff: colormap for difference heatmaps- default 'RdYlBu'--- ideally use divergent heatmaps
    

    '''
   
    stimn=np.nonzero(self.stimuli==Stim)[0]

    WhichC,WhichCN=self.GetGroups(betwcond)

    if type(Conds)==int:    
        Conditions=np.copy(self.Conds)
    else:
        print('use provided conditions: ' ,Conds)
        Conditions=np.copy(Conds)
      
   
    statPMat,statEntropyMat=self.CalcStatPs(nhor,nver,MinFix=5,InferS=2,timemin=timemin, timemax=timemax, timecol=timecol)

   # if substring and len(stimn)==2:

    fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
    statmats=[]
    
    for ccond,cond in enumerate(Conditions):
        Idx=WhichCN==cond
        print(np.shape(np.nanmean(statPMat[Idx,stimn,:,:],0)))
        Statpm=np.nanmean(statPMat[Idx,stimn,:,:],0)
        if cutoff>-1:
            Statpm[Statpm<np.nanpercentile(Statpm,cutoff)]=np.NAN
        self.VisGrid(Statpm,Stim,ax=ax[0,ccond],alpha=alpha,cmap=cmap_ind)
        ax[0,ccond].set_title(cond)
        statmats.append(statPMat[Idx,stimn,:,:])
    diffmat=np.nanmean(statmats[0],0)-np.nanmean(statmats[1],0) 

    tt,pp=np.zeros((nver,nhor)),np.zeros((nver,nhor))
    for ch in range(nhor):
        for cv in range(nver):
            d1,d2=statmats[0][:,cv,ch],statmats[1][:,cv,ch]
           
            tt[cv,ch],pp[cv,ch] = stats.ttest_ind(d1[np.isfinite(d1)],d2[np.isfinite(d2)])

    cbar=self.VisGrid(diffmat,Stim,ax=ax[1,0],alpha=.7,cmap=cmap_diff,vmax=np.nanmax(np.abs(diffmat)),cbar=True)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(str(Conditions[0])+'<---->'+str(Conditions[1]), rotation=270)

    ax[1,0].set_title('difference')
    if useT:
        if t_abs:
            cbar=self.VisGrid(np.abs(tt),Stim,ax=ax[1,1],alpha=.7,cmap='Greens',cbar=True)
            ax[1,1].set_title('abs t-values')
        else:
            cbar=self.VisGrid(tt,Stim,ax=ax[1,1],alpha=.7,cmap=cmap_diff,vmax=4,cbar=True)
            ax[1,1].set_title('t-value')
            cbar.ax.get_yaxis().labelpad = 30
            cbar.ax.set_ylabel(str(Conditions[0])+'<---->'+str(Conditions[1]), rotation=270)

    else:
        cbar=self.VisGrid(np.log(pp),Stim,ax=ax[1,1],alpha=.7,cmap='Greens',cbar=True)
        ax[1,1].set_title('log(p)')

    self.Highlight_Sign(Stim,pp,ax[1,1]) # highlight significant cells, by showing grid buondaries (dashed (p<.05) or solid line (p<.01) 
    
    print('num significant uncorrected: ',np.sum(pp<.05))
    print('num significant Bonferroni - Holm corrected: ',BonfHolm(pp))

    return tt,pp


def CompareStimGridFix(self,Stim,Conds=0,nhor=5,nver=5,cmap_ind='plasma',cmap_diff='RdYlBu',alpha=.5,t_abs=False,timemin=0, timemax=np.inf, timecol=0,useT=True,cutoff=-1): 
    ''' 

    Stim: slist of two stimulus names to compare, or common substring to find pairs of stimuli (for the latter two options)
    
    Conds: explicitly provide conditions (if there are more than 2, this is necessary)
    t_abs: default=False,  absolute t value vs raw t-values grid
    nhor: number of horizonal cells for the grid
    nver: number of vertical cells for the grid
    center (true): stimulus position correction (based on difference between stimulus and screen resolution), stimulus must be presented centrally!
    substring: if two different stimuli are compared,  can be paired stimuli have to found based on common part in stimulusID
    cmap_ind: colormap for the heatmap of each group, default: 'plasma'
    cmap_diff: colormap for difference heatmaps- default 'RdYlBu'--- ideally use divergent heatmaps
    

    '''
    
    
    if type(Stim)==list:
        if len(Stim) != 2:
            raise ValueError('Length 2 list expected')

        Stims=np.array(Stim)
        stimn=np.zeros(2)
        for cs,s in enumerate(Stims):
            stimn[cs]=np.nonzero(self.stimuli==s)[0]
            if stimn[cs] <= -1:
                raise ValueError('Stim not found')

    else:    
        self.stimuli=self.stimuli.astype('str')
        stimn=np.char.find(self.stimuli,Stim)
        
        Stims=self.stimuli[stimn>-1]
        stimn=np.nonzero(stimn>-1)[0]
    stimn=np.intp(stimn)
    stimShow=Stims[0]
    print('stimns found:',stimn,Stims)
   
    statPMat,statEntropyMat=self.CalcStatPs(nhor,nver,MinFix=5,InferS=2,timemin=timemin, timemax=timemax, timecol=timecol)

   # if substring and len(stimn)==2:

    fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
    statmats=[]
    for cs,s in enumerate(stimn):
        Statpm=np.nanmean(statPMat[:,s,:,:],0)
        if cutoff>-1:
            Statpm[Statpm<np.nanpercentile(Statpm,cutoff)]=np.NAN
        self.VisGrid(Statpm,Stims[cs],ax=ax[0,cs],alpha=alpha,cmap=cmap_ind)
        ax[0,cs].set_title(Stims[cs])
        statmats.append(statPMat[:,s,:,:])


    diffmat=np.nanmean(statPMat[:,stimn[0],:,:],0)-np.nanmean(statPMat[:,stimn[1],:,:],0) 
    
  

    tt,pp=np.zeros((nver,nhor)),np.zeros((nver,nhor))
    for ch in range(nhor):
        for cv in range(nver):
            d1,d2=statmats[0][:,cv,ch],statmats[1][:,cv,ch]
            tt[cv,ch],pp[cv,ch] = stats.ttest_ind(d1[np.isfinite(d1)],d2[np.isfinite(d2)])



    cbar=self.VisGrid(diffmat,stimShow,ax=ax[1,0],alpha=.7,cmap=cmap_diff,vmax=np.nanmax(np.abs(diffmat)),cbar=True)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(str(Stims[0])+'<---->'+str(Stims[1]), rotation=270)
  
    ax[1,0].set_title('difference')
    if useT:
        if t_abs:
            cbar=self.VisGrid(np.abs(tt),stimShow,ax=ax[1,1],alpha=.7,cmap='Greens',cbar=True)
            ax[1,1].set_title('abs t-values')
        else:
            cbar=self.VisGrid(tt,stimShow,ax=ax[1,1],alpha=.7,cmap=cmap_diff,vmax=4,cbar=True)
            ax[1,1].set_title('t-value')
            cbar.ax.get_yaxis().labelpad = 30
            cbar.ax.set_ylabel(str(Stims[0])+'<---->'+str(Stims[1]), rotation=270)       

    else:
        cbar=self.VisGrid(np.log(pp),stimShow,ax=ax[1,1],alpha=.7,cmap='Greens',cbar=True)
        ax[1,1].set_title('log(p)')

  
    self.Highlight_Sign(stimShow,pp,ax[1,1]) # highlight significant cells, by showing grid buondaries (dashed (p<.05) or solid line (p<.01) 
    
    print('num significant uncorrected: ',np.sum(pp<.05))
    print('num significant Bonferroni - Holm corrected: ',BonfHolm(pp))

    return tt,pp


    