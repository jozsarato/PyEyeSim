  
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
from .statshelper import SaliencyMapFilt,SaccadesTrial,ScanpathL,StatEntropy
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim ,CheckCoor

def CompareGroupsFix(self,betwcond):
    '''
    Description: Run a set of between-group fixation comparisons, generate plots, and print descriptive statistics.
    
    Arguments: 
    betwcond (str): Name of the conditions for between-group fixation comparisons.
    '''
    
    WhichC,WhichCN=self.GetGroups(betwcond)
    if hasattr(self,'entropies')==False:   # check if entropy has already been calculated
        print('Calculating entropy')
        Entropies,self.entropmax,self.entropies_ind=self.GetEntropies()
    Cols=['darkred','cornflowerblue']
    #plt.figure(figsize=(8,8))
    fig,ax=plt.subplots()
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
     
        
        print(cc,c,'Num fix= ',np.round(np.mean(np.nanmean(FixGr,1)),2),'+/-',np.round(np.std(np.nanmean(FixGr,1)),2))
        print(cc,c,'Entropy= ',np.round(np.mean(np.nanmean(EntrGr,1)),2),'+/-',np.round(np.std(np.nanmean(EntrGr,1)),2))
        print(cc,c,'tot scanpath len = ',np.round(np.mean(np.nanmean(self.len_scanpath[Idx,:],1)),2),'+/-',np.round(np.std(np.nanmean(self.len_scanpath[Idx,:],1)),2),'pix')
        print(cc,c,'saccade amplitude = ',np.round(np.mean(np.nanmean(self.sacc_ampl[Idx,:],1)),2),'+/-',np.round(np.std(np.nanmean(self.sacc_ampl[Idx,:],1)),2),'pix')

        MeanPlot(self.np,FixGr,yLab='Num Fixations',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc],ax=ax[0,0])
        MeanPlot(self.np,EntrGr,yLab='Entropy',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc],ax=ax[0,1])
        MeanPlot(self.np,self.len_scanpath[Idx,:],yLab='tot scanpath len (pix)',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc],ax=ax[1,0])
        MeanPlot(self.np,self.sacc_ampl[Idx,:],yLab='saccade amplitude (pix)',xtickL=self.stimuli,newfig=0,label=c,color=Cols[cc],ax=ax[1,1])
        
        
    t,p=stats.ttest_ind(Entrs[0],Entrs[1])
    print(' ')
    print('Overall group differences: ')
    print('Entropy t=',np.round(t,4),', p=',np.round(p,4))
    #if pglib:
     #   pg.ttest(Fixies[0],Fixies[1],paired=False)
    #else:
    t,p=stats.ttest_ind(Fixies[0],Fixies[1])
    print('Num Fix t=',np.round(t,4),', p= ',np.round(p,4))
    t,p=stats.ttest_ind(ScanpLs[0],ScanpLs[1])
    

    print('Scanpath lengths t=',np.round(t,4),', p=',np.round(p,4))
    t,p=stats.ttest_ind(SaccAmpls[0],SaccAmpls[1])

    print('Saccade amplitudes t=',np.round(t,4),', p=',np.round(p,4))


    plt.legend()
    plt.tight_layout()
    return 

    
def CompareGroupsHeatmap(self,Stim,betwcond,StimPath='',SD=25,CutArea=0,Conds=0,Center=0):
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
    WhichC,WhichCN=self.GetGroups(betwcond)
    if hasattr(self,'subjects')==0:
        self.GetParams()    
    #Cols=['darkred','cornflowerblue']
    plt.figure(figsize=(10,5))
   # FixCounts=self.FixCountCalc(Stim)
    
    if CutArea:
        FixCounts=self.FixCountCalc(Stim,CutAct=1) 
    else:
        FixCounts=self.FixCountCalc(Stim,CutAct=0) 
    assert np.sum(FixCounts)>0,'!!no fixations found'
    hmaps=[]
    
    if type(Conds)==int:    
        Conditions=np.copy(self.Conds)
    else:
        print('use provided conditions: ' ,Conds)
        Conditions=np.copy(Conds)
    for cc,c in enumerate(Conditions):
        Idx=np.nonzero(WhichCN==c)[0]   
        plt.subplot(2,2,cc+1)
        hmap=self.Heatmap(Stim,SD=SD,Ind=0,Vis=1,FixCounts=FixCounts[Idx,:,:],CutArea=CutArea,center=Center)
        plt.title(c)
        plt.colorbar()
        hmaps.append(hmap)
    plt.subplot(2,2,3)
    if hasattr(self,'images'):
        plt.imshow( self.images[Stim])

    Diff=hmaps[0]-hmaps[1]
    #plt.imshow(Diff,cmap='RdBu',alpha=.5)
    
    plt.imshow(Diff,cmap='RdBu', vmin=-np.nanmax(np.abs(Diff)), vmax=np.nanmax(np.abs(Diff)),alpha=.5)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(Conditions[0])+' - '+str(Conditions[1]))
    cbar=plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(str(Conditions[0])+'<---->'+str(Conditions[1]), rotation=270)
    plt.subplot(2,2,4)
    if hasattr(self,'images'):
        plt.imshow( self.images[Stim])
    plt.imshow(np.abs(Diff), vmin=0, vmax=np.nanmax(np.abs(Diff)),alpha=.5)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('Absolute diff: '+str(np.round(np.nansum(np.abs(Diff)),3)))
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
 
   
    
    
def FixDurProgGroups(self,withinColName,nfixmax=10):
    ''' 
    Description: Calculate and visualize fixation duration progression within groups defined by a category column.

    Arguments:
    withinColName (str): The name of the category column defining groups for analysis.
    nfixmax (int): The maximum number of fixations to consider in the progression (default: 10).
    '''
    self.FixDurProg(nfixmax=nfixmax,Stim=0,Vis=0)
    WhichC=self.GetCats(withinColName)
    for cc,c in enumerate(self.WithinConds):
        Idx=np.nonzero(WhichC==c)[0]
        Y=np.nanmean(np.nanmean(self.durprog[:,Idx],1),0)
        Err=stats.sem(np.nanmean(self.durprog[:,Idx],1),axis=0,nan_policy='omit')
        PlotDurProg(nfixmax,Y,Err,c)
    plt.legend()
