#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:58:03 2022

@author: jarato
"""

import numpy as np
import pandas as pd

# % import  library helper functions

from .visualhelper import MeanPlot, HistPlot
import matplotlib.pyplot as plt
from .statshelper import ScanpathL
import warnings

from .similarity import extract_heatmap_arrays, extract_angle_arrays, RSA_from_angles, RSA_from_heatmaps, extract_euc_dist_arrays, RSA_from_euc_dists, entropy_fix

class EyeData:

    from ._visuals import (
        VisScanPath,
        MySaccadeVis,
        VisLOOHMM,
        VisHMM,
        MyTrainTestVis,
        VisGrid,
        Highlight_Sign,
        VisHeatmap,VisSimmat
    )
    from ._dataproc import (
        GetParams,
        GetStimuli,
        GetFixationData,
        GetDurations,
        GetGroups,
        GetCats,
        GetSaccades,
        GetEntropies,
        InferSize,
        Heatmap,
        FixCountCalc,
        GetStimSubjMap,
    )
    from ._stats import (
        AngleCalc,
        AngtoPix,
        PixdoDeg,
        Entropy,
        FixDurProg,
        BinnedCount,
        GetInddiff,
        GetInddiff_v2,
        RunDiffDivs,
        GetBinnedStimFixS,
        StatPDiffInd2,
        StatPDiffInd1,
        CalcStatPs,
        CalcRets,
        CalcImmRets,
        BinnedDescriptives,
    )
    from ._comparegroups import (
        CompareGroupsFix,
        CompareWithinGroupsFix,
        FixDurProgGroups,
        BinnedDescriptivesGroups,
        CompareGroupsMat,
        CompareGroupsGridFix,
        CompareStimHeatmap,
        CompareStimGridFix,
    )
    from ._scanpathsim import (
        AOIFix,
        SacSimPipeline,
        SacSim1Group,
        SaccadeSel,
        ScanpathSim2Groups,
        SacSimPipelineAll2All,SacSim1GroupAll2All
        
    )
    
    from .similarity import RSA_heatmap_pipeline


    try:
        from ._hmm import (
            DataArrayHmm,
            MyTrainTest,
            FitLOOHMM,
            FitVisHMM,
            FitVisHMMGroups,
            HMMSimPipeline,
            HMMSimPipelineAll2All,
            HMMIndividual1Stim
        )
    except:
        warnings.warn(
            "hmmlearn not found, hidden markov model functionality will not work"
        )

    try:
        from ._comparegroups import CompareGroupsHeatmap
    except:
        warnings.warn(
            "scikit image not found, compare groups heatmap will not work - scikit image needed for downsampling"
        )

    def __init__(self, data, x_size, y_size):
        """
        Description: initalizing eye-tracking data object.

        Arguments:

        data (pandas.DataFrame): The eye-tracking data.
        x_size (int): Screen size in pixels (width).
        y_size (int): Screen size in pixels (height).
        """

        self.data = data
        self.x_size = x_size
        self.y_size = y_size

        print("dataset size: ", np.shape(self.data))
        print(
            "presentation size:  x=", self.x_size, "pixels y=", self.y_size, " pixels"
        )
        print(
            "presentation size:  x=", self.x_size, "pixels y=", self.y_size, " pixels"
        )
        DefColumns = {
            "Stimulus": "Stimulus",
            "subjectID": "subjectID",
            "mean_x": "mean_x",
            "mean_y": "mean_y",
        }

        for df in DefColumns:
            try:
                data[DefColumns[df]]
                print("column found: ", df, " default: ", DefColumns[df])
            except:
                print(
                    df,
                    " not found !!, provide column as",
                    df,
                    "=YourColumn , default: ",
                    DefColumns[df],
                )

    def info(self):
        """
        Description: prints screen information, dataset name and study design.
        """
        print("screen x_size", self.x_size)
        print("screen y_size", self.y_size)

    def data(self):
        """
        Description: shows dataset.
        """
        return self.data

    def setColumns(
        self,
        Stimulus="Stimulus",
        subjectID="subjectID",
        mean_x="mean_x",
        mean_y="mean_y",
        FixDuration=0,
    ):
        if type(FixDuration) != "int":
            self.data = self.data.rename(
                columns={
                    Stimulus: "Stimulus",
                    subjectID: "subjectID",
                    mean_x: "mean_x",
                    mean_y: "mean_y",
                    FixDuration: "duration",
                }
            )
        else:
            self.data = self.data.rename(
                columns={
                    Stimulus: "Stimulus",
                    subjectID: "subjectID",
                    mean_x: "mean_x",
                    mean_y: "mean_y",
                }
            )

    def setStimuliPath(
        self, StimPath=None, StimExt=".jpg", infersubpath=False, sizecorrect=True
    ):
        if StimPath == None:
            warnings.warn("Stim path not provided")
            return
        else:
            self.GetStimuli(
                StimExt, StimPath, infersubpath=infersubpath, sizecorrect=sizecorrect
            )
            print("stimuli loaded succesfully, access as self.images")

    def setSubjStim(self):
        try:
            subjs, stims = self.GetParams()
            print(
                "info found for "
                + str(len(subjs))
                + " subjects, and "
                + str(len(stims))
                + " stimuli"
            )
        except:
            warnings.warn("stimulus and subject info not found")

    def DataInfo(
        self,
        Stimulus="Stimulus",
        subjectID="subjectID",
        mean_x="mean_x",
        mean_y="mean_y",
        FixDuration=0,
        StimPath=None,
        StimExt=".jpg",
        infersubpath=False,
        Visual=False,
        sizecorrect=True,
    ):
        """
        Description: Provide information about amount of stimuli and subjects.
        Arguments:
        Stimulus (str): Column name for stimulus information in the eye-tracking data.
        subjectID (str): Column name for subject ID information in the eye-tracking data.
        mean_x (str): Column name for mean x-coordinate of fixations in the eye-tracking data.
        mean_y (str): Column name for mean y-coordinate of fixations in the eye-tracking data.
        FixDuration (int or str): Column name or integers for fixation duration in the eye-tracking data.
            If an integer, fixation duration column is assumed absent. It will be renamed "duration" afterwards
        StimPath (str): Path to stimuli. Set to 0 if not provided.
        StimExt (str): File extension of stimuli (default: '.jpg').
        infersubpath (bool): Flag to infer stimulus subpaths based on subject IDs (default: False).  -- if stimuli are stored in subfolders for multiple categories
        sizecorrect--> if True correct with stimulus resolution - screen difference, assuming central presentation
        """

        # pipeline
        self.setColumns(Stimulus, subjectID, mean_x, mean_y, FixDuration)

        self.setSubjStim()
        if sizecorrect:
            print(
                "sizecorrect = ",
                sizecorrect,
                "; If stimulus not full screen, assume central presentation, use correction",
            )
        else:
            print(
                "sizecorrect =",
                sizecorrect,
                "; No correction for stimulus size and screen difference-- (eg for non full screen stimuli starting at pixel 0)",
            )
        self.setStimuliPath(StimPath, StimExt, infersubpath, sizecorrect=sizecorrect)

        print("run descriptive analysis")
        self.RunDescriptiveFix(Visual=Visual)
        pass

    def RunDescriptiveFix(self, Visual=0, duration=0):
        """
        Description:  Calculate descriptive statistics for fixation data in dataset.

        Arguments:
        Visual (int): Flag indicating whether to generate visual plots (default: 0). Use 1 to show plots.
        duration (int): Flag indicating whether fixation duration data is present (default: 0). Use one if fixation duration is present.

        Returns: Mean fixation number, Number of valid fixations, inferred stim boundaries and mean and SD of fixation locations, mean Saccade amplitude, mean scanpath length.
        """

        Subjects, Stimuli = self.GetParams()
        print(
            "Data for ",
            len(self.subjects),
            "observers and ",
            len(self.stimuli),
            " stimuli.",
        )
        self.boundsX, self.boundsY = self.InferSize(Interval=99)
        self.actsize = (self.boundsX[:, 1] - self.boundsX[:, 0]) * (
            self.boundsY[:, 1] - self.boundsY[:, 0]
        )
        self.nfixations = np.zeros((self.ns, self.np))
        self.nfixations[:] = np.nan
        self.sacc_ampl = np.zeros((self.ns, self.np))
        self.len_scanpath = np.zeros((self.ns, self.np))

        MeanFixXY = np.zeros(((self.ns, self.np, 2)))
        SDFixXY = np.zeros(((self.ns, self.np, 2)))
        if duration:
            self.durations = np.zeros((self.ns, self.np))

        for cs, s in enumerate(self.subjects):
            for cp, p in enumerate(self.stimuli):
                FixTrialX, FixTrialY = self.GetFixationData(s, p)

                if len(FixTrialX) > 0:
                    self.nfixations[cs, cp] = len(FixTrialX)
                    self.sacc_ampl[cs, cp], self.len_scanpath[cs, cp] = ScanpathL(
                        FixTrialX, FixTrialY
                    )
                    MeanFixXY[cs, cp, 0], MeanFixXY[cs, cp, 1] = np.mean(
                        FixTrialX
                    ), np.mean(FixTrialY)
                    SDFixXY[cs, cp, 0], SDFixXY[cs, cp, 1] = np.std(FixTrialX), np.std(
                        FixTrialY
                    )
                    if duration:
                        self.durations[cs, cp] = np.mean(self.GetDurations(s, p))
                else:
                    MeanFixXY[cs, cp, :], SDFixXY[cs, cp, :] = np.nan, np.nan
                    self.sacc_ampl[cs, cp], self.len_scanpath[cs, cp] = np.nan, np.nan
                    if duration:
                        self.durations[cs, cp] = np.nan

        print(
            "Mean fixation number: ",
            np.round(np.nanmean(np.nanmean(self.nfixations, 1)), 2),
            " +/- ",
            np.round(np.nanstd(np.nanmean(self.nfixations, 1)), 2),
        )
        if duration:
            print(
                "Mean fixation duration: ",
                np.round(np.nanmean(np.nanmean(self.durations, 1)), 1),
                " +/- ",
                np.round(np.nanstd(np.nanmean(self.durations, 1)), 1),
                "msec",
            )
        else:
            print("fixation duration not asked for")
        print("Num of trials with zero fixations:", np.sum(self.nfixations == 0))
        print("Num valid trials ", np.sum(self.nfixations > 0))
        print(
            "Mean X location: ",
            np.round(np.mean(np.nanmean(MeanFixXY[:, :, 0], 1)), 1),
            " +/- ",
            np.round(np.std(np.nanmean(MeanFixXY[:, :, 0], 1)), 1),
            " pixels",
        )
        print(
            "Mean Y location: ",
            np.round(np.mean(np.nanmean(MeanFixXY[:, :, 1], 1)), 1),
            " +/- ",
            np.round(np.std(np.nanmean(MeanFixXY[:, :, 1], 1)), 1),
            " pixels",
        )
        print(
            "Mean saccade  amplitude: ",
            np.round(np.mean(np.nanmean(self.sacc_ampl, 1)), 1),
            " +/- ",
            np.round(np.std(np.nanmean(self.sacc_ampl, 1)), 1),
            " pixels",
        )
        print(
            "Mean scanpath  length: ",
            np.round(np.mean(np.nanmean(self.len_scanpath, 1)), 1),
            " +/- ",
            np.round(np.std(np.nanmean(self.len_scanpath, 1)), 1),
            " pixels",
        )

        if Visual:
            MeanPlot(self.np, self.nfixations, yLab="num fixations", xtickL=Stimuli)
            MeanPlot(
                self.np,
                self.len_scanpath,
                yLab=" total scanpath length (pixels)",
                xtickL=Stimuli,
            )

            HistPlot(self.nfixations, xtickL="Average Num Fixations")
        Bounds = pd.DataFrame(columns=["Stimulus"], data=Stimuli)
        Bounds["BoundX1"] = self.boundsX[:, 0]
        Bounds["BoundX2"] = self.boundsX[:, 1]
        Bounds["BoundY1"] = self.boundsY[:, 0]
        Bounds["BoundY2"] = self.boundsY[:, 1]

        try:
            import xarray as xr

            if duration:
                self.durs = xr.DataArray(
                    self.durations,
                    dims=("subjectID", "Stimulus"),
                    coords={"subjectID": Subjects, "Stimulus": Stimuli},
                )
            self.nfix = xr.DataArray(
                self.nfixations,
                dims=("subjectID", "Stimulus"),
                coords={"subjectID": Subjects, "Stimulus": Stimuli},
            )
            self.meanfix_xy = xr.DataArray(
                MeanFixXY,
                dims=("subjectID", "Stimulus", "XY"),
                coords={"subjectID": Subjects, "Stimulus": Stimuli, "XY": ["X", "Y"]},
            )
            self.sdfix_xy = xr.DataArray(
                SDFixXY,
                dims=("subjectID", "Stimulus", "XY"),
                coords={"subjectID": Subjects, "Stimulus": Stimuli, "XY": ["X", "Y"]},
            )
        except:
            print("xarray format descriptives not created, as xarray not installed")
        self.bounds = Bounds
        return Stimuli, Subjects

    def visualizeData(self):
        individual_sizes = []

        for s in self.stimuli:
            x, y, _ = self.images[s].shape
            individual_sizes.append((y, x))
        # Extract max x and max y from individual_sizes
        max_x = max(size[1] for size in individual_sizes)
        max_y = max(size[0] for size in individual_sizes)

        plt.figure(figsize=(10, 8))
        plt.scatter(self.data.mean_x, self.data.mean_y, alpha=0.5)

        for s in individual_sizes:
            rect = plt.Rectangle((0, 0), s[0], s[1], fill=False, edgecolor="red")
            plt.gca().add_patch(rect)
        # Calculate percentage of coordinates outside the rectangle

        total_points = len(self.data)

        points_outside = sum(
            (self.data.mean_x < 0)
            | (self.data.mean_x > (max_y))
            | (self.data.mean_y < 0)
            | (self.data.mean_y > (max_x))
        )
        percentage_outside = (points_outside / total_points) * 100

        plt.title(
            f"Fixation Points with Screen Boundaries\n{percentage_outside:.2f}% of points outside bounds ({points_outside} points) - max x: {max_y}, max y: {max_x}"
        )
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates
        plt.show()


#  class ends here
