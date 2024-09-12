from PyEyeSim import EyeData
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure, binary_erosion, maximum_filter, gaussian_filter
from scipy.spatial import distance

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def extract_focus_peaks(x_coords, y_coords, dimensions, sal=25,):
    LocalSalMapRaw = np.zeros((dimensions[1], dimensions[0]))
    numberOfFixations = len(x_coords)
    # counter for fixations that are outside of the image
    OutsideOfFrame = 0
    # pointer to fixation values
    FixationsX = np.intp(np.round(x_coords))
    FixationsY = np.intp(np.round(y_coords))

    # count up fixation coordinates in a dim x dim np array
    for i in range(numberOfFixations):
        try:
            LocalSalMapRaw[FixationsY[i], FixationsX[i]] += 1
        except:
            OutsideOfFrame+=1
    LocalSalMap = gaussian_filter(LocalSalMapRaw, sal)
    
    finalHeatmap = LocalSalMap.copy()
    LocalSalMapRaw = LocalSalMapRaw.transpose()

    detectedPeaks = detect_peaks(LocalSalMap.transpose())

    finalSalMap = LocalSalMap.transpose()
    
    finalSalMap[~detectedPeaks] = 0
    
    # Flattening the array and finding the 5th highest number
    minSaliency = np.partition(finalSalMap.flatten(), -5)[-5]

    # Setting all values of arr lower than the threshold to 0.
    finalSalMap[finalSalMap < minSaliency] = 0
    focus_x, focus_y = np.where(finalSalMap > 0)
    
    return focus_x, focus_y, finalHeatmap, LocalSalMapRaw


def create_test_df(frameX, frameY, size = 1000,n_subjects = 3, groups = 1, nr_imgs = 1):
    """Creates a test DataFrame with randomly generated fixation data.

    Generates a DataFrame with the specified number of rows, containing random 
    x/y coordinates for fixations on a specified test image, along with random
    durations. Useful for testing fixation processing functions.

    Args:
    size: Number of rows to generate.
    test_img: Name of test image file.
    
    Returns:
    test_img: Test image filename.
    test_df: DataFrame containing generated fixation data.
    """
    test_imgs = ['test_{}.jpg'.format(i+1) for i in range(nr_imgs)]
    labels = ['tr_{}'.format(i + 1) for i in range(n_subjects)]
    test_df = pd.DataFrame({
        'RECORDING_SESSION_LABEL': np.random.choice(labels, size),
        'image_1': np.random.choice(test_imgs, size), 
        'CURRENT_FIX_X': np.random.randint(0, frameX, size),  # generating random integers for X
        'CURRENT_FIX_Y': np.random.randint(0, frameY, size),  # generating random integers for Y
        'CURRENT_FIX_DURATION': np.random.randint(0, 501, size)  # generating random integers for duration
    })
    if groups > 1:
        labels_unique = test_df['RECORDING_SESSION_LABEL'].unique()
        group_dict = {label: np.random.randint(1, groups+1) for label in labels_unique}
        test_df['group'] = test_df['RECORDING_SESSION_LABEL'].map(group_dict)

    print("test frame created")
    return test_imgs, test_df

class TestGeneralFunctions(unittest.TestCase):

    #heatmap
    def test_heatmap(self):
        fixation_rows = 10
        sizeX,sizeY = 5,5
        N = 3
        test_img, test_df = create_test_df( frameX=sizeX, frameY=sizeY, size=fixation_rows, n_subjects = N)

        test_eye_data = EyeData(test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        xs, ys, heatmap, salmap = extract_focus_peaks(test_eye_data.data.mean_x, test_eye_data.data.mean_y, (sizeX, sizeY))

        heatmap = heatmap/N #normalizing heatmap as function below uses mean of all fixations and not sum

        testHM = test_eye_data.Heatmap(test_img, CutArea=0)

        self.assertTrue(np.allclose(testHM, heatmap, rtol=1e-03, atol=1e-04))             
     #fixCount#
        
    def test_fixCountCalc(self):
        fixation_rows = 1000
        sizeX,sizeY = 3840,2160
        test_img, test_df = create_test_df(sizeX, sizeY, size=fixation_rows)

        test_eye_data = EyeData( test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        test_res = test_eye_data.FixCountCalc(test_img, CutAct=0)

        self.assertEqual(np.sum(test_res), fixation_rows)
class TestCompareFunctions(unittest.TestCase):

    def test_heatmap_compare_stim(self):
        fixation_rows = 1000
        sizeX,sizeY = 3840,2160
        test_img, test_df = create_test_df(sizeX, sizeY, size=fixation_rows, n_subjects=40, groups=2, nr_imgs=2)
 
        test_eye_data = EyeData(test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        test_eye_data.CompareStimHeatmap(test_img,SD=40,alpha=.6)
    
    def test_heatmap_compare_group(self):
        fixation_rows = 1000
        sizeX,sizeY = 3840,2160
        test_img, test_df = create_test_df(sizeX, sizeY, size=fixation_rows, n_subjects=40, groups=2)

        test_eye_data = EyeData( test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        test_eye_data.CompareGroupsHeatmap(test_img[0],'group',SD=40,alpha=.6)

    def test_heatmap_compare_group_within(self):
        fixation_rows = 1000
        sizeX,sizeY = 3840,2160
        test_img, test_df = create_test_df(sizeX, sizeY, size=fixation_rows, n_subjects=40, groups=2)

        test_eye_data = EyeData( test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        test_eye_data.CompareWithinGroupsFix('group')   

    def test_group_grid(self):
        fixation_rows = 1000
        sizeX,sizeY = 3840,2160
        test_img, test_df = create_test_df(sizeX, sizeY, size=fixation_rows, n_subjects=40, groups=2)

        test_eye_data = EyeData( test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        test_eye_data.CompareGroupsGridFix(test_img[0], 'group')   

    def test_group_fix(self):
        fixation_rows = 1000
        sizeX,sizeY = 3840,2160
        test_img, test_df = create_test_df(sizeX, sizeY, size=fixation_rows, n_subjects=40, groups=2)

        test_eye_data = EyeData(test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')

        test_eye_data.CompareGroupsFix('group')   


if __name__ == '__main__':
    unittest.main()
