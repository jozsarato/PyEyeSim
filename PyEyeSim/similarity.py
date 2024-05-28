import math
from scipy.spatial.distance import cdist
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from skimage.transform import resize

def angle_from_horizontal(row, prev_x, prev_y):
    """
    calculate the angle from the horizontal for a all given saccades
    """
    dx = row['mean_x'] - prev_x
    dy = row['mean_y'] - prev_y
    angle = math.degrees(math.atan2(dy, dx))
    
    # Adjust the angle to be between -90 and 90 degrees
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    
    return angle

def circular_distance(angle1, angle2):
    """
    Calculate the circular distance between two angles in degrees.
    """
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)

def prune_sort_corr_angles(arr_1, arr_2, kind='euclidean'):
    '''
    between two arrays of angles (two subjects) find the mean similarity for all pairs of angles
    prune the arrays to the len of the smaller
    '''
    arr_1.sort()
    arr_2.sort()

    # Find distances between each element in arr2 and arr1
    dists = np.abs(arr_2[:, None] - arr_1)
    if kind == 'simple':
        arr_1 = np.deg2rad(arr_1)
        arr_2 = np.deg2rad(arr_2)
        dists = np.abs(arr_2[:, np.newaxis] - arr_1)
        return dists.mean(), arr_1, arr_2
    # Find number of elements to keep
    n_keep = min(len(arr_1), len(arr_2))

    # Drop farthest elements from both arrays
    keep_idx1 = np.argsort(np.min(dists, axis=0))[:n_keep]
    keep_idx2 = np.argsort(np.min(dists, axis=1))[:n_keep]

    arr1_pruned = np.sort(arr_1[keep_idx1])
    arr2_pruned = np.sort(arr_2[keep_idx2])

    angles1_rad = np.deg2rad(arr1_pruned)
    angles2_rad = np.deg2rad(arr2_pruned)
 
    distances = np.array([circular_distance(a1, a2) for a1, a2 in zip(angles1_rad, angles2_rad)])
    #distances = cdist(angles1_rad, angles2_rad)
    return distances.mean(), arr1_pruned, arr2_pruned

def extract_angle_arrays(data, stims):
    angle_corrs = {}

    len_bef = len(data.data[['subjectID', 'Stimulus']].drop_duplicates())

    filtered_data = data.data[(data.data['mean_x'] <= 500) & (data.data['mean_x'] >= 0) & (data.data['mean_y'] <= 500) & (data.data['mean_y'] >= 0)]
    if len_bef != len(filtered_data[['subjectID', 'Stimulus']].drop_duplicates()):
        print(f"subjects were pruned by {len_bef - len(filtered_data[['subjectID', 'Stimulus']].drop_duplicates())} subjects")

    for stim in stims:
        angle_corrs[stim] = {}

        for s in filtered_data[filtered_data['Stimulus']==stim]['subjectID'].unique():
            sub_df = filtered_data[(filtered_data['Stimulus']==stim) & (filtered_data['subjectID']==s)]
            prev_x = sub_df['mean_x'].iloc[0]
            prev_y = sub_df['mean_y'].iloc[0]

            angles = []
            for idx, row in sub_df.iterrows():
                if idx ==0:
                    continue
                angle = angle_from_horizontal(row, prev_x, prev_y)

                angles.append(angle)
                prev_x, prev_y = row['mean_x'], row['mean_y']

            angle_corrs[stim][s] = angles
    return angle_corrs

def RSA_from_mem(data, stims):
    rdm_mem_per_img = {}
    for stim in stims:
        unique_subIDs = data.data[data.data['Stimulus']==stim]['subjectID'].unique()
        RDMs = np.zeros((len(unique_subIDs), len(unique_subIDs)))
        for idx1, subj1 in enumerate(unique_subIDs):
            for idx2, subj2 in enumerate(unique_subIDs):
                mem1 = data.data[(data.data['Stimulus']==stim) & (data.data['subjectID']==subj1)]['memory_bin'].values[0]
                mem2 = data.data[(data.data['Stimulus']==stim) & (data.data['subjectID']==subj2)]['memory_bin'].values[0]
                RDMs[idx1, idx2] = 2 if (mem1 == 1 and mem2 == 1) else 1
        rdm_mem_per_img[stim] = RDMs

    for key, matrix in rdm_mem_per_img.items():
        rdm_mem_per_img[key] = np.tril(matrix, -1)
    return rdm_mem_per_img

def RSA_from_angles(angles_per_img, kind='euclidean'):
    rdm_angles_per_img = {}
    for stim in angles_per_img:
        RDMs = np.zeros((len(angles_per_img[stim]), len(angles_per_img[stim])))
        for idx1, subj1 in enumerate(angles_per_img[stim]):
            for idx2, subj2 in enumerate(angles_per_img[stim]):

                corrs, _, _ =  prune_sort_corr_angles(np.abs(angles_per_img[stim][subj1]), np.abs(angles_per_img[stim][subj2]), kind=kind)
                RDMs[idx1, idx2] = corrs

        rdm_angles_per_img[stim] = RDMs
    for key, matrix in rdm_angles_per_img.items():
        rdm_angles_per_img[key] = np.tril(matrix, -1)
    return rdm_angles_per_img

def RSA_from_heatmaps(heatmaps_per_img):
    for stim in heatmaps_per_img:
        for subj in heatmaps_per_img[stim]:
            heatmaps_per_img[stim][subj] = heatmaps_per_img[stim][subj].flatten()
    rdm_per_img = {}
    for stim in heatmaps_per_img:
        RDMs = np.zeros((len(heatmaps_per_img[stim]), len(heatmaps_per_img[stim])))
        for idx1, subj1 in enumerate(heatmaps_per_img[stim]):
            for idx2, subj2 in enumerate(heatmaps_per_img[stim]):
                RDMs[idx1, idx2] = np.corrcoef(heatmaps_per_img[stim][subj1], heatmaps_per_img[stim][subj2])[0,1]
        rdm_per_img[stim] = RDMs

    for key, matrix in rdm_per_img.items():
        rdm_per_img[key] = np.tril(matrix, -1)
    return rdm_per_img

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
    
    return LocalSalMap.copy(), OutsideOfFrame


def extract_heatmap_arrays(data,stims, dims, resize_to=(10,10)):
    salmap_dict = {}
    penalty_dict = {}
    for stim in stims:
        salmap_dict[stim] = {}
        penalty_dict[stim] = {}
        for s in data.data[data.data['Stimulus']==stim]['subjectID'].unique():
            sub_df = data.data[(data.data['Stimulus']==stim) & (data.data['subjectID']==s)]
            heatmap, penalty = extract_focus_peaks(sub_df.mean_x,sub_df.mean_y, dims, 25)
            heatmap = resize(heatmap, resize_to)
            salmap_dict[stim][s] = heatmap
            penalty_dict[stim][s] = penalty
    return salmap_dict, penalty_dict
