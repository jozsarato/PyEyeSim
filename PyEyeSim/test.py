from PyEyeSim import EyeData
import unittest
import pandas as pd
class TestFixCountCalc(unittest.TestCase):
    def test_case_1(self):
        test_df = pd.read_csv('/Users/juliusduin/Desktop/uni_work/PyEyeSim/test_data/fixCountTest.csv')
        stimuli = test_df['image_1'].unique()
        sizeX,sizeY=3840,2160
        test_eye_data = EyeData('test_1', 'between', test_df, sizeX, sizeY)
        test_eye_data.DataInfo(Stimulus='image_1',subjectID='RECORDING_SESSION_LABEL',mean_x='CURRENT_FIX_X',mean_y='CURRENT_FIX_Y',FixDuration='CURRENT_FIX_DURATION')
        test_eye_data.RunDescriptiveFix()
        test_eye_data.FixCountCalc(stimuli[0])

if __name__ == '__main__':
    unittest.main()
