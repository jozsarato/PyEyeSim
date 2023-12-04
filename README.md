# PyEyeSim

this is a Python library for eye-movement analysis, visualization and comparison, with a focus on scanpath comparison.

The ultimate goal of the library is to make advanced fixation map statistics (eg: entropy) and scanpath comparison  accesible (hidden markov model based, and saccade direction based).

The library also provides general descripitve statistics about eye-movements. It is intended to work with ordered fixation data. (a row for each fixation), that is loaded into a pandas dataframe.   

Additionaly, easy visualizations about the statistics (overall stats, stimulus based stats, within trial progrression) and heatmaps are also provided. 

three main scanpath similarity functionalities:

1. Within group similarity  (for a single group of observers in a single condition)
2. Between condition similarity (for single group of observers, observing the same stimuli in two conditions)
3. Between group similarity (for two groups of observers observing the same stimuli)


The library started to develop for use in art perception studies, therefore, there is an emphasis on stimulus based eye-movement comparison.


#### Installation:
if you cloned the library and are in the root folder of the library using the terminal (mac) or anaconda prompt (windows), you can install it, using the command: 
pip install -e .


#### Demo:
for examples of using the library, see the PyEyeDemoBasic.ipynb in the Notebooks folder


#### Dependencies:
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
xarray
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

##### for full funcionality
  hmmlearn
