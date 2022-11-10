# PyEyeSim

this is a library for eye-movement comparison


The ultimate goal of the library is to make advanced fixation map statistics (eg: entropy) and scanpath comparison  accesible.

The library also provides general descripitve statistics about eye-movements. It is intended to work with ordered fixation data. (a row for each fixation), that is accessable in a pandas dataframe.   
Additionaly, visualizations about the statistics and heatmaps are also provided.


three main functionalities:

1. Within group similarity  (for a single group of observers in a single condition)
2. Between condition similarity (for single group of observers, observing the same stimuli in two conditions)
3. Between group similarity (for two groups of observers observing the same stimuli)



if you cloned the library and are in the root folder of the library, you can install it, using: pip install -e .


for examples of using the library, see the Demo.ipynb in the Notebooks folder
