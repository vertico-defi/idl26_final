# IDL Final Assignment: Deep Learning Pipeline Repair

Author: Richard Van Winkle, Matrikelnummer: 3214183

# Environment Setup

Create a conda environment with the following linux command

`conda create -n deepLearning python=3.10`

press y to finish the install. Activate the environment with the following command:

`conda activate deepLearning`

Make sure to see `(deepLearning)` on the left side of your command line. Then install the NumPy and PyTorch packages with the following commands:

`pip install numpy`
`pip install torch`


# Directory Setup

Before running the code, add the audit file, so that we can collect the initial errors. In the root directory run the following command:

`touch AUDIT_LOG.md`

Looking at the `train.py` file, it will try to read from a config.json file, add that to the directory with the following command:

`touch config.json`

Before doing any major work on this dataset, we will have to find out how many channels the images have for future processing. Use this oppurtunity to also sample/view some of the images, and eventually to perform a sanity check on how the models are working. So create a python file to help you view the images and print some data on them. This file is neccessary but not part of the graded work, we'll have chatGPT mock something up for us, the command to create the file is:

`touch view_pt_samples.py`