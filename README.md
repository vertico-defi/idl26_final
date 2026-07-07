# IDL Final Assignment: Deep Learning Pipeline Repair

Author: Richard Van Winkle, Matrikelnummer: 3214183

## Environment Setup

Create a conda environment with the following linux command

`conda create -n dlFinal_gpu python=3.10 -y`
`conda activate dlFinal_gpu`

`python -m pip install --upgrade pip`
`python -m pip install numpy`

Use wheel download of pytorch with cuda that is compatible with a nvidia GTX 970 graphics card.
`python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118`


## Directory Setup

Before running the code, add the audit file, so that we can collect the initial errors. In the root directory run the following command:

`touch AUDIT_LOG.md`

Looking at the `train.py` file, it will try to read from a config.json file, add that to the directory with the following command:

`touch config.json`

