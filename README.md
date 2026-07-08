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

Also, for the oral defense, let's take some notes and record the outputs of debugging and auditing using an debug_notes.md file type the following command in the termial:

`touch debug_notes.md`

Looking at the `train.py` file, it will try to read from a config.json file, add that to the directory with the following command:

`touch config.json`

Also there will need to be a `test.py` file so that we can run the model on the test data, run the following command in the terminal inside of the code directory:

`touch test.py`

## Configuration

Training is controlled through config.json in the repository root.

The file should use the following structure:

{
  "DATA": "lesions",
  "DATA_PATH": "./data",
  "MODEL": "ResNet18",
  "CHANNELS": 3,
  "NUM_CLASSES": 7,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 20,
  "BATCH_SIZE": 8
}

## Available dataset settings

When changing datasets, update DATA, CHANNELS, and NUM_CLASSES together.

Dataset | DATA value | CHANNELS	| NUM_CLASSES
Cells	|   "cells"	 |    3	    |    8
Chest	| "chest"	 |    1	    |    2
Lesions	| "lesions"	 |    3	    |    7
Organs	| "organs"	 |    1	    |   11
Orgs	| "orgs"	 |    1	    |   11

## Available model settings

Change the MODEL value to select the architecture:

Model	MODEL value
AlexNet	"AlexNet"
VGG16	"VGG16"
ResNet18	"ResNet18"
Example configurations
Lesions with ResNet18
{
  "DATA": "lesions",
  "DATA_PATH": "./data",
  "MODEL": "ResNet18",
  "CHANNELS": 3,
  "NUM_CLASSES": 7,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 20,
  "BATCH_SIZE": 8
}
Chest with ResNet18
{
  "DATA": "chest",
  "DATA_PATH": "./data",
  "MODEL": "ResNet18",
  "CHANNELS": 1,
  "NUM_CLASSES": 2,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 20,
  "BATCH_SIZE": 8
}
Cells with VGG16
{
  "DATA": "cells",
  "DATA_PATH": "./data",
  "MODEL": "VGG16",
  "CHANNELS": 3,
  "NUM_CLASSES": 8,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 20,
  "BATCH_SIZE": 8
}

Run training with:

python3 code/train.py