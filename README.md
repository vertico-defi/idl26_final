# IDL Final Assignment: Deep Learning Pipeline Repair

Author: Richard Van Winkle
Matrikelnummer: 3214183

## Overview

This repository contains the repaired pipeline for the IDL 2026 final assignment. The code trains and tests AlexNet, VGG16, and ResNet18 on the provided medical image datasets.

The main files are:

| File             | Purpose                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| `code/data.py`   | Loads the selected `.pt` dataset and creates train, validation, and test loaders |
| `code/models.py` | Contains AlexNet, VGG16, and ResNet18                                            |
| `code/fit.py`    | Contains the training and validation loop                                        |
| `code/train.py`  | Trains the selected model from `config.json` and saves the best checkpoint       |
| `code/test.py`   | Loads the saved checkpoint and evaluates the model on the test set               |
| `config.json`    | Controls dataset, model, and training parameters                                 |
| `AUDIT_LOG.md`   | Documents the bugs and fixes                                                     |
| `REPORT.md`      | Contains the benchmark results and final analysis                                |

## Environment Setup

Create and activate the environment:

```bash
conda create -n dlFinal_gpu python=3.10 -y
conda activate dlFinal_gpu
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn
```

Install PyTorch. I used a CUDA wheel compatible with the NVIDIA GTX 970:

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

## Data Setup

The data files should be placed in the root-level `data/` directory.

Expected files:

```text
data/cells.pt
data/chest.pt
data/lesions.pt
data/orgs.pt
data/organs.pt
```

The data files are not committed to the repository.

## Configuration

Training is controlled through `config.json` in the repository root.

Example:

```json
{
  "DATA": "lesions",
  "DATA_PATH": "./data",
  "MODEL": "ResNet18",
  "CHANNELS": 3,
  "NUM_CLASSES": 7,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 25,
  "BATCH_SIZE": 8
}
```

When changing datasets, update `DATA`, `CHANNELS`, and `NUM_CLASSES` together.

| Dataset | `DATA` value | `CHANNELS` | `NUM_CLASSES` |
| ------- | -----------: | ---------: | ------------: |
| Cells   |    `"cells"` |          3 |             8 |
| Chest   |    `"chest"` |          1 |             2 |
| Lesions |  `"lesions"` |          3 |             7 |
| Orgs    |     `"orgs"` |          1 |            11 |
| Organs  |   `"organs"` |          1 |            11 |

Available model values:

| Model    | `MODEL` value |
| -------- | ------------: |
| AlexNet  |   `"AlexNet"` |
| VGG16    |     `"VGG16"` |
| ResNet18 |  `"ResNet18"` |

## Train a Model

After setting `config.json`, run:

```bash
python3 code/train.py
```

The training script saves the best validation checkpoint to:

```text
checkpoints/<DATA>_<MODEL>.pt
```

Example:

```text
checkpoints/lesions_ResNet18.pt
```

Checkpoints are generated locally and are not committed.

## Test a Model

After training, run:

```bash
python3 code/test.py
```

The test script loads the matching checkpoint and evaluates the test split.

It prints:

* test accuracy
* macro precision
* macro recall
* macro F1-score
* training runtime
* training peak memory
* inference latency per sample
* inference peak memory

It also appends the result to:

```text
results/benchmark_results.txt
```

## Benchmark Results

The final benchmark table is stored in:

```text
results/benchmark_results.txt
```

The full analysis and model recommendations are written in:

```text
REPORT.md
```

The final benchmark shows that the required test accuracy targets are reached by at least one model for each dataset:

| Dataset | Passing model | Test accuracy |
| ------- | ------------- | ------------: |
| Cells   | ResNet18      |        96.99% |
| Chest   | VGG16         |        88.14% |
| Lesions | AlexNet       |        74.76% |
| Orgs    | ResNet18      |        92.33% |
| Organs  | ResNet18      |        65.00% |

## Notes

`AUDIT_LOG.md` contains the technical audit table with the discovered issues, root causes, fixes, and commit hashes.

`REPORT.md` contains the final benchmark comparison, green-efficiency analysis, and dataset/model recommendations.
