# Final Benchmark Report

This report summarizes the final benchmark results after reconstructing and stabilizing the recovered deep-learning pipeline for the IDL 2026 final assignment. The recovered codebase originally contained runtime errors, tensor-shape errors, silent training bugs, hardcoded model dimensions, and missing evaluation infrastructure.

The final pipeline can train and test all three required architectures:

- AlexNet
- VGG16
- ResNet18

across the available datasets:

- `cells`
- `chest`
- `lesions`
- `orgs`
- `organs`

The evaluation was done through the repaired training pipeline and the added test pipeline. For each dataset/model pair, the best validation checkpoint was saved during training and then evaluated on the held-out test split. The final benchmark table includes accuracy, macro precision, macro recall, macro F1-score, training runtime, peak training memory, inference latency per sample, and peak inference memory.

## 1. Benchmark Setup

All benchmark runs used the same general training configuration:

| Setting | Value |
|---------|------:|
| Optimizer | Adam |
| Learning rate | `0.001` |
| Epochs | `25` |
| Batch size | `8` |
| Checkpoint selection | Best validation accuracy |
| Test evaluation | Held-out test split |
| Metrics | Accuracy, macro precision, macro recall, macro F1 |
| Efficiency metrics | Training runtime, training peak memory, inference latency, inference peak memory |

The dataset-specific configuration values were:

| Dataset | Channels | Number of classes |
|---------|---------:|------------------:|
| `cells` |    3     |        8          |
| `chest` |    1     |        2          |
| `lesions` |  3     |        7          |
| `orgs` |     1     |       11          |
| `organs` |   1     |       11          |

The benchmark was run for every dataset/model permutation:

```text
5 datasets x 3 models = 15 benchmark runs
```

The test script appends the final results to:

```text
results/benchmark_results.txt
```

## 2. Full Benchmark Results

| Dataset | Model | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Train Runtime (s) | Train Peak Mem (MB) | Inference (ms/sample) | Inference Peak Mem (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cells | AlexNet | 96.64 | 96.72 | 96.03 | 96.33 | 257.17 | 89.27 | 0.2122 | 47.33 |
| cells | VGG16 | 95.09 | 95.15 | 94.65 | 94.81 | 1447.00 | 327.02 | 1.1712 | 134.59 |
| cells | ResNet18 | 96.99 | 97.06 | 96.87 | 96.92 | 1951.35 | 354.11 | 1.5868 | 135.60 |
| chest | AlexNet | 86.70 | 90.72 | 82.44 | 84.50 | 96.74 | 88.88 | 0.3459 | 46.99 |
| chest | VGG16 | 88.14 | 91.13 | 84.53 | 86.42 | 551.60 | 326.71 | 1.2967 | 134.31 |
| chest | ResNet18 | 84.13 | 89.00 | 79.10 | 81.13 | 747.41 | 353.80 | 1.6979 | 135.32 |
| lesions | AlexNet | 74.76 | 48.40 | 42.30 | 44.05 | 151.35 | 89.26 | 0.2321 | 47.32 |
| lesions | VGG16 | 71.57 | 34.07 | 32.94 | 32.65 | 846.06 | 327.01 | 1.1912 | 134.59 |
| lesions | ResNet18 | 73.97 | 51.91 | 46.86 | 45.37 | 1143.63 | 354.10 | 1.5959 | 135.59 |
| orgs | AlexNet | 88.15 | 86.37 | 86.96 | 86.48 | 283.18 | 89.02 | 0.1752 | 47.06 |
| orgs | VGG16 | 88.89 | 87.42 | 86.92 | 87.01 | 1624.78 | 326.78 | 1.1687 | 134.34 |
| orgs | ResNet18 | 92.33 | 91.42 | 91.14 | 91.06 | 2235.33 | 353.87 | 1.5595 | 135.35 |
| organs | AlexNet | 53.50 | 46.53 | 46.69 | 44.68 | 9.80 | 89.02 | 0.7314 | 47.06 |
| organs | VGG16 | 39.50 | 19.02 | 28.49 | 21.35 | 53.80 | 326.78 | 1.6792 | 134.34 |
| organs | ResNet18 | 65.00 | 62.96 | 59.42 | 57.97 | 72.34 | 353.87 | 2.0616 | 135.35 |

## 3. Accuracy Target Check

The assignment gave the following minimum expected test accuracies:

| Dataset | Required Test Accuracy (%) | Best Model | Best Observed Test Accuracy (%) | Status |
|---|---:|---|---:|---|
| cells | 90 | ResNet18 | 96.99 | PASS |
| chest | 87 | VGG16 | 88.14 | PASS |
| lesions | 67 | AlexNet | 74.76 | PASS |
| orgs | 83 | ResNet18 | 92.33 | PASS |
| organs | 40 | ResNet18 | 65.00 | PASS |

All required dataset targets were reached by at least one architecture.

## 4. Architecture Recommendations by Dataset

### cells

The best accuracy on cells was achieved by ResNet18:

| Model | Test Acc (%) | Macro F1 (%) | Train Runtime (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|
| AlexNet | 96.64 | 96.33 | 257.17 | 0.2122 |
| VGG16 | 95.09 | 94.81 | 1447.00 | 1.1712 |
| ResNet18 | 96.99 | 96.92 | 1951.35 | 1.5868 |

Recommendation: AlexNet is the practical recommendation for cells if efficiency matters, because it is only 0.35 percentage points below ResNet18 in test accuracy but is much faster and uses much less memory. If the only criterion is maximum accuracy, ResNet18 is best.

### chest

The best accuracy on chest was achieved by VGG16:

| Model | Test Acc (%) | Macro F1 (%) | Train Runtime (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|
| AlexNet | 86.70 | 84.50 | 96.74 | 0.3459 |
| VGG16 | 88.14 | 86.42 | 551.60 | 1.2967 |
| ResNet18 | 84.13 | 81.13 | 747.41 | 1.6979 |

Recommendation: VGG16 is the final recommendation for chest, because it is the only model that crossed the 87% test accuracy requirement. AlexNet is close and much cheaper, but it stayed slightly below the required threshold.

### lesions

The best accuracy on lesions was achieved by AlexNet:

| Model | Test Acc (%) | Macro F1 (%) | Train Runtime (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|
| AlexNet | 74.76 | 44.05 | 151.35 | 0.2321 |
| VGG16 | 71.57 | 32.65 | 846.06 | 1.1912 |
| ResNet18 | 73.97 | 45.37 | 1143.63 | 1.5959 |

Recommendation: AlexNet is the final recommendation for lesions. It achieved the highest test accuracy and the lowest computational cost. The macro F1 scores are much lower than the accuracy scores for all three models, which suggests class imbalance or uneven class difficulty. Because of that, accuracy alone should not be over-interpreted for this dataset.

### orgs

The best accuracy on orgs was achieved by ResNet18:

| Model | Test Acc (%) | Macro F1 (%) | Train Runtime (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|
| AlexNet | 88.15 | 86.48 | 283.18 | 0.1752 |
| VGG16 | 88.89 | 87.01 | 1624.78 | 1.1687 |
| ResNet18 | 92.33 | 91.06 | 2235.33 | 1.5595 |

Recommendation: ResNet18 is the best accuracy recommendation for orgs. However, AlexNet is the green-efficiency recommendation, because it still passes the 83% target while using much less runtime and memory.

### organs

The best accuracy on organs was achieved by ResNet18:

| Model | Test Acc (%) | Macro F1 (%) | Train Runtime (s) | Inference (ms/sample) |
|---|---:|---:|---:|---:|
| AlexNet | 53.50 | 44.68 | 9.80 | 0.7314 |
| VGG16 | 39.50 | 21.35 | 53.80 | 1.6792 |
| ResNet18 | 65.00 | 57.97 | 72.34 | 2.0616 |

Recommendation: ResNet18 is the final recommendation for organs, because it has the strongest accuracy and macro F1. AlexNet also clears the 40% target and is much cheaper, but the accuracy gap is large enough that I would still recommend ResNet18 for this scarce-data case.

## 5. Green Initiative Analysis

The green initiative requires comparing model complexity against runtime, memory, and predictive performance. The main result from the benchmark is that AlexNet is usually the most efficient model, while ResNet18 is usually the strongest accuracy model. VGG16 is often expensive without giving the strongest result, except for chest, where it is the only model that reaches the required accuracy threshold.

### 5.1 AlexNet as the Efficient Architecture

For cells, AlexNet gives almost the same accuracy as ResNet18:

| Dataset | AlexNet Acc (%) | ResNet18 Acc (%) | Accuracy Gap | AlexNet Runtime | ResNet18 Runtime |
|---|---:|---:|---:|---:|---:|
| cells | 96.64 | 96.99 | -0.35 | 257.17 s | 1951.35 s |

On cells, AlexNet is about 7.6 times faster in training than ResNet18 while losing only 0.35 percentage points in test accuracy. Its training memory is also much lower:

| Model | Train Peak Memory |
|---|---:|
| AlexNet | 89.27 MB |
| ResNet18 | 354.11 MB |

For lesions, AlexNet is both more accurate and more efficient than ResNet18:

| Dataset | AlexNet Acc (%) | ResNet18 Acc (%) | AlexNet Runtime | ResNet18 Runtime |
|---|---:|---:|---:|---:|
| lesions | 74.76 | 73.97 | 151.35 s | 1143.63 s |

For orgs, ResNet18 has the best accuracy, but AlexNet still passes the required target:

| Dataset | AlexNet Acc (%) | ResNet18 Acc (%) | AlexNet Runtime | ResNet18 Runtime |
|---|---:|---:|---:|---:|
| orgs | 88.15 | 92.33 | 283.18 s | 2235.33 s |

So for orgs, ResNet18 is the best accuracy model, but AlexNet is a reasonable low-compute deployment candidate when the main priority is speed and memory reduction.

### 5.2 VGG16

VGG16 is not the best green model overall. It usually uses much more runtime and memory than AlexNet, and it does not consistently beat ResNet18 in accuracy. Its main successful case is chest, where it is the only architecture that meets the required 87% test accuracy.

For chest:

| Model | Test Accuracy |
|---|---:|
| AlexNet | 86.70% |
| VGG16 | 88.14% |
| ResNet18 | 84.13% |

So VGG16 is justified for chest, but I would not recommend it as the general low-compute model.

### 5.3 Final Green Recommendation

The most practical green recommendation is:

| Dataset | Green Recommendation | Reason |
|---|---|---|
| cells | AlexNet | Accuracy is nearly equal to ResNet18, with much lower runtime and memory. |
| chest | VGG16 | It is the only model that meets the required accuracy target. |
| lesions | AlexNet | Best accuracy and lowest compute. |
| orgs | AlexNet if efficiency is preferred; ResNet18 if accuracy is preferred | AlexNet passes target cheaply; ResNet18 is more accurate. |
| organs | ResNet18 | Scarce-data accuracy is more important here, and ResNet18 performs best. |

Overall, AlexNet is the strongest green candidate in this benchmark. It is not the best model for every dataset, but when it reaches the target accuracy, it often does so at a fraction of the runtime and memory cost of the heavier models.

## 6. Organs Scarce-Data Analysis

The organs dataset is different from the other datasets because it is the small, newly introduced dataset. The goal was to bring it into the restored pipeline and check whether the recovered models can learn useful structure from limited data.

The results were:

| Model | Test Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) |
|---|---:|---:|---:|---:|
| AlexNet | 53.50 | 46.53 | 46.69 | 44.68 |
| VGG16 | 39.50 | 19.02 | 28.49 | 21.35 |
| ResNet18 | 65.00 | 62.96 | 59.42 | 57.97 |

ResNet18 performed best on organs, reaching 65.00% test accuracy. This is above the 40% target and is also clearly stronger than the other two models in macro F1. AlexNet also crossed the 40% threshold and was much faster, but ResNet18 gave better classification quality.

The main practical conclusion is that the scarce-data setting benefits from the stronger residual architecture. At the same time, the result should not be interpreted as fully solved clinical performance. The dataset is small, and the final macro F1 of 57.97% shows that there is still room for improvement.

Future work for organs should include:

- Transfer learning from the larger `orgs` dataset.
- Stronger data augmentation.
- Repeated runs with different random seeds.
- Stratified validation splitting.
- Class-wise error analysis.
- Collection of more labeled samples.

In this final benchmark, I report the scratch-training baseline across all three architectures. The results show that the restored pipeline can train and test the organs dataset, and that ResNet18 is the best current choice for this low-sample setting.

## 7. Notes on Model Behavior

Several patterns are visible in the final results.

First, after removing the excessive dropout override, AlexNet became a strong baseline. This is important because AlexNet and VGG16 both use dropout, while ResNet18 does not use dropout in this implementation. A hardcoded dropout value of 0.99 heavily damaged the models that used dropout. Removing that override allowed AlexNet and VGG16 to train properly.

Second, accuracy and macro F1 are not always close. The clearest example is lesions, where AlexNet reached 74.76% accuracy but only 44.05% macro F1. That means the model is probably doing much better on common classes than on rare classes. For that reason, I included macro precision, macro recall, and macro F1 in the benchmark and did not rely only on accuracy.

Third, validation accuracy and test accuracy do not always move together. This was especially visible during chest experiments. The final recommendation for chest is based on test accuracy, not validation accuracy, because the assignment target is based on test performance.

## 8. Limitations

The final pipeline is functional and reaches the required accuracy targets, but there are still limitations.

- The validation split is deterministic rather than randomized or stratified. I fixed the data leakage issue first by removing overlap between training and validation data. A stratified validation split would be a good future improvement.
- Each benchmark row is based on one run. Because neural network training has stochastic elements, repeated runs with different seeds would give a more robust estimate.
- No external pretrained weights were used. The organs section therefore reports a scratch-training benchmark, not a full pretrained transfer-learning solution.
- The test set was used only for final evaluation, but there is no additional external validation dataset beyond the provided split.
- The macro F1 scores show that some datasets, especially lesions, likely have class imbalance or uneven class difficulty. More class-wise analysis would be useful before clinical use.

## 9. Final Conclusion

The restored pipeline now trains and tests all required architectures across the target datasets. The final benchmark results show that every dataset reaches the required test accuracy threshold using at least one model:

| Dataset | Passing Model | Test Acc (%) |
|---|---|---:|
| cells | ResNet18 | 96.99 |
| chest | VGG16 | 88.14 |
| lesions | AlexNet | 74.76 |
| orgs | ResNet18 | 92.33 |
| organs | ResNet18 | 65.00 |

The best overall accuracy model is usually ResNet18, especially for orgs and organs. The best green-efficiency model is usually AlexNet, especially for cells, lesions, and as an efficient passing option for orgs. VGG16 is specifically useful for chest, where it is the only model in this benchmark to exceed the required 87% test accuracy.

The final recommendation is therefore not a single architecture for all cases, but a dataset-specific deployment choice:

| Dataset | Final Recommended Model |
|---|---|
| cells | AlexNet for efficiency; ResNet18 for maximum accuracy |
| chest | VGG16 |
| lesions | AlexNet |
| orgs | ResNet18 for maximum accuracy; AlexNet for green deployment |
| organs | ResNet18 |

This gives a corrected, benchmarked, and configurable restored pipeline that can support training and prediction across all required datasets and model architectures.
