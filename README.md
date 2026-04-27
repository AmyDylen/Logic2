# LOGIC2 Usage Guide

## Overview

LOGIC2 is a multi-protocol (TLS, HTTP, TCP) malware traffic detection system. This toolkit contains three core scripts for standard training and evaluation, Leave-One-Family-Out (LOFO) cross-family generalization testing, and few-shot learning experiments, respectively.

**Download link for the MTA-C2 dataset**: `MTA-C2.zip`

------

## Script 1: 2.logic2-multi.py

### Functions

Standard training and evaluation script, supporting multi-protocol joint detection.

### Data Filtering Modes

- `multi_only`: Retain only multi-flow clusters (flow count > 1)
- `multi_and_long_single`: Retain multi-flow clusters + long single flows
- `all`: Retain all flow clusters

### Usage

bash




```
python 2.logic2-multi.py --c2-data <C2 data path> --benign-data <benign data path> [options]
```

### Parameter Description




|       Parameter        | Type  |  Default   |                   Description                   |
| :--------------------: | :---: | :--------: | :---------------------------------------------: |
|      `--c2-data`       |  str  |  Required  |            File path of C2 datasets             |
|    `--benign-data`     |  str  |  Required  |          File path of benign datasets           |
|     `--val-split`      | float |    0.2     |          Proportion of validation set           |
|        `--seed`        |  int  |     42     |         Random seed for reproducibility         |
|        `--mode`        |  str  |    eval    |           Running mode: train / eval            |
|       `--epochs`       |  int  |     20     |               Fine-tuning epochs                |
|   `--warmup-epochs`    |  int  |     5      |             Warm-up training epochs             |
|     `--batch-size`     |  int  |     16     |                   Batch size                    |
|         `--lr`         | float |    1e-4    |                  Learning rate                  |
|    `--filter-mode`     |  str  | multi_only |               Data filtering mode               |
| `--benign-sample-size` |  int  |    None    | Quantity limit of benign samples (per protocol) |

### Examples

**Evaluation only (with pre-trained model):**

bash




```
python 2.logic2-multi.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV --mode eval
```

**Training + Evaluation:**

bash



```
python 2.logic2-multi.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV --mode train --epochs 20
```

------

## Script 2: 2.logic2-multi-lofo.py

### Functions

Leave-One-Family-Out (LOFO) cross-validation, designed to evaluate the cross-family generalization capability of the model.

### Core Principle

Leave one malware family as the test set in each round, train the model with all remaining families, so as to validate the detection performance against unknown malware families.

### Usage

bash






```
python 2.logic2-multi-lofo.py --c2-data <C2 data path> --benign-data <benign data path> [options]
```

### Parameter Description



|     Parameter     | Type |       Default        |                         Description                         |
| :---------------: | :--: | :------------------: | :---------------------------------------------------------: |
|    `--c2-data`    | str  |      ./MTA-csv       | C2 dataset path (containing subfolders of malware families) |
|  `--benign-data`  | str  |      ./CTU-csv       |                     Benign dataset path                     |
|  `--output-dir`   | str  | ./lofo_multi_results |          Output directory for experimental results          |
| `--skip-families` | str  |          ""          |         Skipped malware families (comma-separated)          |
|  `--filter-mode`  | str  |      multi_only      |                     Data filtering mode                     |

### Example

bash






```
python 2.logic2-multi-lofo.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV --output-dir ./lofo_results
```

------

## Script 3: 2.logic2-multi-fewshot.py

### Functions

Few-shot learning experiment script, used to evaluate detection performance with extremely limited labeled data.

### K-shot Settings

- K ∈ {1, 3, 5, 10}
- Exactly K flow clusters are randomly sampled from each qualified C2 family

### Usage

bash



```
python 2.logic2-multi-fewshot.py --c2-data <C2 data path> --benign-data <benign data path> [options]
```

### Parameter Description



|       Parameter        | Type |      Default      |                Description                |
| :--------------------: | :--: | :---------------: | :---------------------------------------: |
|      `--c2-data`       | str  |     ./MTA-csv     |              C2 dataset path              |
|    `--benign-data`     | str  |     ./CTU-csv     |            Benign dataset path            |
|       `--k-shot`       | int  |         5         |          K-shot value (1/3/5/10)          |
| `--benign-sample-size` | int  |        500        |     Quantity limit of benign samples      |
|     `--output-dir`     | str  | ./fewshot_results | Output directory for experimental results |
|    `--filter-mode`     | str  |    multi_only     |            Data filtering mode            |
|      `--protocol`      | str  |        ALL        |  Target protocol: ALL / TLS / HTTP / TCP  |
|      `--verbose`       | bool |       False       |      Enable detailed evaluation logs      |

### Examples

**5-shot Experiment:**

bash




```
python 2.logic2-multi-fewshot.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV --k-shot 5
```

**1-shot Experiment:**

bash



运行







```
python 2.logic2-multi-fewshot.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV --k-shot 1
```

**TLS-only Test:**

bash




```
python 2.logic2-multi-fewshot.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV --k-shot 5 --protocol TLS
```

------

## Data Preprocessing Tools

Relevant preprocessing scripts are named in the format of `1.x.xxxx.py`.

------

## Evaluation Metrics Explanation

All three scripts output the following evaluation metrics:


|  Metric   |     Description      |
| :-------: | :------------------: |
| Accuracy  |   Overall Accuracy   |
| Precision |    Precision Rate    |
|  Recall   |     Recall Rate      |
| F1-Score  |       F1 Score       |
|    FPR    | False Positive Rate  |
|  ROC-AUC  | Area Under ROC Curve |
