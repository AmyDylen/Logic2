# LOGIC2 使用说明

## 概述

LOGIC2 是一个支持多协议（TLS、HTTP、TCP）的恶意软件流量检测系统。本工具包包含三个核心脚本，分别用于标准训练评估、跨家族泛化测试（LOFO）和少样本学习（Few-shot）。

**MTA-C2数据集下载链接见**: `MTA-C2.zip`


## 脚本一：2.logic2-multi.py

### 功能
标准训练和评估脚本，支持多协议联合检测。

### 数据筛选模式
- `multi_only`: 只保留多流簇（流数量 > 1）
- `multi_and_long_single`: 保留多流簇 + 单流长流
- `all`: 保留所有流簇

### 使用方法

```bash
python 2.logic2-multi.py --c2-data <C2数据路径> --benign-data <良性数据路径> [选项]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--c2-data` | str | 必需 | C2数据文件夹路径 |
| `--benign-data` | str | 必需 | 良性数据文件夹路径 |
| `--val-split` | float | 0.2 | 验证集比例 |
| `--seed` | int | 42 | 随机种子 |
| `--mode` | str | eval | 运行模式：train/eval |
| `--epochs` | int | 20 | 微调轮数 |
| `--warmup-epochs` | int | 5 | 预热轮数 |
| `--batch-size` | int | 16 | 批次大小 |
| `--lr` | float | 1e-4 | 学习率 |
| `--filter-mode` | str | multi_only | 数据筛选模式 |
| `--benign-sample-size` | int | None | 良性样本数量限制（每个协议） |

### 示例

**仅评估（使用预训练模型）:**
```bash
python 2.logic2-multi.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV-new --mode eval
```

**训练 + 评估:**
```bash
python 2.logic2-multi.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV-new --mode train --epochs 20
```


---

## 脚本二：2.logic2-multi-lofo.py

### 功能
Leave-One-Family-Out (LOFO) 交叉验证，用于评估模型的跨家族泛化能力。

### 核心思路
每次留出一个恶意软件家族作为测试集，用其他家族训练，检验模型对未知家族的检测能力。

### 使用方法

```bash
python 2.logic2-multi-lofo.py --c2-data <C2数据路径> --benign-data <良性数据路径> [选项]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--c2-data` | str | ./MTA-csv | C2数据路径（包含家族子文件夹） |
| `--benign-data` | str | ./CTU-csv | 良性数据路径 |
| `--output-dir` | str | ./lofo_multi_results | 结果输出目录 |
| `--skip-families` | str | "" | 跳过的家族名称（逗号分隔） |
| `--filter-mode` | str | multi_only | 数据筛选模式 |

### 示例

```bash
python 2.logic2-multi-lofo.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV-new --output-dir ./lofo_results
```

---

## 脚本三：2.logic2-multi-fewshot.py

### 功能
Few-shot 少样本学习实验，评估在极有限标注数据下的检测性能。

### K-shot 设置
- K ∈ {1, 3, 5, 10}
- 每个符合条件的 C2 家族随机采样 exactly K 个流簇

### 使用方法

```bash
python 2.logic2-multi-fewshot.py --c2-data <C2数据路径> --benign-data <良性数据路径> [选项]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--c2-data` | str | ./MTA-csv | C2数据路径 |
| `--benign-data` | str | ./CTU-csv | 良性数据路径 |
| `--k-shot` | int | 5 | K-shot 值（1/3/5/10） |
| `--benign-sample-size` | int | 500 | 良性样本数量限制 |
| `--output-dir` | str | ./fewshot_results | 结果输出目录 |
| `--filter-mode` | str | multi_only | 数据筛选模式 |
| `--protocol` | str | ALL | 测试协议：ALL/TLS/HTTP/TCP |
| `--verbose` | bool | False | 是否输出详细评估信息 |

### 示例

**5-shot 实验:**
```bash
python 2.logic2-multi-fewshot.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV-new --k-shot 5
```

**1-shot 实验:**
```bash
python 2.logic2-multi-fewshot.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV-new --k-shot 1
```

**只测试 TLS 协议:**
```bash
python 2.logic2-multi-fewshot.py --c2-data ./MTA-C2 --benign-data ./CTU-CSV-new --k-shot 5 --protocol TLS
```

---

## 数据预处理工具包含为1.x.xxxx.py


---

## 输出指标说明

所有脚本输出以下评估指标：

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1-Score | F1 分数 |
| FPR | 误报率（False Positive Rate） |
| ROC-AUC | ROC 曲线下面积 |

---

## 注意事项

1. **数据格式**: CSV 文件需包含以下列：`Session ID`, `Length`, `Timestamp`, `Session Duration`, `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Protocols`, `Type`, `Packet Count`, `Block Count`, `Protocol`, `TLS Version`

2. **协议识别**: 系统通过文件名后缀识别协议类型：
   - `_tls1.0.csv`, `_tls1.1.csv`, `_tls1.2.csv`, `_tls1.3.csv` → TLS
   - `_http.csv` → HTTP
   - `_tcp.csv` → TCP

3. **GPU 加速**: 如有 NVIDIA GPU，程序会自动使用 CUDA 加速

4. **随机种子**: 默认使用 seed=42 以保证结果可复现
