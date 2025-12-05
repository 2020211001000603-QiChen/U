# MELD-DA数据集实验配置指南

本文档说明如何从MIntRec数据集切换到MELD-DA数据集进行实验。

## 一、命令行参数修改

### 1. 基本参数修改

在运行脚本或命令行中，需要修改以下参数：

```bash
# MIntRec配置
--dataset MIntRec
--config_file_name umc_MIntRec.py

# MELD-DA配置（修改为）
--dataset MELD-DA
--config_file_name umc_MELD-DA.py
```

### 2. 完整示例

**MIntRec运行命令：**
```bash
python run.py \
    --dataset MIntRec \
    --data_path 'Datasets' \
    --config_file_name umc_MIntRec \
    --train \
    --gpu_id '0'
```

**MELD-DA运行命令：**
```bash
python run.py \
    --dataset MELD-DA \
    --data_path 'Datasets' \
    --config_file_name umc_MELD-DA \
    --train \
    --gpu_id '0'
```

## 二、配置文件差异

### 1. 配置文件位置
- MIntRec: `UMC-main/configs/umc_MIntRec.py`
- MELD-DA: `UMC-main/configs/umc_MELD-DA.py`

### 2. 主要参数差异对比

| 参数 | MIntRec | MELD-DA |
|------|---------|---------|
| `train_temperature_sup` | [1.4] | [20] |
| `train_temperature_unsup` | [1] | [20] |
| `lr_pre` | 2e-5 | [2e-5] |
| `lr` | [3e-4] | [2e-4] |
| `delta` | [0.02] | [0.05] |
| `thres` | [0.05] | [0.1] |
| `num_pretrain_epochs` | 100 | [100] |
| `num_train_epochs` | 100 | [100] |
| `base_dim` | 256 | [256] |
| `grad_clip` | -1.0 | [-1.0] |

### 3. MIntRec特有参数（消融实验相关）

MIntRec配置文件包含以下消融实验相关参数，MELD-DA配置中没有：
- `enable_video_dual`
- `enable_audio_dual`
- `enable_text_guided_attention`
- `enable_self_attention`
- `enable_clustering_optimization`
- `enable_clustering_loss`
- `enable_progressive_learning`
- `enable_adaptive_threshold`
- `enable_early_stop`
- `max_threshold`, `min_threshold`
- `performance_window`, `patience`
- `clustering_weight`, `contrastive_weight`等损失权重参数
- `fixed_train_epochs`

**注意**：如果需要在MELD-DA上运行消融实验，需要将MIntRec配置中的消融实验参数添加到MELD-DA配置文件中。

## 三、数据集特定差异

### 1. 数据集配置（`data/__init__.py`）

| 项目 | MIntRec | MELD-DA |
|------|---------|---------|
| **标签数量** | 20个 | 12个 |
| **标签列表** | ['Complain', 'Praise', ...] | ['a', 'ag', 'ans', 'ap', 'b', 'c', 'dag', 'g', 'o', 'q', 's', 'oth'] |
| **文本最大序列长度** | 30 | 70 |
| **视频最大序列长度** | 230 | 250 |
| **音频最大序列长度** | 480 | 520 |
| **文本特征维度** | 768 | 768 |
| **视频特征维度** | 1024 | 1024 |
| **音频特征维度** | 768 | 768 |

### 2. 数据读取差异（`data/text_pre.py`）

```python
# MIntRec: select_id = 3
# MELD-DA: select_id = 2
```

### 3. 索引和标签读取差异（`data/base.py`）

**MIntRec:**
```python
if args.dataset in ['MIntRec']:
    index = '_'.join([line[0], line[1], line[2]])
    label_id = label_map[line[4]]
```

**MELD-DA:**
```python
elif args.dataset in ['MELD-DA']:
    label_id = label_map[line[3]]
    index = '_'.join([line[0], line[1]])
```

## 四、数据文件结构要求

确保数据文件结构正确：

```
Datasets/
├── MELD-DA/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── video_data/
│   │   └── swin_feats.pkl  (或其他视频特征文件)
│   └── audio_data/
│       └── wavlm_feats.pkl (或其他音频特征文件)
```

## 五、快速切换步骤

### 方法1：直接修改命令行参数（推荐）

```bash
# 只需修改这两个参数
python run.py \
    --dataset MELD-DA \
    --config_file_name umc_MELD-DA \
    --data_path 'Datasets' \
    --train \
    --gpu_id '0'
```

### 方法2：修改运行脚本

编辑 `examples/run_umc.sh` 或其他运行脚本：

```bash
# 修改前
--dataset MIntRec \
--config_file_name umc_MIntRec \

# 修改后
--dataset MELD-DA \
--config_file_name umc_MELD-DA \
```

## 六、注意事项

1. **配置文件名称**：`--config_file_name` 参数不需要 `.py` 后缀
2. **数据集名称大小写**：必须使用 `MELD-DA`（注意大小写和连字符）
3. **消融实验**：如果需要在MELD-DA上运行消融实验，需要：
   - 将MIntRec配置中的消融实验参数复制到MELD-DA配置文件
   - 或者修改MELD-DA配置文件，添加消融实验支持
4. **数据路径**：确保 `--data_path` 指向包含MELD-DA数据集的目录
5. **特征文件**：确保视频和音频特征文件路径正确（`--video_feats_path` 和 `--audio_feats_path`）

## 七、验证配置

运行前可以通过以下方式验证：

```bash
# 先不训练，只测试配置是否正确
python run.py \
    --dataset MELD-DA \
    --config_file_name umc_MELD-DA \
    --data_path 'Datasets' \
    --gpu_id '0'
```

如果配置正确，应该能正常加载数据集和模型配置，不会报错。

