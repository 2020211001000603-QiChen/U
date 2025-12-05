# IEMOCAP-DA 数据集实验配置说明

## 一、配置文件更新

已更新 `configs/umc_IEMOCAP-DA.py`，应用了与 MELD-DA 相同的优化。

## 二、主要修改内容

### 1. **批次大小调整**（适合48GB GPU）
- `pretrain_batch_size`: 128 → **64**
- `train_batch_size`: 128 → **64**
- `eval_batch_size`: 128 → **64**
- `test_batch_size`: 128 → **64**

### 2. **模型维度调整**
- `base_dim`: [256] → **[128]**（减少特征维度，降低内存）

### 3. **新增功能支持**
- ✅ 消融实验配置支持
- ✅ 对比学习参数
- ✅ 聚类损失参数
- ✅ 渐进式学习优化

### 4. **IEMOCAP-DA 特有参数**（已保留）

| 参数 | IEMOCAP-DA | MELD-DA | 说明 |
|------|-----------|---------|------|
| `pretrain` | [False] | [True] | IEMOCAP-DA不使用预训练 |
| `train_temperature_sup` | [10] | [20] | 监督学习温度 |
| `train_temperature_unsup` | [20] | [20] | 无监督学习温度 |
| `lr` | [5e-4] | [2e-4] | 学习率 |

## 三、数据集信息

### IEMOCAP-DA 数据集配置

```python
'labels': ['oth', 'ap', 'o', 'g', 's', 'a', 'b', 'c', 'ans', 'q', 'ag', 'dag']  # 12个标签
'max_seq_lengths': {
    'text': 44,      # 文本序列长度
    'video': 230,    # 视频序列长度
    'audio': 380     # 音频序列长度
}
'feat_dims': {
    'text': 768,
    'video': 1024,
    'audio': 768
}
```

## 四、运行方式

### 方法1：使用提供的脚本

```bash
chmod +x examples/run_umc_iemocap.sh
./examples/run_umc_iemocap.sh
```

### 方法2：命令行直接运行

```bash
python run.py \
    --dataset IEMOCAP-DA \
    --data_path 'Datasets' \
    --train \
    --save_model \
    --save_results \
    --seed 0 \
    --gpu_id '0' \
    --num_workers 2 \
    --config_file_name umc_IEMOCAP-DA
```

## 五、参数对比

### IEMOCAP-DA vs MELD-DA

| 参数 | IEMOCAP-DA | MELD-DA | 差异说明 |
|------|-----------|---------|---------|
| **批次大小** | 64 | 64 | 相同 |
| **base_dim** | [128] | [128] | 相同 |
| **pretrain** | [False] | [True] | IEMOCAP-DA不使用预训练 |
| **train_temperature_sup** | [10] | [20] | IEMOCAP-DA使用更低的温度 |
| **lr** | [5e-4] | [2e-4] | IEMOCAP-DA使用更高的学习率 |
| **delta** | [0.05] | [0.05] | 相同 |
| **thres** | [0.1] | [0.1] | 相同 |

## 六、数据文件要求

确保数据文件结构正确：

```
Datasets/
├── IEMOCAP-DA/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── video_data/
│   │   └── swin_feats.pkl
│   └── audio_data/
│       └── wavlm_feats.pkl
```

## 七、注意事项

### 1. **预训练设置**

IEMOCAP-DA 配置中 `pretrain: [False]`，意味着：
- ❌ 不会运行预训练阶段
- ✅ 直接进入主训练阶段
- 这与 MELD-DA 不同（MELD-DA 使用预训练）

### 2. **温度参数**

IEMOCAP-DA 使用：
- `train_temperature_sup: [10]` - 比 MELD-DA 的 [20] 更低
- `train_temperature_unsup: [20]` - 与 MELD-DA 相同

### 3. **学习率**

IEMOCAP-DA 使用更高的学习率：
- `lr: [5e-4]` - 比 MELD-DA 的 [2e-4] 更高

### 4. **内存优化**

- 批次大小已优化为 64（适合48GB GPU）
- `base_dim` 减少到 128（降低内存占用）
- `num_workers` 建议设置为 2（减少内存占用）

## 八、快速开始

### 1. 检查数据文件

```bash
ls -la Datasets/IEMOCAP-DA/
# 应该看到：train.tsv, dev.tsv, test.tsv, video_data/, audio_data/
```

### 2. 运行基线实验

```bash
# 使用脚本
./examples/run_umc_iemocap.sh

# 或直接运行
python run.py \
    --dataset IEMOCAP-DA \
    --config_file_name umc_IEMOCAP-DA \
    --train \
    --save_model \
    --gpu_id '0'
```

### 3. 运行消融实验

```bash
python run.py \
    --dataset IEMOCAP-DA \
    --config_file_name umc_IEMOCAP-DA \
    --ablation_experiment no_dual_projection \
    --train \
    --save_model \
    --gpu_id '0'
```

## 九、配置特点

✅ **适合48GB GPU**：批次大小64，启用所有功能  
✅ **内存优化**：`base_dim` 设为128  
✅ **完整功能**：所有创新点功能均启用  
✅ **保留IEMOCAP-DA特有参数**：温度、学习率、预训练设置等  
✅ **支持消融实验**：可以运行各种消融实验  

## 十、与 MELD-DA 的主要区别

1. **预训练**：IEMOCAP-DA 不使用预训练（`pretrain: [False]`）
2. **温度参数**：监督学习温度更低（10 vs 20）
3. **学习率**：更高（5e-4 vs 2e-4）
4. **其他参数**：基本相同

配置文件已更新完成，可以直接使用！

