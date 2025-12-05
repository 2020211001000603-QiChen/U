# MIntRec数据集消融实验完整指南

## 📌 三个创新点的开关位置

### 开关位置：`configs/umc_MIntRec.py` 中的 `_get_ablation_config()` 方法

所有创新点的开关都在这个方法的 `default_config` 字典中定义。

---

## 🎯 三个创新点的开关参数

### 创新点一：ConFEDE双投影机制

**开关参数**：
```python
'enable_video_dual': True,      # 视频双投影开关
'enable_audio_dual': True,      # 音频双投影开关
'enable_dual_projection': True,  # 双投影总开关
```

**位置**：`configs/umc_MIntRec.py` 第 112-113 行

**作用**：
- `enable_video_dual`: 控制是否对视频特征进行双投影（主要信息+环境信息）
- `enable_audio_dual`: 控制是否对音频特征进行双投影（主要信息+环境信息）

---

### 创新点二：文本引导多模态注意力融合

**开关参数**：
```python
'enable_text_guided_attention': True,  # 文本引导注意力开关
'enable_self_attention': True,         # 自注意力机制开关
'self_attention_layers': 2,           # 自注意力层数
```

**位置**：`configs/umc_MIntRec.py` 第 116-118 行

**作用**：
- `enable_text_guided_attention`: 控制是否使用文本引导的交叉注意力
- `enable_self_attention`: 控制是否使用自注意力机制
- `self_attention_layers`: 控制自注意力的层数

---

### 创新点三：自适应渐进式学习策略

**开关参数**：
```python
# 渐进式学习
'enable_progressive_learning': True,        # 渐进式学习总开关
'enable_adaptive_threshold': True,         # 自适应阈值开关
'enable_performance_monitoring': True,     # 性能监控开关
'enable_early_stop': True,                 # 早停机制开关

# 聚类优化
'enable_clustering_optimization': True,    # 聚类优化总开关
'enable_clustering_loss': True,            # 聚类损失开关
'enable_contrastive_loss': True,           # 对比学习损失开关
'enable_compactness_loss': True,           # 紧密度损失开关
'enable_separation_loss': True,            # 分离度损失开关
```

**位置**：`configs/umc_MIntRec.py` 第 121-135 行

**作用**：
- `enable_progressive_learning`: 控制是否使用渐进式学习策略
- `enable_adaptive_threshold`: 控制是否使用自适应阈值调整
- `enable_clustering_optimization`: 控制是否启用聚类优化路径
- 各种损失开关：控制是否使用相应的损失函数

---

## 🚀 如何运行消融实验

### 方法一：使用命令行参数（推荐）

#### 1. 运行单个消融实验

```bash
python run.py \
    --dataset MIntRec \
    --data_path 'Datasets' \
    --logger_name umc \
    --multimodal_method umc \
    --method umc \
    --train \
    --tune \
    --save_results \
    --seed 0 \
    --gpu_id '0' \
    --video_feats_path 'swin_feats.pkl' \
    --audio_feats_path 'wavlm_feats.pkl' \
    --text_backbone bert-base-uncased \
    --config_file_name umc_MIntRec \
    --ablation_experiment no_dual_projection \
    --results_file_name "results_ablation_no_dual.csv" \
    --output_path "outputs/MIntRec/ablation_no_dual"
```

**关键参数**：
- `--ablation_experiment`: 指定消融实验名称（见下方实验列表）
- `--dataset MIntRec`: 指定数据集
- `--config_file_name umc_MIntRec`: 指定配置文件

#### 2. 运行多个种子（推荐5次）

```bash
for seed in 0 1 2 3 4; do
    python run.py \
        --dataset MIntRec \
        --data_path 'Datasets' \
        --train \
        --tune \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --config_file_name umc_MIntRec \
        --ablation_experiment no_dual_projection \
        --results_file_name "results_ablation_no_dual_seed${seed}.csv" \
        --output_path "outputs/MIntRec/ablation_no_dual/seed${seed}"
done
```

#### 3. 使用已有的脚本

项目已提供消融实验脚本：

```bash
# 创新点一消融实验
sh examples/run_innovation1_ablation.sh

# 创新点二消融实验
sh examples/run_innovation2_ablation.sh

# 创新点三消融实验
sh examples/run_innovation3_ablation.sh
```

---

### 方法二：直接修改配置文件

如果您想手动控制开关，可以直接修改 `configs/umc_MIntRec.py`：

#### 禁用创新点一（双投影）

在 `_get_ablation_config()` 方法的 `default_config` 中：
```python
default_config = {
    'enable_video_dual': False,  # 改为 False
    'enable_audio_dual': False,  # 改为 False
    # ... 其他配置保持不变
}
```

#### 禁用创新点二（文本引导注意力）

```python
default_config = {
    'enable_text_guided_attention': False,  # 改为 False
    'enable_self_attention': False,          # 改为 False
    # ... 其他配置保持不变
}
```

#### 禁用创新点三（渐进式学习）

```python
default_config = {
    'enable_progressive_learning': False,     # 改为 False
    'enable_adaptive_threshold': False,       # 改为 False
    'enable_clustering_optimization': False,  # 改为 False
    # ... 其他配置保持不变
}
```

---

## 📋 预定义的消融实验列表

### 创新点一：ConFEDE机制消融

| 实验名称 | 说明 | 开关设置 |
|---------|------|---------|
| `no_dual_projection` | 禁用双投影 | `enable_video_dual: False`, `enable_audio_dual: False` |
| `full_confede` | 完整ConFEDE机制 | 所有ConFEDE相关功能启用 |

### 创新点二：文本引导注意力消融

需要在配置文件中添加新的实验配置（见下方"自定义消融实验"部分）

### 创新点三：渐进式学习消融

| 实验名称 | 说明 | 开关设置 |
|---------|------|---------|
| `no_clustering_loss` | 禁用聚类损失 | 所有损失函数关闭 |
| `no_progressive_learning` | 禁用渐进式学习 | `enable_progressive_learning: False` |

---

## 🔧 如何自定义消融实验

### 步骤1：在配置文件中添加新实验

编辑 `configs/umc_MIntRec.py`，在 `_get_ablation_config()` 方法中添加：

```python
def _get_ablation_config(self, args):
    default_config = {
        # ... 默认配置
    }
    
    if hasattr(args, 'ablation_experiment') and args.ablation_experiment:
        experiment_name = args.ablation_experiment.lower()
        
        # 添加您的自定义实验
        if experiment_name == 'my_custom_experiment':
            config = default_config.copy()
            config.update({
                'enable_video_dual': False,           # 禁用视频双投影
                'enable_audio_dual': True,            # 保留音频双投影
                'enable_text_guided_attention': False, # 禁用文本引导注意力
                # ... 其他配置
            })
            return config
```

### 步骤2：运行自定义实验

```bash
python run.py \
    --dataset MIntRec \
    --ablation_experiment my_custom_experiment \
    --train \
    --save_results
```

---

## 📊 完整的消融实验方案

### 方案一：单创新点消融（验证每个创新点的独立贡献）

#### 实验1：仅创新点一
```bash
# 需要自定义配置：只启用双投影，禁用其他创新点
python run.py --dataset MIntRec --ablation_experiment only_innovation1 --train
```

#### 实验2：仅创新点二
```bash
# 需要自定义配置：只启用文本引导注意力，禁用其他创新点
python run.py --dataset MIntRec --ablation_experiment only_innovation2 --train
```

#### 实验3：仅创新点三
```bash
# 需要自定义配置：只启用渐进式学习，禁用其他创新点
python run.py --dataset MIntRec --ablation_experiment only_innovation3 --train
```

### 方案二：组合消融（验证创新点协同效果）

#### 实验4：创新点一 + 创新点二
```bash
# 需要自定义配置：启用创新点一和二，禁用创新点三
python run.py --dataset MIntRec --ablation_experiment innovation1_2 --train
```

#### 实验5：创新点一 + 创新点三
```bash
python run.py --dataset MIntRec --ablation_experiment innovation1_3 --train
```

#### 实验6：创新点二 + 创新点三
```bash
python run.py --dataset MIntRec --ablation_experiment innovation2_3 --train
```

#### 实验7：完整UMC（所有创新点）
```bash
# 不指定 ablation_experiment，使用默认配置
python run.py --dataset MIntRec --train
```

### 方案三：详细组件消融

#### 创新点一组件消融
- `video_dual_only`: 仅视频双投影
- `audio_dual_only`: 仅音频双投影
- `full_dual_projection`: 完整双投影

#### 创新点二组件消融
- `text_guided_only`: 仅文本引导注意力
- `self_attention_only`: 仅自注意力
- `full_attention`: 完整注意力机制

#### 创新点三组件消融
- `progressive_only`: 仅渐进式策略
- `clustering_loss_only`: 仅聚类损失
- `full_progressive`: 完整渐进式策略

---

## 🎯 快速开始：运行基线对比

### 运行基线（禁用所有创新点）

首先需要在配置文件中添加基线实验配置：

```python
# 在 _get_ablation_config() 中添加
elif experiment_name == 'baseline':
    config = default_config.copy()
    config.update({
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
    })
```

然后运行：
```bash
python run.py --dataset MIntRec --ablation_experiment baseline --train --save_results
```

### 运行完整UMC（所有创新点）

```bash
# 不指定 ablation_experiment，使用默认配置（所有创新点启用）
python run.py --dataset MIntRec --train --save_results
```

---

## 📝 配置文件关键位置总结

### 开关定义位置

**文件**：`configs/umc_MIntRec.py`

**方法**：`_get_ablation_config()` (第 106-209 行)

**关键代码段**：
```python
default_config = {
    # 创新点一：双投影机制
    'enable_video_dual': True,      # 第 112 行
    'enable_audio_dual': True,       # 第 113 行
    
    # 创新点二：文本引导注意力
    'enable_text_guided_attention': True,  # 第 116 行
    'enable_self_attention': True,         # 第 117 行
    
    # 创新点三：渐进式学习
    'enable_progressive_learning': True,    # 第 127 行
    'enable_adaptive_threshold': True,     # 第 128 行
    'enable_clustering_optimization': True,  # 第 135 行
}
```

### 开关传递位置

**文件**：`configs/umc_MIntRec.py`

**方法**：`_get_hyper_parameters()` (第 69-74 行)

**关键代码**：
```python
'enable_video_dual': ablation_config.get('enable_video_dual', True),
'enable_audio_dual': ablation_config.get('enable_audio_dual', True),
'enable_text_guided_attention': ablation_config.get('enable_text_guided_attention', True),
'enable_self_attention': ablation_config.get('enable_self_attention', True),
'enable_clustering_optimization': ablation_config.get('enable_clustering_optimization', True),
```

---

## 🔍 验证开关是否生效

### 方法1：查看日志

运行时会打印配置信息，检查日志中是否显示：
```
enable_video_dual: True/False
enable_audio_dual: True/False
enable_text_guided_attention: True/False
```

### 方法2：检查模型代码

在 `backbones/FusionNets/UMC.py` 中，开关通过以下方式使用：
```python
if self.enable_video_dual:
    # 执行视频双投影
    video_feat = self.video_dual_projector(video_feat)
```

如果开关为 `False`，这段代码不会执行。

---

## 📚 相关文件

- `configs/umc_MIntRec.py` - MIntRec配置文件（开关定义位置）
- `backbones/FusionNets/UMC.py` - UMC模型实现（开关使用位置）
- `examples/run_innovation1_ablation.sh` - 创新点一消融脚本
- `examples/run_innovation2_ablation.sh` - 创新点二消融脚本
- `examples/run_innovation3_ablation.sh` - 创新点三消融脚本
- `ABLATION_EXPERIMENTS_README.md` - 消融实验详细说明

---

## ⚠️ 注意事项

1. **修改配置文件后需要重新运行**：修改 `umc_MIntRec.py` 后，需要重新启动训练
2. **建议使用命令行参数**：使用 `--ablation_experiment` 参数比直接修改配置文件更方便
. **多次运行取平均**：每个消融实验建议运行5次（seed: 0-4）取平均值
3. **保存结果**：使用 `--save_results` 参数保存实验结果
4. **检查GPU内存**：某些配置可能需要更多GPU内存

---

## 🎯 推荐实验顺序

1. **第一步**：运行完整UMC（所有创新点）
2. **第二步**：运行基线（禁用所有创新点）
3. **第三步**：运行单创新点消融（验证每个创新点的独立贡献）
4. **第四步**：运行组合消融（验证创新点协同效果）
5. **第五步**：运行详细组件消融（深入分析）

---

**最后更新**：2024年


