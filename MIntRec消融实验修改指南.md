# MIntRec消融实验修改指南

## 📍 修改位置

**文件**：`configs/umc_MIntRec.py`  
**方法**：`_get_ablation_config()` (第 106-209 行)

---

## 🎯 步骤一：添加消融实验配置

### 在 `_get_ablation_config()` 方法中添加新实验

打开 `configs/umc_MIntRec.py`，找到第 147 行开始的 `if hasattr(args, 'ablation_experiment')` 部分：

```python
# 根据消融实验名称调整配置
if hasattr(args, 'ablation_experiment') and args.ablation_experiment:
    experiment_name = args.ablation_experiment.lower()
    
    # 在这里添加您的消融实验配置
    if experiment_name == 'your_experiment_name':
        config = default_config.copy()
        config.update({
            # 设置开关
        })
        return config
```

---

## 📋 常用消融实验配置模板

### 1. 基线实验（禁用所有创新点）

在第 202 行的 `else:` 之前添加：

```python
elif experiment_name == 'baseline':
    # 基线：禁用所有创新点
    config = default_config.copy()
    config.update({
        # 创新点一：禁用双投影
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_dual_projection': False,
        
        # 创新点二：禁用文本引导注意力
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        
        # 创新点三：禁用渐进式学习和聚类优化
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_performance_monitoring': False,
        'enable_early_stop': False,
        'enable_clustering_optimization': False,
        'enable_clustering_loss': False,
        'enable_contrastive_loss': False,
        'enable_compactness_loss': False,
        'enable_separation_loss': False,
    })
```

### 2. 仅创新点一（ConFEDE双投影）

```python
elif experiment_name == 'only_innovation1':
    # 仅启用创新点一：ConFEDE双投影
    config = default_config.copy()
    config.update({
        # 创新点一：启用双投影
        'enable_video_dual': True,
        'enable_audio_dual': True,
        'enable_dual_projection': True,
        
        # 创新点二：禁用文本引导注意力
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        
        # 创新点三：禁用渐进式学习
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
    })
```

### 3. 仅创新点二（文本引导注意力）

```python
elif experiment_name == 'only_innovation2':
    # 仅启用创新点二：文本引导注意力
    config = default_config.copy()
    config.update({
        # 创新点一：禁用双投影
        'enable_video_dual': False,
        'enable_audio_dual': False,
        
        # 创新点二：启用文本引导注意力
        'enable_text_guided_attention': True,
        'enable_self_attention': True,
        'self_attention_layers': 2,
        
        # 创新点三：禁用渐进式学习
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
    })
```

### 4. 仅创新点三（渐进式学习）

```python
elif experiment_name == 'only_innovation3':
    # 仅启用创新点三：渐进式学习
    config = default_config.copy()
    config.update({
        # 创新点一：禁用双投影
        'enable_video_dual': False,
        'enable_audio_dual': False,
        
        # 创新点二：禁用文本引导注意力
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        
        # 创新点三：启用渐进式学习
        'enable_progressive_learning': True,
        'enable_adaptive_threshold': True,
        'enable_performance_monitoring': True,
        'enable_early_stop': True,
        'enable_clustering_optimization': True,
        'enable_clustering_loss': True,
        'enable_contrastive_loss': True,
        'enable_compactness_loss': True,
        'enable_separation_loss': True,
    })
```

### 5. 创新点一 + 创新点二（组合）

```python
elif experiment_name == 'innovation1_2':
    # 创新点一 + 创新点二
    config = default_config.copy()
    config.update({
        # 创新点一：启用双投影
        'enable_video_dual': True,
        'enable_audio_dual': True,
        
        # 创新点二：启用文本引导注意力
        'enable_text_guided_attention': True,
        'enable_self_attention': True,
        
        # 创新点三：禁用渐进式学习
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
    })
```

### 6. 仅视频双投影（创新点一部分）

```python
elif experiment_name == 'video_dual_only':
    # 仅启用视频双投影
    config = default_config.copy()
    config.update({
        'enable_video_dual': True,
        'enable_audio_dual': False,  # 禁用音频双投影
        # 其他配置保持默认
    })
```

### 7. 仅音频双投影（创新点一部分）

```python
elif experiment_name == 'audio_dual_only':
    # 仅启用音频双投影
    config = default_config.copy()
    config.update({
        'enable_video_dual': False,  # 禁用视频双投影
        'enable_audio_dual': True,
        # 其他配置保持默认
    })
```

### 8. 仅文本引导注意力（创新点二部分）

```python
elif experiment_name == 'text_guided_only':
    # 仅启用文本引导注意力，禁用自注意力
    config = default_config.copy()
    config.update({
        'enable_text_guided_attention': True,
        'enable_self_attention': False,  # 禁用自注意力
        # 其他配置保持默认
    })
```

### 9. 仅自注意力（创新点二部分）

```python
elif experiment_name == 'self_attention_only':
    # 仅启用自注意力，禁用文本引导注意力
    config = default_config.copy()
    config.update({
        'enable_text_guided_attention': False,  # 禁用文本引导注意力
        'enable_self_attention': True,
        # 其他配置保持默认
    })
```

---

## 🚀 步骤二：运行消融实验

### 方法1：使用命令行参数

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
    --ablation_experiment baseline \
    --results_file_name "results_baseline.csv" \
    --output_path "outputs/MIntRec/baseline"
```

**关键参数**：
- `--ablation_experiment baseline`：指定消融实验名称（与配置文件中添加的名称一致）
- `--dataset MIntRec`：指定数据集
- `--config_file_name umc_MIntRec`：指定配置文件

### 方法2：运行多次取平均（推荐）

```bash
for seed in 0 1 2 3 4; do
    python run.py \
        --dataset MIntRec \
        --ablation_experiment baseline \
        --seed $seed \
        --train \
        --save_results \
        --results_file_name "results_baseline_seed${seed}.csv" \
        --output_path "outputs/MIntRec/baseline/seed${seed}"
done
```

### 方法3：使用批处理脚本

创建 `run_ablation_MIntRec.sh`：

```bash
#!/usr/bin/bash

# MIntRec消融实验脚本

experiments=(
    "baseline"
    "only_innovation1"
    "only_innovation2"
    "only_innovation3"
    "innovation1_2"
)

for experiment in "${experiments[@]}"
do
    echo "=========================================="
    echo "运行消融实验: $experiment"
    echo "=========================================="
    
    for seed in 0 1 2 3 4
    do
        echo "运行种子: $seed"
        
        python run.py \
        --dataset MIntRec \
        --data_path 'Datasets' \
        --train \
        --tune \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --config_file_name umc_MIntRec \
        --ablation_experiment $experiment \
        --results_file_name "results_${experiment}_seed${seed}.csv" \
        --output_path "outputs/MIntRec/${experiment}/seed${seed}"
        
        echo "种子 $seed 完成"
    done
    
    echo "实验 $experiment 完成"
    echo ""
done

echo "所有消融实验完成！"
```

运行：
```bash
chmod +x run_ablation_MIntRec.sh
./run_ablation_MIntRec.sh
```

---

## 📝 完整修改示例

### 在 `configs/umc_MIntRec.py` 中添加以下代码

在第 201 行的 `else:` 之前添加：

```python
elif experiment_name == 'baseline':
    # 基线：禁用所有创新点
    config = default_config.copy()
    config.update({
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
        'enable_clustering_loss': False,
        'enable_contrastive_loss': False,
    })
elif experiment_name == 'only_innovation1':
    # 仅创新点一
    config = default_config.copy()
    config.update({
        'enable_video_dual': True,
        'enable_audio_dual': True,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
    })
elif experiment_name == 'only_innovation2':
    # 仅创新点二
    config = default_config.copy()
    config.update({
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_text_guided_attention': True,
        'enable_self_attention': True,
        'enable_progressive_learning': False,
        'enable_adaptive_threshold': False,
        'enable_clustering_optimization': False,
    })
elif experiment_name == 'only_innovation3':
    # 仅创新点三
    config = default_config.copy()
    config.update({
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_progressive_learning': True,
        'enable_adaptive_threshold': True,
        'enable_clustering_optimization': True,
        'enable_clustering_loss': True,
        'enable_contrastive_loss': True,
    })
```

---

## ✅ 验证修改是否生效

### 方法1：查看日志

运行时会打印配置信息，检查是否显示：
```
enable_video_dual: True/False
enable_audio_dual: True/False
enable_text_guided_attention: True/False
```

### 方法2：检查结果

运行后检查结果文件，对比不同消融实验的性能差异。

---

## 🎯 推荐的消融实验列表

### 必须做的实验

1. **baseline** - 基线（禁用所有创新点）
2. **only_innovation1** - 仅创新点一
3. **only_innovation2** - 仅创新点二
4. **only_innovation3** - 仅创新点三
5. **完整UMC** - 所有创新点（不指定 `--ablation_experiment`）

### 建议做的实验

6. **innovation1_2** - 创新点一 + 二
7. **innovation1_3** - 创新点一 + 三
8. **innovation2_3** - 创新点二 + 三
9. **no_dual_projection** - 禁用双投影（已有）
10. **no_progressive_learning** - 禁用渐进式学习（已有）

---

## 📊 运行完整消融实验方案

### 方案一：快速验证（每个实验1次）

```bash
experiments=("baseline" "only_innovation1" "only_innovation2" "only_innovation3")

for exp in "${experiments[@]}"; do
    python run.py --dataset MIntRec --ablation_experiment $exp --train --save_results
done
```

### 方案二：标准实验（每个实验5次）

```bash
experiments=("baseline" "only_innovation1" "only_innovation2" "only_innovation3")

for exp in "${experiments[@]}"; do
    for seed in 0 1 2 3 4; do
        python run.py \
            --dataset MIntRec \
            --ablation_experiment $exp \
            --seed $seed \
            --train \
            --save_results
    done
done
```

---

## ⚠️ 注意事项

1. **实验名称必须小写**：配置文件中使用 `experiment_name.lower()`，所以实验名称不区分大小写
2. **修改后需要重新运行**：修改配置文件后需要重新启动训练
3. **保存结果**：使用 `--save_results` 参数保存实验结果
4. **多次运行**：建议每个实验运行5次（seed: 0-4）取平均值

---

## 🔍 检查已有实验

当前配置文件中已有的消融实验：
- ✅ `no_dual_projection` - 禁用双投影
- ✅ `no_clustering_loss` - 禁用聚类损失
- ✅ `no_progressive_learning` - 禁用渐进式学习
- ✅ `full_confede` - 完整ConFEDE

---

## 📚 相关文档

- `MIntRec消融实验指南.md` - 详细的消融实验指南
- `创新点开关快速参考.md` - 开关参数快速参考
- `为什么需要多次seed实验.md` - 多次运行的必要性

---

**总结**：修改 `configs/umc_MIntRec.py` 的 `_get_ablation_config()` 方法，添加新的 `elif` 分支定义消融实验配置，然后使用 `--ablation_experiment` 参数运行即可！








