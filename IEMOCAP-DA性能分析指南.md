# IEMOCAP-DA 性能分析指南

## 一、数据集特征对比

### IEMOCAP-DA vs MELD-DA 数据集特征

| 特征 | IEMOCAP-DA | MELD-DA | 影响分析 |
|------|-----------|---------|---------|
| **文本序列长度** | 44 | 70 | ⚠️ IEMOCAP-DA文本更短，信息可能不足 |
| **视频序列长度** | 230 | 250 | ✅ 相近 |
| **音频序列长度** | 380 | 520 | ⚠️ IEMOCAP-DA音频更短，信息可能不足 |
| **标签数量** | 12 | 12 | ✅ 相同 |
| **标签类型** | 对话行为 | 对话行为 | ✅ 相同 |

### 关键差异

1. **文本信息量更少**：44 vs 70（减少37%）
2. **音频信息量更少**：380 vs 520（减少27%）
3. **总体信息密度更低**：可能导致特征表示不够丰富

---

## 二、可能影响性能的因素

### 1. **数据集规模和信息量**

#### 问题分析
- **文本序列短**：44个token可能无法充分表达语义
- **音频序列短**：380帧可能丢失重要音频特征
- **信息不足**：多模态信息量较少，难以学习有效的特征表示

#### 影响
- 特征表示可能不够丰富
- 聚类时区分度不够
- 模型难以学习复杂的多模态关系

### 2. **超参数设置**

#### 当前参数对比

| 参数 | IEMOCAP-DA | MELD-DA | 可能问题 |
|------|-----------|---------|---------|
| `lr` | 5e-4 | 2e-4 | ⚠️ 学习率过高，可能导致训练不稳定 |
| `train_temperature_sup` | 10 | 20 | ⚠️ 温度过低，可能导致学习过于保守 |
| `train_temperature_unsup` | 20 | 20 | ✅ 相同 |
| `pretrain` | True (已修改) | True | ✅ 现在相同 |
| `base_dim` | 128 | 128 | ✅ 相同 |

#### 潜在问题
1. **学习率过高**（5e-4 vs 2e-4）：
   - 可能导致训练不稳定
   - 难以收敛到最优解
   - 梯度更新过大

2. **监督温度过低**（10 vs 20）：
   - 温度越低，softmax分布越尖锐
   - 可能导致模型过于自信
   - 难以学习细粒度的特征区分

### 3. **数据质量**

#### 可能的问题
- **标注质量**：IEMOCAP-DA的标注可能不如MELD-DA一致
- **数据分布**：类别分布可能不平衡
- **噪声水平**：数据中可能包含更多噪声

### 4. **模型架构适配**

#### 可能的不匹配
- **序列长度适配**：模型可能更适合较长的序列
- **特征维度**：base_dim=128可能对IEMOCAP-DA来说太小
- **注意力机制**：短序列可能无法充分利用注意力机制

---

## 三、分析方法

### 方法1：数据统计分析

创建分析脚本来检查数据特征：

```python
# analyze_iemocap_data.py
import pandas as pd
import numpy as np
from collections import Counter

def analyze_dataset(data_path, dataset_name):
    """分析数据集特征"""
    
    # 读取数据
    train_df = pd.read_csv(f'{data_path}/{dataset_name}/train.tsv', sep='\t')
    
    # 1. 类别分布
    label_counts = Counter(train_df.iloc[:, 2] if dataset_name == 'IEMOCAP-DA' else train_df.iloc[:, 3])
    print(f"\n=== {dataset_name} 类别分布 ===")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(train_df)*100:.2f}%)")
    
    # 2. 类别不平衡度
    counts = list(label_counts.values())
    imbalance_ratio = max(counts) / min(counts)
    print(f"\n类别不平衡度: {imbalance_ratio:.2f}")
    
    # 3. 文本长度分布
    if dataset_name == 'IEMOCAP-DA':
        text_col = train_df.iloc[:, 1]
    else:
        text_col = train_df.iloc[:, 2]
    
    text_lengths = [len(str(text).split()) for text in text_col]
    print(f"\n文本长度统计:")
    print(f"  平均: {np.mean(text_lengths):.2f}")
    print(f"  中位数: {np.median(text_lengths):.2f}")
    print(f"  最小: {min(text_lengths)}")
    print(f"  最大: {max(text_lengths)}")
    
    return {
        'label_distribution': label_counts,
        'imbalance_ratio': imbalance_ratio,
        'text_length_stats': {
            'mean': np.mean(text_lengths),
            'median': np.median(text_lengths),
            'min': min(text_lengths),
            'max': max(text_lengths)
        }
    }

# 对比分析
iemocap_stats = analyze_dataset('Datasets', 'IEMOCAP-DA')
meld_stats = analyze_dataset('Datasets', 'MELD-DA')
```

### 方法2：训练过程分析

#### 检查训练日志

```bash
# 查看训练损失变化
grep "Training Loss" logs/*.log

# 查看聚类指标变化
grep "NMI\|ARI\|ACC" logs/*.log

# 查看阈值变化
grep "threshold" logs/*.log
```

#### 分析指标

1. **损失曲线**：
   - 是否收敛？
   - 是否震荡？
   - 是否过拟合？

2. **聚类指标趋势**：
   - NMI、ARI、ACC是否提升？
   - 是否达到平台期？

3. **阈值变化**：
   - 渐进式阈值是否正常工作？
   - 是否过早或过晚达到最大值？

### 方法3：特征质量分析

创建特征分析脚本：

```python
# analyze_features.py
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def analyze_features(model, dataloader, device):
    """分析学习到的特征质量"""
    
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取特征
            feats = model.get_features(batch)
            features.append(feats.cpu().numpy())
            labels.append(batch['label_ids'].cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 1. 计算类内距离和类间距离
    intra_class_distances = []
    inter_class_distances = []
    
    for label in np.unique(labels):
        class_features = features[labels == label]
        # 类内距离
        centroid = np.mean(class_features, axis=0)
        intra_dist = np.mean([np.linalg.norm(f - centroid) for f in class_features])
        intra_class_distances.append(intra_dist)
        
        # 类间距离
        other_features = features[labels != label]
        other_centroid = np.mean(other_features, axis=0)
        inter_dist = np.linalg.norm(centroid - other_centroid)
        inter_class_distances.append(inter_dist)
    
    print(f"平均类内距离: {np.mean(intra_class_distances):.4f}")
    print(f"平均类间距离: {np.mean(inter_class_distances):.4f}")
    print(f"分离度: {np.mean(inter_class_distances) / np.mean(intra_class_distances):.4f}")
    
    # 2. Silhouette Score
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=0)
    cluster_labels = kmeans.fit_predict(features)
    silhouette = silhouette_score(features, cluster_labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    return {
        'intra_class_distance': np.mean(intra_class_distances),
        'inter_class_distance': np.mean(inter_class_distances),
        'separation_ratio': np.mean(inter_class_distances) / np.mean(intra_class_distances),
        'silhouette_score': silhouette
    }
```

### 方法4：对比实验分析

#### 运行对比实验

```bash
# 实验1：使用MELD-DA的参数
python run.py --dataset IEMOCAP-DA --config_file_name umc_IEMOCAP-DA \
    --train --save_results --seed 0

# 实验2：调整学习率
# 修改配置文件：lr: [2e-4]  # 从5e-4改为2e-4

# 实验3：调整温度
# 修改配置文件：train_temperature_sup: [20]  # 从10改为20

# 实验4：增加base_dim
# 修改配置文件：base_dim: [256]  # 从128改为256
```

---

## 四、改进建议

### 建议1：调整超参数（优先尝试）

#### 1.1 降低学习率

```python
# configs/umc_IEMOCAP-DA.py
'lr': [2e-4],  # 从5e-4改为2e-4，与MELD-DA一致
```

**理由**：
- 当前学习率可能过高，导致训练不稳定
- MELD-DA使用2e-4效果更好

#### 1.2 提高监督温度

```python
# configs/umc_IEMOCAP-DA.py
'train_temperature_sup': [20],  # 从10改为20，与MELD-DA一致
```

**理由**：
- 温度过低可能导致学习过于保守
- 提高温度可以让模型学习更平滑的分布

#### 1.3 尝试不同的学习率

```python
# 可以尝试多个值
'lr': [1e-4, 2e-4, 3e-4],  # 使用tune模式自动搜索
```

### 建议2：增加模型容量

#### 2.1 增加base_dim

```python
# configs/umc_IEMOCAP-DA.py
'base_dim': [256],  # 从128改为256，增加模型容量
```

**理由**：
- IEMOCAP-DA信息量较少，可能需要更大的模型容量来学习有效特征
- 但会增加内存占用

#### 2.2 增加注意力层数

```python
# configs/umc_IEMOCAP-DA.py
'self_attention_layers': 3,  # 从2改为3
```

### 建议3：数据增强

#### 3.1 增加文本序列长度（如果可能）

检查是否可以：
- 使用更长的上下文
- 合并相邻的对话片段
- 使用更长的文本窗口

#### 3.2 音频特征增强

- 检查音频特征提取方法
- 尝试不同的音频特征（如果可用）

### 建议4：训练策略调整

#### 4.1 增加预训练轮数

```python
# configs/umc_IEMOCAP-DA.py
'num_pretrain_epochs': [150],  # 从100增加到150
```

**理由**：
- IEMOCAP-DA信息量少，可能需要更多预训练来学习多模态对齐

#### 4.2 调整阈值策略

```python
# configs/umc_IEMOCAP-DA.py
'thres': [0.05],      # 从0.1降低到0.05，更保守的阈值
'delta': [0.02],      # 从0.05降低到0.02，更慢的增长
```

**理由**：
- 信息量少的数据集可能需要更保守的阈值策略
- 更慢的增长可以让模型有更多时间学习

### 建议5：损失函数调整

#### 5.1 调整损失权重

```python
# configs/umc_IEMOCAP-DA.py
'clustering_weight': 1.5,        # 增加聚类损失权重
'contrastive_weight': 0.8,       # 增加对比学习权重
'compactness_weight': 1.2,       # 增加紧密度损失权重
'separation_weight': 1.2,        # 增加分离度损失权重
```

**理由**：
- IEMOCAP-DA可能需要更强的聚类约束
- 增加损失权重可以强化聚类目标

---

## 五、诊断检查清单

### 数据层面
- [ ] 检查数据文件是否正确加载
- [ ] 检查类别分布是否平衡
- [ ] 检查文本/音频长度分布
- [ ] 检查数据质量（是否有缺失值、异常值）

### 模型层面
- [ ] 检查模型是否正常初始化
- [ ] 检查特征维度是否正确
- [ ] 检查损失函数是否正常计算
- [ ] 检查梯度是否正常更新

### 训练层面
- [ ] 检查损失是否下降
- [ ] 检查学习率是否合适
- [ ] 检查批次大小是否合适
- [ ] 检查训练是否收敛

### 聚类层面
- [ ] 检查聚类中心是否合理
- [ ] 检查阈值策略是否有效
- [ ] 检查特征分离度
- [ ] 检查聚类指标趋势

---

## 六、快速诊断脚本

创建一个诊断脚本：

```python
# diagnose_iemocap.py
import os
import pandas as pd
import numpy as np
from collections import Counter

def diagnose_iemocap():
    """快速诊断IEMOCAP-DA数据集"""
    
    data_path = 'Datasets/IEMOCAP-DA'
    
    print("=" * 50)
    print("IEMOCAP-DA 数据集诊断")
    print("=" * 50)
    
    # 1. 检查文件
    required_files = ['train.tsv', 'dev.tsv', 'test.tsv']
    print("\n1. 文件检查:")
    for file in required_files:
        file_path = os.path.join(data_path, file)
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if exists:
            df = pd.read_csv(file_path, sep='\t', header=None)
            print(f"      样本数: {len(df)}")
    
    # 2. 类别分布
    print("\n2. 类别分布:")
    train_df = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t', header=None)
    labels = train_df.iloc[:, 2].tolist()
    label_counts = Counter(labels)
    
    for label, count in sorted(label_counts.items()):
        pct = count / len(labels) * 100
        print(f"   {label}: {count} ({pct:.2f}%)")
    
    # 3. 不平衡度
    counts = list(label_counts.values())
    imbalance = max(counts) / min(counts)
    print(f"\n   类别不平衡度: {imbalance:.2f}")
    if imbalance > 3:
        print("   ⚠️ 类别严重不平衡，可能影响性能")
    
    # 4. 文本长度
    print("\n3. 文本长度统计:")
    texts = train_df.iloc[:, 1].tolist()
    lengths = [len(str(t).split()) for t in texts]
    print(f"   平均长度: {np.mean(lengths):.2f}")
    print(f"   中位数: {np.median(lengths):.2f}")
    print(f"   最小: {min(lengths)}")
    print(f"   最大: {max(lengths)}")
    
    if np.mean(lengths) < 10:
        print("   ⚠️ 文本长度较短，信息可能不足")
    
    # 5. 参数建议
    print("\n4. 参数调整建议:")
    if imbalance > 3:
        print("   - 考虑使用类别权重平衡")
    if np.mean(lengths) < 10:
        print("   - 考虑增加base_dim或使用更长的上下文")
    print("   - 尝试降低学习率: lr = 2e-4")
    print("   - 尝试提高温度: train_temperature_sup = 20")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    diagnose_iemocap()
```

---

## 七、系统化改进方案

### 方案A：保守调整（推荐先试）

```python
# configs/umc_IEMOCAP-DA.py
'lr': [2e-4],                    # 降低学习率
'train_temperature_sup': [20],   # 提高温度
'thres': [0.05],                 # 降低初始阈值
'delta': [0.02],                 # 降低增长步长
```

### 方案B：增加模型容量

```python
# configs/umc_IEMOCAP-DA.py
'base_dim': [256],               # 增加特征维度
'self_attention_layers': 3,      # 增加注意力层数
'num_pretrain_epochs': [150],    # 增加预训练轮数
```

### 方案C：强化聚类约束

```python
# configs/umc_IEMOCAP-DA.py
'clustering_weight': 1.5,        # 增加聚类损失权重
'compactness_weight': 1.2,        # 增加紧密度权重
'separation_weight': 1.2,         # 增加分离度权重
```

---

## 八、分析步骤总结

1. **数据诊断**：运行诊断脚本，检查数据特征
2. **参数对比**：对比IEMOCAP-DA和MELD-DA的参数差异
3. **训练监控**：观察训练过程中的损失和指标变化
4. **特征分析**：分析学习到的特征质量
5. **对比实验**：尝试不同的参数组合
6. **逐步优化**：从保守调整开始，逐步尝试更激进的改进

---

## 九、预期改进效果

### 参数调整预期

| 调整 | 预期效果 | 风险 |
|------|---------|------|
| 降低学习率 | 训练更稳定，可能提升性能 | 训练可能变慢 |
| 提高温度 | 学习更平滑的分布 | 可能降低区分度 |
| 增加base_dim | 更强的特征表示能力 | 内存占用增加 |
| 调整阈值策略 | 更保守的聚类策略 | 可能收敛更慢 |

---

## 十、下一步行动

1. **立即尝试**：降低学习率到2e-4，提高温度到20
2. **运行诊断**：使用诊断脚本分析数据特征
3. **监控训练**：仔细观察训练过程，记录关键指标
4. **对比实验**：尝试不同的参数组合
5. **分析结果**：对比不同配置的效果

需要我帮您创建诊断脚本或修改配置文件吗？

