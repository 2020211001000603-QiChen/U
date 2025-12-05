# UMC项目完整代码解释和运行指南

## 项目概述

UMC (Unsupervised Multimodal Clustering) 是一个用于多模态语义发现的无监督聚类方法。该项目实现了三个主要创新点：

1. **ConFEDE双投影机制** - 对视频和音频模态分别应用相似性和相异性投影
2. **文本引导注意力和自注意力机制** - 以文本为锚点的交叉注意力和多层自注意力
3. **聚类优化架构** - 专门的聚类投影器、融合器和损失函数

## 项目结构详解

### 1. 入口点和运行方式

#### 主入口文件：`run.py`
这是整个项目的核心入口文件，包含以下主要功能：

**命令行参数解析：**
```python
# 基本参数
--dataset: 数据集名称 (MIntRec, MELD-DA, IEMOCAP-DA)
--method: 方法名称 (umc)
--multimodal_method: 多模态方法 (umc)
--text_backbone: 文本骨干网络 (bert)
--seed: 随机种子
--gpu_id: GPU设备ID

# 数据路径
--data_path: 数据根目录
--video_feats_path: 视频特征文件
--audio_feats_path: 音频特征文件

# 运行模式
--train: 是否训练
--tune: 是否调参
--save_model: 是否保存模型
--save_results: 是否保存结果

# 消融实验
--ablation_experiment: 消融实验名称
--run_all_ablation: 运行所有消融实验
```

**运行流程：**
```python
if __name__ == '__main__':
    args = parse_arguments()
    
    if args.run_all_ablation:
        # 运行所有消融实验
        run_all_ablation_experiments(args)
    elif args.ablation_experiment:
        # 运行单个消融实验
        apply_ablation_config(args, args.ablation_experiment)
        # 执行训练和测试
    else:
        # 正常运行
        param = ParamManager(args)
        data = DataManager(args)
        logger = set_logger(args)
        run(args, data, logger)
```

### 2. 配置文件系统

#### 主要配置文件：`configs/umc_MIntRec.py`

**超参数配置：**
```python
hyper_parameters = {
    # 基础训练参数
    'pretrain_batch_size': 128,
    'train_batch_size': 128,
    'num_pretrain_epochs': 100,
    'num_train_epochs': 100,
    'lr_pre': 2e-5,
    'lr': [3e-4],
    
    # UMC特定参数
    'base_dim': 256,           # 基础特征维度
    'nheads': 8,               # 注意力头数
    'encoder_layers_1': 1,     # 编码器层数
    
    # 创新点开关
    'enable_video_dual': True,      # 视频双投影
    'enable_audio_dual': True,      # 音频双投影
    'enable_text_guided_attention': True,  # 文本引导注意力
    'enable_self_attention': True,         # 自注意力
    'enable_clustering_optimization': True, # 聚类优化
    
    # 渐进式学习参数
    'delta': [0.02],           # 阈值增长步长
    'thres': [0.05],           # 初始阈值
    'max_threshold': 0.5,      # 最大阈值
    'min_threshold': 0.05,     # 最小阈值
}
```

**消融实验配置：**
```python
ablation_configs = {
    'baseline_traditional': {
        # 禁用所有创新点
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_clustering_optimization': False,
    },
    'full_confede': {
        # 启用所有ConFEDE功能
        'enable_video_dual': True,
        'enable_audio_dual': True,
        'enable_text_guided_attention': True,
        'enable_self_attention': True,
    },
    # ... 更多消融实验配置
}
```

### 3. 核心模块详解

#### 3.1 数据管理模块 (`data/`)

**DataManager类：**
- 负责加载和预处理多模态数据
- 支持文本、视频、音频三种模态
- 提供数据加载器和批处理功能

**数据预处理：**
```python
# 文本预处理
text_preprocessor = TextPreprocessor()
text_features = text_preprocessor.process(text_data)

# 视频预处理  
video_preprocessor = VideoPreprocessor()
video_features = video_preprocessor.process(video_data)

# 音频预处理
audio_preprocessor = AudioPreprocessor()
audio_features = audio_preprocessor.process(audio_data)
```

#### 3.2 模型架构模块 (`backbones/`)

**FusionNets/UMC.py - 核心模型：**
```python
class UMC(nn.Module):
    def __init__(self, args):
        # 基础编码器
        self.text_embedding = BERTEncoder(args)
        
        # 特征投影层
        self.text_layer = nn.Linear(args.text_feat_dim, base_dim)
        self.video_layer = nn.Linear(args.video_feat_dim, base_dim)
        self.audio_layer = nn.Linear(args.audio_feat_dim, base_dim)
        
        # ConFEDE双投影模块
        if self.enable_video_dual:
            self.video_dual_projector = VideoDualProjector(...)
        if self.enable_audio_dual:
            self.audio_dual_projector = AudioDualProjector(...)
        
        # 注意力机制
        self.text_guided_video_attn = MultiheadAttention(...)
        self.text_guided_audio_attn = MultiheadAttention(...)
        
        # 聚类优化模块
        if self.use_clustering_projector:
            self.clustering_projector = ClusteringProjector(...)
        if self.use_clustering_fusion:
            self.clustering_fusion = ClusteringFusion(...)
```

**前向传播流程：**
1. 输入预处理和特征提取
2. ConFEDE双投影处理
3. 交叉注意力计算
4. 文本引导注意力
5. 自注意力层处理
6. 特征池化和交互
7. 门控融合
8. 聚类优化
9. 损失计算

#### 3.3 训练管理模块 (`methods/unsupervised/UMC/`)

**Manager类 - 训练控制器：**
```python
class UMCManager:
    def __init__(self, args, data, model):
        self.args = args
        self.data = data
        self.model = model
        
        # 渐进式学习优化器
        self.progressive_learning = AdaptiveProgressiveLearning(
            initial_threshold=args.thres,
            max_threshold=args.max_threshold,
            min_threshold=args.min_threshold
        )
    
    def _train(self, args):
        """训练主循环"""
        for epoch in range(args.num_train_epochs):
            # 计算自适应阈值
            threshold = self.progressive_learning.compute_threshold(
                epoch, args.num_train_epochs, 
                current_performance, current_loss
            )
            
            # 训练一个epoch
            self._train_epoch(epoch, threshold)
            
            # 评估性能
            performance = self._evaluate()
            
            # 早停检查
            if self._should_early_stop():
                break
```

**渐进式学习策略：**
```python
class AdaptiveProgressiveLearning:
    def compute_threshold(self, epoch, total_epochs, performance, loss):
        # 1. 基础阈值（S型曲线增长）
        base_threshold = self._compute_base_threshold(epoch, total_epochs)
        
        # 2. 性能自适应调整
        performance_adjustment = self._compute_performance_adjustment()
        
        # 3. 损失自适应调整
        loss_adjustment = self._compute_loss_adjustment()
        
        # 4. 综合计算
        adaptive_threshold = base_threshold + performance_adjustment + loss_adjustment
        
        return np.clip(adaptive_threshold, self.min_threshold, self.max_threshold)
```

#### 3.4 损失函数模块 (`losses/`)

**对比学习损失：**
```python
class ContrastiveLoss(nn.Module):
    def forward(self, features, labels=None):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        if labels is not None:
            # 监督对比学习
            mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
            positives = similarity_matrix * mask
            negatives = similarity_matrix * (1 - mask)
            loss = -positives + negatives
        else:
            # 自监督对比学习
            logits = similarity_matrix
            labels = torch.arange(features.size(0))
            loss = F.cross_entropy(logits, labels)
        
        return loss.mean()
```

**聚类损失：**
```python
class ClusteringLoss(nn.Module):
    def forward(self, features, labels, centroids=None):
        # 紧密度损失：同类样本聚集
        compactness_loss = self._compute_compactness_loss(features, labels, centroids)
        
        # 分离度损失：不同类样本分离
        separation_loss = self._compute_separation_loss(centroids)
        
        total_loss = self.compactness_weight * compactness_loss + self.separation_weight * separation_loss
        return total_loss, compactness_loss, separation_loss
```

### 4. 运行脚本分析

#### 4.1 基础运行脚本：`examples/run_umc.sh`

```bash
#!/usr/bin/bash

# 运行UMC模型
python run.py \
--dataset MIntRec \                    # 数据集
--data_path 'Datasets' \              # 数据路径
--multimodal_method umc \             # 多模态方法
--method umc \                        # 方法名称
--train \                             # 训练模式
--tune \                              # 调参模式
--save_results \                      # 保存结果
--seed 0 \                            # 随机种子
--gpu_id '0' \                        # GPU设备
--video_feats_path 'swin_feats.pkl' \ # 视频特征文件
--audio_feats_path 'wavlm_feats.pkl' \# 音频特征文件
--text_backbone bert-base-uncased \   # 文本骨干网络
--config_file_name umc_MIntRec \      # 配置文件
--output_path "outputs/MIntRec/umc"    # 输出路径
```

#### 4.2 消融实验脚本

**单个消融实验：**
```bash
python run.py \
--dataset MIntRec \
--ablation_experiment baseline_traditional \  # 消融实验名称
--results_file_name "results_baseline.csv" \
--output_path "outputs/MIntRec/baseline"
```

**运行所有消融实验：**
```bash
python run.py \
--dataset MIntRec \
--run_all_ablation \                  # 运行所有消融实验
--results_file_name "results_all_ablation.csv"
```

### 5. 完整运行指南

#### 5.1 环境准备

**1. 创建Python环境：**
```bash
conda create --name umc python=3.8
conda activate umc
```

**2. 安装依赖：**
```bash
# 安装PyTorch
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
pip install -r requirements.txt
```

**3. 下载数据和模型：**
```bash
# 下载多模态特征
# 从百度云或Google Drive下载特征文件

# 下载预训练BERT模型
# 从百度云下载BERT模型
```

#### 5.2 数据准备

**目录结构：**
```
UMC-main/
├── Datasets/
│   ├── MIntRec/
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   ├── video_data/
│   │   │   └── swin_feats.pkl
│   │   └── audio_data/
│   │       └── wavlm_feats.pkl
│   ├── MELD-DA/
│   └── IEMOCAP-DA/
├── cache/                    # BERT模型缓存
├── logs/                     # 日志文件
├── outputs/                  # 输出结果
└── models/                   # 保存的模型
```

#### 5.3 运行实验

**1. 基础实验：**
```bash
# 运行完整UMC模型
sh examples/run_umc.sh

# 或者直接使用Python
python run.py --dataset MIntRec --train --save_results
```

**2. 消融实验：**
```bash
# 运行单个消融实验
python run.py --dataset MIntRec --ablation_experiment baseline_traditional --train --save_results

# 运行所有消融实验
python run.py --dataset MIntRec --run_all_ablation --save_results
```

**3. 不同数据集：**
```bash
# MIntRec数据集
python run.py --dataset MIntRec --config_file_name umc_MIntRec

# MELD-DA数据集  
python run.py --dataset MELD-DA --config_file_name umc_MELD-DA

# IEMOCAP-DA数据集
python run.py --dataset IEMOCAP-DA --config_file_name umc_IEMOCAP-DA
```

#### 5.4 参数调优

**修改配置文件：**
```python
# configs/umc_MIntRec.py
hyper_parameters = {
    'lr': [1e-4, 3e-4, 5e-4],        # 学习率调优
    'train_batch_size': [64, 128, 256], # 批次大小调优
    'base_dim': [128, 256, 512],      # 特征维度调优
    'nheads': [4, 8, 16],             # 注意力头数调优
}
```

**运行调参：**
```bash
python run.py --dataset MIntRec --train --tune --save_results
```

### 6. 输出和结果

#### 6.1 日志文件
- 位置：`logs/`
- 格式：`{method}_{dataset}_{seed}_{timestamp}.log`
- 内容：训练过程、参数记录、性能指标

#### 6.2 模型文件
- 位置：`models/`
- 格式：`{method}_{dataset}_{seed}.pkl`
- 内容：训练好的模型权重

#### 6.3 结果文件
- 位置：`outputs/`
- 格式：`results.csv`
- 内容：聚类指标（NMI, ARI, ACC, FMI）

#### 6.4 性能指标
```python
# 聚类评估指标
metrics = {
    'NMI': normalized_mutual_info_score(true_labels, pred_labels),
    'ARI': adjusted_rand_score(true_labels, pred_labels), 
    'ACC': clustering_accuracy(true_labels, pred_labels),
    'FMI': fowlkes_mallows_score(true_labels, pred_labels)
}
```

### 7. 常见问题和解决方案

#### 7.1 内存不足
```python
# 减少批次大小
'train_batch_size': 64,
'eval_batch_size': 64,

# 减少特征维度
'base_dim': 128,
```

#### 7.2 GPU显存不足
```python
# 使用梯度累积
'gradient_accumulation_steps': 2,

# 减少序列长度
'max_seq_length': 128,
```

#### 7.3 训练不收敛
```python
# 调整学习率
'lr': [1e-5, 2e-5, 5e-5],

# 增加训练轮数
'num_train_epochs': 200,

# 调整损失权重
'clustering_weight': 0.5,
'contrastive_weight': 1.0,
```

### 8. 扩展和修改

#### 8.1 添加新的消融实验
```python
# 在run.py中添加新的消融配置
'ablation_new_experiment': {
    'enable_video_dual': False,
    'enable_audio_dual': True,
    # ... 其他配置
}
```

#### 8.2 修改模型架构
```python
# 在UMC.py中添加新的模块
class NewModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.layer(x)
```

#### 8.3 添加新的损失函数
```python
# 在losses/中添加新的损失
class NewLoss(nn.Module):
    def forward(self, features, labels):
        # 实现新的损失计算
        return loss
```

## 总结

UMC项目是一个完整的多模态无监督聚类框架，具有以下特点：

1. **模块化设计** - 清晰的模块分离，便于理解和修改
2. **消融实验支持** - 完整的消融实验框架，验证各组件贡献
3. **渐进式学习** - 自适应阈值调整，提升训练效果
4. **多数据集支持** - 支持MIntRec、MELD-DA、IEMOCAP-DA等数据集
5. **完整的评估** - 多种聚类指标评估

通过这个详细的代码解释，您可以：
- 理解整个项目的架构和流程
- 知道如何运行基础实验和消融实验
- 了解如何修改和扩展代码
- 解决常见的运行问题

建议从基础实验开始，逐步尝试消融实验，最后进行自定义修改。
