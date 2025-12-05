# UMC模型完整流程图

## 1. 整体架构流程

```
输入数据 → 预处理 → 特征提取 → 多模态融合 → 聚类优化 → 损失计算 → 输出结果
```

## 2. 详细数据流程图

### 阶段1: 输入预处理
```
原始输入:
├── text_feats: (B, L, text_dim)     # 文本特征
├── video_feats: (B, L, video_dim)   # 视频特征  
└── audio_feats: (B, L, audio_dim)   # 音频特征

预处理:
├── 数据类型转换: float32
├── 设备迁移: GPU/CPU
└── 参数解析: mode, labels
```

### 阶段2: 基础特征提取
```
文本处理:
text_feats → BERT编码 → text_bert (B, L, t_dim)
text_bert → 线性层 → text_feat (B, L, base_dim)
text_feat → LayerNorm → 标准化文本特征

视频处理:
video_feats → 线性层 → video_seq (B, L, base_dim)
video_seq → LayerNorm → 标准化视频特征

音频处理:
audio_feats → 线性层 → audio_seq (B, L, base_dim)
audio_seq → LayerNorm → 标准化音频特征
```

### 阶段3: ConFEDE双投影处理 (创新点1)
```
视频双投影:
video_seq → VideoDualProjector
├── simi_proj: 提取主要信息 (内容、动作)
└── dissimi_proj: 提取环境信息 (光照、背景)
→ 融合 → 增强视频特征

音频双投影:
audio_seq → AudioDualProjector  
├── simi_proj: 提取语音内容 (情感、语义)
└── dissimi_proj: 提取环境信息 (噪音、音质)
→ 融合 → 增强音频特征
```

### 阶段4: 多层注意力机制
```
交叉注意力:
text_feat_t (L, B, base_dim) ←→ video_seq_t (L, B, base_dim)
text_feat_t (L, B, base_dim) ←→ audio_seq_t (L, B, base_dim)
→ x_video, x_audio

文本引导注意力:
text_feat_t → text_guided_video_attn → text_guided_video
text_feat_t → text_guided_audio_attn → text_guided_audio

自注意力融合:
[text_feat_t, text_guided_video, text_guided_audio] → 自注意力层
→ attended_features → 特征分离
```

### 阶段5: 特征池化和交互
```
注意力池化:
text_bert_cls (B, t_dim) → 投影 → text_bert_proj (B, base_dim)
enhanced_video_feat → attention_pooling(text_bert_proj) → text_video_enh_pooled
enhanced_audio_feat → attention_pooling(text_bert_proj) → text_audio_enh_pooled

特征交互:
[text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled] 
→ feature_interaction → 交互后特征

门控融合:
交互后特征 → gated_fusion → enhanced_text
enhanced_text → LayerNorm → 最终融合特征
```

### 阶段6: 聚类优化路径 (创新点2)
```
全局特征提取:
├── text_global: text_bert_cls (B, t_dim)
├── video_global: video_feats.mean(dim=1) (B, video_dim)
└── audio_global: audio_feats.mean(dim=1) (B, audio_dim)

聚类投影:
├── text_global → ClusteringProjector → text_clustered (B, base_dim)
├── video_global → ClusteringProjector → video_clustered (B, base_dim)
└── audio_global → ClusteringProjector → audio_clustered (B, base_dim)

聚类融合:
[text_clustered, video_clustered, audio_clustered] 
→ ClusteringFusion → clustering_features (B, base_dim)

特征融合:
features + clustering_features × clustering_weight → 最终特征
```

### 阶段7: 损失计算
```
对比学习:
features → contrastive_proj → contrastive_features
contrastive_features + labels → enhanced_contrastive_loss

聚类损失:
clustering_features + labels → enhanced_clustering_loss
├── compactness_loss: 类内紧密度
└── separation_loss: 类间分离度
```

### 阶段8: 输出结果
```
根据mode返回不同结果:

mode='features':
→ features (B, base_dim)

mode='clustering-features':  
→ clustering_features (B, base_dim)

mode='train-mm':
→ features, mlp_output, contrastive_loss, clustering_loss, compactness_loss, separation_loss

mode='pretrain-mm':
→ mlp_output, contrastive_loss, clustering_loss, compactness_loss, separation_loss
```

## 3. 训练流程

### 聚类过程 (K-means++)
```
特征提取 → K-means++初始化 → 聚类中心计算 → 伪标签生成
```

### 训练循环
```
Epoch 0: K-means++初始化聚类中心
Epoch 1+: 使用上一轮聚类中心

每个Epoch:
├── 聚类 → 伪标签
├── 监督学习 (高质量样本)
├── 无监督学习 (低质量样本)  
└── 损失更新
```

## 4. 关键创新点

### 创新点1: ConFEDE机制
- **目的**: 从单一模态中提取相似和相异信息
- **实现**: VideoDualProjector + AudioDualProjector
- **效果**: 增强特征表达能力

### 创新点2: 聚类优化路径
- **目的**: 专门优化聚类任务的特征
- **实现**: ClusteringProjector + ClusteringFusion
- **效果**: 提升聚类质量

### 创新点3: 多层注意力融合
- **目的**: 深度理解多模态交互
- **实现**: 交叉注意力 + 文本引导注意力 + 自注意力
- **效果**: 更好的多模态融合

## 5. 数据维度变化

```
输入: (B, L, text_dim), (B, L, video_dim), (B, L, audio_dim)
↓
中间: (B, L, base_dim) × 3
↓  
池化: (B, base_dim) × 3
↓
融合: (B, base_dim)
↓
输出: (B, base_dim) + 各种损失
```

这个流程图展示了UMC模型的完整数据处理过程，从原始多模态输入到最终的聚类优化特征输出。
