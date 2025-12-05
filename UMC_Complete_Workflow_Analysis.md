# UMC项目完整运行流程详解

## 🚀 项目整体流程概览

UMC项目是一个完整的多模态无监督聚类系统，从数据准备到最终结果输出，包含以下主要阶段：

```
数据准备 → 模型初始化 → 预训练 → 主训练 → 测试评估 → 结果输出
    ↓         ↓         ↓       ↓       ↓        ↓
  数据加载   参数配置   对比学习  聚类训练  性能评估  指标保存
```

## 📊 详细流程分析

### 阶段1：数据准备和预处理 (DataManager)

#### 1.1 数据加载流程
```python
# 入口：run.py → DataManager(args)
class DataManager:
    def __init__(self, args):
        # 1. 获取数据集配置
        bm = benchmarks[args.dataset]  # MIntRec, MELD-DA, IEMOCAP-DA
        max_seq_lengths, feat_dims = bm['max_seq_lengths'], bm['feat_dims']
        
        # 2. 设置序列长度和特征维度
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = max_seq_lengths
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims
        
        # 3. 加载多模态数据
        self.mm_data, self.train_outputs = get_data(args, self.logger)
```

#### 1.2 多模态数据处理
```python
def get_data(args, logger):
    # 1. 读取标签和索引
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, 'train.tsv')
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, 'dev.tsv')
    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, 'test.tsv')
    
    # 2. 合并训练和验证数据
    train_data_index = train_data_index + dev_data_index
    train_label_ids = train_label_ids + dev_label_ids
    
    # 3. 加载三种模态数据
    text_data = get_t_data(args, data_args)      # BERT文本特征
    video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)  # Swin视频特征
    audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)   # WavLM音频特征
    
    # 4. 创建多模态数据集
    mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
    mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])
    
    return {'train': mm_train_data, 'test': mm_test_data}, train_outputs
```

#### 1.3 数据格式说明
```python
# 每个样本的数据结构
sample = {
    'text_feats': torch.tensor([...]),      # BERT tokenized features
    'video_feats': torch.tensor([...]),     # Swin video features  
    'video_lengths': torch.tensor([...]),   # Video sequence lengths
    'audio_feats': torch.tensor([...]),     # WavLM audio features
    'audio_lengths': torch.tensor([...]),   # Audio sequence lengths
    'label_ids': torch.tensor([...])        # Ground truth labels (f0or evaluation)
}
```

### 阶段2：模型初始化和配置

#### 2.1 参数管理器初始化
```python
# ParamManager负责加载和配置所有超参数
param = ParamManager(args)
args = param.args

# 主要参数包括：
hyper_parameters = {
    # 基础训练参数
    'pretrain_batch_size': 128,
    'train_batch_size': 128,
    'num_pretrain_epochs': 100,
    'num_train_epochs': 100,
    'lr_pre': 2e-5,
    'lr': [3e-4],
    
    # UMC特定参数
    'base_dim': 256,           # 统一特征维度
    'nheads': 8,               # 注意力头数
    'encoder_layers_1': 1,     # Transformer编码器层数
    
    # 创新点开关
    'enable_video_dual': True,      # ConFEDE视频双投影
    'enable_audio_dual': True,      # ConFEDE音频双投影
    'enable_text_guided_attention': True,  # 文本引导注意力
    'enable_self_attention': True,         # 自注意力机制
    'enable_clustering_optimization': True, # 聚类优化
    
    # 渐进式学习参数
    'delta': [0.02],           # 阈值增长步长
    'thres': [0.05],          # 初始阈值
    'max_threshold': 0.5,     # 最大阈值
    'min_threshold': 0.05,    # 最小阈值
}
```

#### 2.2 模型架构初始化
```python
# ModelManager负责创建UMC模型
model = ModelManager(args)

# UMC模型初始化过程：
class UMC(nn.Module):
    def __init__(self, args):
        # 1. 基础编码器
        self.text_embedding = BERTEncoder(args)
        
        # 2. 特征投影层
        self.text_layer = nn.Linear(args.text_feat_dim, base_dim)
        self.video_layer = nn.Linear(args.video_feat_dim, base_dim)
        self.audio_layer = nn.Linear(args.audio_feat_dim, base_dim)
        
        # 3. ConFEDE双投影模块（可选）
        if self.enable_video_dual:
            self.video_dual_projector = VideoDualProjector(base_dim, base_dim)
        if self.enable_audio_dual:
            self.audio_dual_projector = AudioDualProjector(base_dim, base_dim)
        
        # 4. 注意力机制
        self.cross_attn_video_layers = nn.ModuleList([...])
        self.cross_attn_audio_layers = nn.ModuleList([...])
        self.text_guided_video_attn = MultiheadAttention(...)
        self.text_guided_audio_attn = MultiheadAttention(...)
        
        # 5. 聚类优化模块（可选）
        if self.use_clustering_projector:
            self.clustering_projector = ClusteringProjector(base_dim, base_dim)
        if self.use_clustering_fusion:
            self.clustering_fusion = ClusteringFusion(base_dim, args.num_labels)
```

### 阶段3：训练管理器初始化

#### 3.1 UMCManager初始化
```python
# UMCManager负责整个训练流程控制
method_manager = method_map[args.method]  # 获取UMC管理器
method = method_manager(args, data, model)

class UMCManager:
    def __init__(self, args, data, model):
        self.args = args
        self.data = data
        self.model = model
        
        # 渐进式学习优化器
        self.progressive_learning = AdaptiveProgressiveLearning(
            initial_threshold=args.thres,
            max_threshold=args.max_threshold,
            min_threshold=args.min_threshold,
            performance_window=3,
            patience=5
        )
        
        # 预训练管理器
        self.pretrain_manager = PretrainUMCManager(args, data, model)
```

#### 3.2 渐进式学习策略
```python
class AdaptiveProgressiveLearning:
    def compute_threshold(self, epoch, total_epochs, current_performance, current_loss):
        """计算自适应阈值"""
        # 1. 记录历史性能
        self._record_history(epoch, current_performance, current_loss)
        
        # 2. 计算基础阈值（S型曲线增长）
        base_threshold = self._compute_base_threshold(epoch, total_epochs)
        
        # 3. 性能自适应调整
        performance_adjustment = self._compute_performance_adjustment()
        
        # 4. 损失自适应调整
        loss_adjustment = self._compute_loss_adjustment()
        
        # 5. 稳定性调整
        stability_adjustment = self._compute_stability_adjustment()
        
        # 6. 综合计算最终阈值
        adaptive_threshold = (base_threshold + 
                            performance_adjustment + 
                            loss_adjustment + 
                            stability_adjustment)
        
        return np.clip(adaptive_threshold, self.min_threshold, self.max_threshold)
```

### 阶段4：预训练阶段 (Pretrain)

#### 4.1 预训练流程
```python
def _train(self, args):
    """主训练函数"""
    if args.pretrain:
        logger.info('Pretraining begins...')
        self.pretrain_manager._train(args)  # 执行预训练
        logger.info('Pretraining is finished...')
    
    # 主训练阶段
    logger.info('Training begins...')
    self._train_main(args)
    logger.info('Training is finished...')
```

#### 4.2 预训练目标
```python
# 预训练阶段主要目标：
# 1. 对比学习：学习多模态特征表示
# 2. 特征对齐：对齐不同模态的特征空间
# 3. 初始化：为后续聚类训练提供良好的初始化

# 预训练损失函数
def pretrain_loss(self, features, labels):
    # 对比学习损失
    contrastive_loss = self.contrastive_loss_fn(features, labels)
    
    # 特征对齐损失
    alignment_loss = self.alignment_loss_fn(features)
    
    total_loss = contrastive_loss + alignment_loss
    return total_loss
```

### 阶段5：主训练阶段 (Main Training)

#### 5.1 训练主循环
```python
def _train_main(self, args):
    """主训练循环"""
    for epoch in range(args.num_train_epochs):
        # 1. 计算自适应阈值
        threshold = self.progressive_learning.compute_threshold(
            epoch, args.num_train_epochs, 
            current_performance, current_loss
        )
        
        # 2. 训练一个epoch
        epoch_loss = self._train_epoch(epoch, threshold)
        
        # 3. 评估性能
        performance = self._evaluate()
        
        # 4. 早停检查
        if self.progressive_learning.should_early_stop():
            logger.info(f'Early stopping at epoch {epoch}')
            break
        
        # 5. 保存最佳模型
        if performance > self.best_performance:
            self.best_performance = performance
            save_model(self.model, args.model_output_path)
```

#### 5.2 单epoch训练流程
```python
def _train_epoch(self, epoch, threshold):
    """训练一个epoch"""
    self.model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(self.train_dataloader):
        # 1. 数据准备
        text_feats = batch['text_feats'].to(self.device)
        video_feats = batch['video_feats'].to(self.device)
        audio_feats = batch['audio_feats'].to(self.device)
        labels = batch['label_ids'].to(self.device)
        
        # 2. 前向传播
        features, mlp_output, contrastive_loss, clustering_loss = self.model(
            text_feats, video_feats, audio_feats, 
            mode='train-mm', labels=labels
        )
        
        # 3. 损失计算
        total_loss = (
            contrastive_loss * self.contrastive_weight +
            clustering_loss * self.clustering_weight
        )
        
        # 4. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 5. 记录损失
        if batch_idx % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')
    
    return total_loss.item()
```

#### 5.3 模型前向传播详解
```python
def forward(self, text_feats, video_feats, audio_feats, mode=None, labels=None):
    """UMC模型前向传播"""
    # 1. 特征提取和归一化
    text_bert = self.text_embedding(text_feats)  # BERT编码
    text_feat = self.text_feat_layer(text_bert)  # 投影到base_dim
    text_feat = self.ln_text(text_feat)
    video_seq = self.ln_video(self.video_layer(video_feats))
    audio_seq = self.ln_audio(self.audio_layer(audio_feats))
    
    # 2. ConFEDE双投影处理（创新点一）
    if self.enable_video_dual:
        video_seq = self.video_dual_projector(video_seq)
    if self.enable_audio_dual:
        audio_seq = self.audio_dual_projector(audio_seq)
    
    # 3. 交叉注意力机制
    text_feat_t = text_feat.permute(1, 0, 2)
    video_seq_t = video_seq.permute(1, 0, 2)
    audio_seq_t = audio_seq.permute(1, 0, 2)
    
    x_video = text_feat_t
    x_audio = text_feat_t
    
    for layer in self.cross_attn_video_layers:
        x_video, _ = layer(x_video, video_seq_t, video_seq_t)
    for layer in self.cross_attn_audio_layers:
        x_audio, _ = layer(x_audio, audio_seq_t, audio_seq_t)
    
    # 4. 文本引导注意力（创新点二）
    if self.enable_text_guided_attention:
        text_guided_video, _ = self.text_guided_video_attn(text_feat_t, x_video, x_video)
        text_guided_audio, _ = self.text_guided_audio_attn(text_feat_t, x_audio, x_audio)
    else:
        text_guided_video = x_video
        text_guided_audio = x_audio
    
    # 5. 自注意力层（创新点二）
    combined_features = torch.cat([text_feat_t, text_guided_video, text_guided_audio], dim=0)
    attended_features = combined_features
    if self.self_attention_layers:
        for self_attn_layer in self.self_attention_layers:
            attended_features, _ = self_attn_layer(
                attended_features, attended_features, attended_features
            )
            attended_features = attended_features + combined_features
    
    # 6. 特征分离和归一化
    seq_len = text_feat_t.shape[0]
    enhanced_text_feat = attended_features[:seq_len]
    enhanced_video_feat = attended_features[seq_len:2*seq_len]
    enhanced_audio_feat = attended_features[2*seq_len:]
    
    enhanced_text_feat = self.post_attention_norm(enhanced_text_feat)
    enhanced_video_feat = self.post_attention_norm(enhanced_video_feat)
    enhanced_audio_feat = self.post_attention_norm(enhanced_audio_feat)
    
    # 7. 特征池化和交互
    text_bert_cls = text_bert[:, 0]
    text_bert_proj = self.bert_text_layer(text_bert_cls)
    
    if self.use_attention_pooling:
        text_video_enh_pooled = self.attention_pooling(enhanced_video_feat, text_bert_proj)
        text_audio_enh_pooled = self.attention_pooling(enhanced_audio_feat, text_bert_proj)
    else:
        text_video_enh_pooled = enhanced_video_feat.mean(dim=0)
        text_audio_enh_pooled = enhanced_audio_feat.mean(dim=0)
    
    # 8. 特征交互
    interaction_input = torch.stack([text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled], dim=1)
    interaction_input = interaction_input.transpose(0, 1)
    interacted_features, _ = self.feature_interaction(interaction_input, interaction_input, interaction_input)
    interacted_features = interacted_features.transpose(0, 1)
    
    text_bert_proj = interacted_features[:, 0]
    text_video_enh_pooled = interacted_features[:, 1]
    text_audio_enh_pooled = interacted_features[:, 2]
    
    # 9. 门控融合
    if self.enable_gated_fusion:
        enhanced_text = self.gated_fusion(text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled)
    else:
        enhanced_text = torch.cat([text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled], dim=-1)
        enhanced_text = self.fusion_layer(enhanced_text)
    
    # 10. 聚类优化路径（创新点三）
    clustering_features = enhanced_text
    clustering_loss = None
    
    if self.enable_clustering_optimization:
        if self.use_clustering_projector:
            clustering_features, center_features = self.clustering_projector(enhanced_text)
        if self.use_clustering_fusion:
            clustering_features, cluster_weights = self.clustering_fusion(clustering_features)
        
        if labels is not None:
            clustering_loss, compactness_loss, separation_loss = self.clustering_loss_fn(
                clustering_features, labels
            )
    
    # 11. 对比学习
    contrastive_features = self.contrastive_proj(enhanced_text)
    contrastive_loss = None
    if labels is not None:
        contrastive_loss = self.contrastive_loss(contrastive_features, labels)
    
    # 12. 返回结果
    if mode == 'train-mm':
        mlp_output = self.shared_embedding_layer(enhanced_text)
        return enhanced_text, mlp_output, contrastive_loss, clustering_loss
    else:
        return enhanced_text
```

### 阶段6：测试和评估

#### 6.1 测试流程
```python
def _test(self, args):
    """测试阶段"""
    logger.info('Testing begins...')
    
    # 1. 加载最佳模型
    restore_model(self.model, args.model_output_path)
    
    # 2. 提取特征
    features = self._extract_features(args)
    
    # 3. 聚类
    cluster_labels = self._cluster_features(features)
    
    # 4. 评估
    metrics = self._evaluate_clustering(cluster_labels)
    
    logger.info('Testing is finished...')
    return metrics
```

#### 6.2 特征提取
```python
def _extract_features(self, args):
    """提取测试集特征"""
    self.model.eval()
    features_list = []
    
    with torch.no_grad():
        for batch in self.test_dataloader:
            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            
            # 提取特征
            features = self.model(text_feats, video_feats, audio_feats, mode='features')
            features_list.append(features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)
```

#### 6.3 聚类和评估
```python
def _cluster_features(self, features):
    """对特征进行聚类"""
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=self.args.num_labels, random_state=self.args.seed)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def _evaluate_clustering(self, cluster_labels):
    """评估聚类结果"""
    true_labels = self.test_labels
    
    # 计算聚类指标
    metrics = clustering_score(true_labels, cluster_labels)
    
    return {
        'NMI': metrics['NMI'],
        'ARI': metrics['ARI'], 
        'ACC': metrics['ACC'],
        'FMI': metrics['FMI']
    }
```

### 阶段7：消融实验流程

#### 7.1 消融实验配置
```python
# 在run.py中定义消融实验
ablation_configs = {
    'baseline_traditional': {
        # 禁用所有创新点
        'enable_video_dual': False,
        'enable_audio_dual': False,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_clustering_optimization': False,
    },
    'confede_only': {
        # 仅启用ConFEDE双投影
        'enable_video_dual': True,
        'enable_audio_dual': True,
        'enable_text_guided_attention': False,
        'enable_self_attention': False,
        'enable_clustering_optimization': False,
    },
    'full_umc_model': {
        # 启用所有创新点
        'enable_video_dual': True,
        'enable_audio_dual': True,
        'enable_text_guided_attention': True,
        'enable_self_attention': True,
        'enable_clustering_optimization': True,
    },
    # ... 更多消融实验配置
}
```

#### 7.2 消融实验执行
```python
def run_all_ablation_experiments(args):
    """运行所有消融实验"""
    ablation_experiments = [
        'baseline_traditional', 'confede_only', 'text_guided_only', 
        'gated_fusion_only', 'full_confede', 'full_umc_model'
    ]
    
    results = {}
    
    for experiment_name in ablation_experiments:
        # 1. 创建实验特定的args
        exp_args = copy.deepcopy(args)
        exp_args.ablation_experiment = experiment_name
        
        # 2. 应用消融实验配置
        apply_ablation_config(exp_args, experiment_name)
        
        # 3. 执行训练和测试
        try:
            param = ParamManager(exp_args)
            data = DataManager(exp_args)
            logger = set_logger(exp_args)
            
            work(exp_args, data, logger)
            results[experiment_name] = 'success'
        except Exception as e:
            results[experiment_name] = f'failed: {str(e)}'
    
    return results
```

### 阶段8：结果输出和保存

#### 8.1 结果保存
```python
def save_results(args, outputs, debug_args=None):
    """保存实验结果"""
    results_file = os.path.join(args.results_path, args.results_file_name)
    
    # 写入CSV文件
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 写入实验参数
        row = [
            args.dataset,
            args.method,
            args.ablation_experiment if hasattr(args, 'ablation_experiment') else 'baseline',
            args.seed,
            outputs['NMI'],
            outputs['ARI'],
            outputs['ACC'],
            outputs['FMI']
        ]
        writer.writerow(row)
```

#### 8.2 日志记录
```python
def set_logger(args):
    """设置日志记录"""
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 文件日志
    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    # 控制台日志
    ch = logging.StreamHandler()
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger
```

## 🔄 完整运行流程总结

### 命令行执行流程
```bash
# 1. 基础实验
python run.py --dataset MIntRec --train --save_results

# 2. 消融实验
python run.py --dataset MIntRec --ablation_experiment baseline_traditional --train --save_results

# 3. 运行所有消融实验
python run.py --dataset MIntRec --run_all_ablation --save_results

# 4. 使用脚本运行
sh examples/run_umc.sh
```

### 程序执行流程
```
1. 参数解析 (parse_arguments)
   ↓
2. 消融实验配置 (apply_ablation_config)
   ↓
3. 参数管理器初始化 (ParamManager)
   ↓
4. 数据管理器初始化 (DataManager)
   ↓
5. 日志设置 (set_logger)
   ↓
6. 模型初始化 (ModelManager)
   ↓
7. 训练管理器初始化 (UMCManager)
   ↓
8. 预训练阶段 (PretrainUMCManager._train)
   ↓
9. 主训练阶段 (UMCManager._train_main)
   ↓
10. 测试阶段 (UMCManager._test)
    ↓
11. 结果保存 (save_results)
```

### 数据流向
```
原始数据 (TSV文件)
    ↓
数据加载 (DataManager)
    ↓
多模态数据集 (MMDataset)
    ↓
数据加载器 (DataLoader)
    ↓
批次数据 (Batch)
    ↓
模型前向传播 (UMC.forward)
    ↓
特征表示 (Features)
    ↓
聚类结果 (Cluster Labels)
    ↓
评估指标 (Metrics)
    ↓
结果保存 (CSV文件)
```

### 关键输出文件
```
logs/                    # 训练日志
├── umc_MIntRec_0_2024-01-01-12-00-00.log

models/                  # 保存的模型
├── umc_MIntRec_0.pkl

outputs/                 # 输出结果
├── results.csv          # 聚类指标结果
├── features.npy         # 提取的特征
└── cluster_labels.npy   # 聚类标签
```

这个完整的流程展示了UMC项目从数据准备到最终结果输出的全过程，每个阶段都有明确的职责和输出，形成了一个完整的多模态无监督聚类系统。
