import torch
import torch.nn.functional as F
from losses import loss_map
from ..SubNets.FeatureNets import BERTEncoder, SubNet, RoBERTaEncoder, AuViSubNet
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from .sampler import ConvexSampler
from torch import nn
from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.transformers_encoder.multihead_attention import MultiheadAttention

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
__all__ = ['umc']

class SimpleLinearProjector(nn.Module):
    """简单线性映射替代双投影 - 用于消融实验"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(SimpleLinearProjector, self).__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.projection(x) + x  # 残差连接

class VideoDualProjector(nn.Module):
    """视频双投影模块 - 只对视频模态应用ConFEDE机制"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(VideoDualProjector, self).__init__()
        
        # 相似性投影：捕获主要信息（人物、动作、表情）
        self.simi_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 相异性投影：捕获环境信息（背景、场景、环境）
        self.dissimi_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 融合层：将双投影结果融合回原维度
        self.fusion = nn.Linear(output_dim * 2, input_dim)
        
    def forward(self, x):
        # 双投影
        simi_feat = self.simi_proj(x)      # 主要信息
        dissimi_feat = self.dissimi_proj(x) # 环境信息
        
        # 拼接并融合
        dual_feat = torch.cat([simi_feat, dissimi_feat], dim=-1)
        enhanced_feat = self.fusion(dual_feat)
        
        # 残差连接，保持信息不丢失
        return enhanced_feat + x

class AudioDualProjector(nn.Module):
    """音频双投影模块 - 只对音频模态应用ConFEDE机制"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(AudioDualProjector, self).__init__()
        
        # 相似性投影：捕获主要信息（语音内容、语调、情感）
        self.simi_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 相异性投影：捕获环境信息（背景噪音、环境音、音质）
        self.dissimi_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 融合层：将双投影结果融合回原维度
        self.fusion = nn.Linear(output_dim * 2, input_dim)
        
    def forward(self, x):
        # 双投影
        simi_feat = self.simi_proj(x)      # 主要信息
        dissimi_feat = self.dissimi_proj(x) # 环境信息
        
        # 拼接并融合
        dual_feat = torch.cat([simi_feat, dissimi_feat], dim=-1)
        enhanced_feat = self.fusion(dual_feat)
        
        # 残差连接，保持信息不丢失
        return enhanced_feat + x



class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels=None):
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        if labels is not None:
            # 创建正样本掩码
            mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
            mask = mask - torch.eye(mask.size(0)).to(mask.device)
            
            # 正样本相似度
            positives = similarity_matrix * mask
            positives = positives.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
            # 负样本相似度
            negatives = similarity_matrix * (1 - mask)
            negatives = torch.logsumexp(negatives, dim=1)
            
            # 总损失
            loss = -positives + negatives
            return loss.mean()
        else:
            # 自监督对比学习
            logits = similarity_matrix
            labels = torch.arange(features.size(0)).to(features.device)
            return F.cross_entropy(logits, labels)

class ClusteringProjector(nn.Module):
    """聚类投影器 - 专门为聚类任务设计的特征投影"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(ClusteringProjector, self).__init__()
        
        # 聚类特征投影层
        self.clustering_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 聚类中心投影层
        self.center_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, features):
        # 聚类特征投影
        clustering_features = self.clustering_proj(features)
        
        # 聚类中心投影
        center_features = self.center_proj(clustering_features)
        
        return clustering_features, center_features

class ClusteringFusion(nn.Module):
    """聚类融合器 - 专门为聚类优化的特征融合"""
    def __init__(self, input_dim, num_clusters, dropout=0.1):
        super(ClusteringFusion, self).__init__()
        
        self.num_clusters = num_clusters
        
        # 聚类权重计算
        self.cluster_weight_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_clusters),
            nn.Softmax(dim=-1)
        )
        
        # 聚类特征融合
        self.cluster_fusion = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, features):
        # 计算聚类权重
        cluster_weights = self.cluster_weight_net(features)
        
        # 聚类特征融合
        fused_features = self.cluster_fusion(features)
        
        return fused_features, cluster_weights

class ClusteringLoss(nn.Module):
    """聚类损失函数 - 包含紧密度和分离度损失"""
    def __init__(self, compactness_weight=1.0, separation_weight=1.0):
        super(ClusteringLoss, self).__init__()
        self.compactness_weight = compactness_weight
        self.separation_weight = separation_weight
        
    def forward(self, features, labels, centroids=None):
        """
        计算聚类损失
        Args:
            features: 特征向量 (B, D)
            labels: 聚类标签 (B,)
            centroids: 聚类中心 (K, D)
        """
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        if centroids is not None:
            centroids = F.normalize(centroids, dim=1)
        
        # 紧密度损失：同类样本应该聚集在一起
        compactness_loss = 0.0
        if labels is not None and centroids is not None:
            # 计算每个样本到其聚类中心的距离
            sample_centroids = centroids[labels]
            distances = F.mse_loss(features, sample_centroids, reduction='none')
            compactness_loss = distances.mean()
        
        # 分离度损失：不同类样本应该分离
        separation_loss = 0.0
        if labels is not None and centroids is not None:
            # 计算不同聚类中心之间的距离
            num_clusters = centroids.size(0)
            if num_clusters > 1:
                # 计算所有聚类中心对之间的距离
                centroid_distances = torch.cdist(centroids, centroids, p=2)
                # 排除对角线元素
                mask = torch.eye(num_clusters, device=centroids.device).bool()
                centroid_distances = centroid_distances.masked_select(~mask)
                # 分离度损失：希望聚类中心距离越大越好
                separation_loss = -centroid_distances.mean()
        
        # 总损失
        total_loss = (self.compactness_weight * compactness_loss + 
                     self.separation_weight * separation_loss)
        
        return total_loss, compactness_loss, separation_loss

class UMC(nn.Module):
    
    def __init__(self, args):
        super(UMC, self).__init__()

        self.t_dim = 768
        self.a_dim = 768
        self.v_dim = 256
        
        self.args = args
        base_dim = args.base_dim
        
        self.num_heads = args.nheads
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.attn_mask = args.attn_mask
        
        # 功能开关
        # 创新点一：ConFEDE双投影机制 - 默认关闭用于消融实验
        self.enable_video_dual = getattr(args, 'enable_video_dual', False)  # 默认关闭
        self.enable_audio_dual = getattr(args, 'enable_audio_dual', False)  # 默认关闭
        # 替代性消融实验：双投影策略选择
        self.dual_projection_strategy = getattr(args, 'dual_projection_strategy', 'dual_projection')  # 'dual_projection' | 'simple_linear'
        # 创新点二：文本引导注意力和自注意力机制 - 默认关闭用于消融实验
        self.enable_text_guided_attention = getattr(args, 'enable_text_guided_attention', False)  # 默认关闭
        self.enable_self_attention = getattr(args, 'enable_self_attention', False)  # 默认关闭
        self.self_attention_layers = getattr(args, 'self_attention_layers', 2)
        
        # 消融实验开关
        self.enable_gated_fusion = getattr(args, 'enable_gated_fusion', True)
        self.enable_cross_attention = getattr(args, 'enable_cross_attention', True)  # 多层Cross-Attention开关
        self.enable_feature_interaction = getattr(args, 'enable_feature_interaction', True)  # 特征交互层开关
        # 创新点三：聚类优化架构 - 默认关闭用于消融实验
        self.use_clustering_projector = getattr(args, 'use_clustering_projector', False)  # 默认关闭
        self.use_clustering_fusion = getattr(args, 'use_clustering_fusion', False)  # 默认关闭
        self.use_attention_pooling = getattr(args, 'use_attention_pooling', False)  # 默认关闭
        self.use_layer_norm = getattr(args, 'use_layer_norm', False)  # 默认关闭
        
        # 基础编码器
        self.text_embedding = BERTEncoder(args)
        
        # 特征投影层
        self.text_layer = nn.Linear(args.text_feat_dim, base_dim)
        self.video_layer = nn.Linear(args.video_feat_dim, base_dim)
        self.audio_layer = nn.Linear(args.audio_feat_dim, base_dim)
         
        # Transformer编码器
        self.v_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)
        self.a_encoder = self.get_transformer_encoder(base_dim, args.encoder_layers_1)
        
        # 共享嵌入层
        self.shared_embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(base_dim, base_dim),
        )

        # 融合层 - 处理拼接后的3*base_dim维度
        self.fusion_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(3 * base_dim, base_dim),
        )

        # 改进：添加LayerNorm提升训练稳定性
        self.ln_text = nn.LayerNorm(base_dim)
        self.ln_video = nn.LayerNorm(base_dim)
        self.ln_audio = nn.LayerNorm(base_dim)
        self.ln_fusion = nn.LayerNorm(base_dim)
        
        # 替代性消融实验：简单线性映射（当双投影关闭时使用）
        self.use_simple_linear_projection = getattr(args, 'use_simple_linear_projection', False)
        
        # 新增：视频双投影（可选）
        if self.enable_video_dual:
            if self.dual_projection_strategy == 'simple_linear':
                self.video_dual_projector = SimpleLinearProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
            else:
                self.video_dual_projector = VideoDualProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
        elif self.use_simple_linear_projection:
            # 替代方案：使用简单线性映射替代双投影
            self.video_simple_projector = SimpleLinearProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
        
        # 新增：音频双投影（可选）
        if self.enable_audio_dual:
            if self.dual_projection_strategy == 'simple_linear':
                self.audio_dual_projector = SimpleLinearProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
            else:
                self.audio_dual_projector = AudioDualProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
        elif self.use_simple_linear_projection:
            # 替代方案：使用简单线性映射替代双投影
            self.audio_simple_projector = SimpleLinearProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
        
        # 多层cross-attention
        self.cross_attn_video_layers = nn.ModuleList([
            MultiheadAttention(base_dim, self.num_heads, attn_dropout=self.attn_dropout) for _ in range(1)
        ])
        self.cross_attn_audio_layers = nn.ModuleList([
            MultiheadAttention(base_dim, self.num_heads, attn_dropout=self.attn_dropout) for _ in range(1)
        ])
        
        # 新增：以文本为锚点的交叉注意力层
        self.text_guided_video_attn = MultiheadAttention(base_dim, self.num_heads, attn_dropout=self.attn_dropout)
        self.text_guided_audio_attn = MultiheadAttention(base_dim, self.num_heads, attn_dropout=self.attn_dropout)
        
        # 新增：自注意力层
        if self.enable_self_attention:
            self.self_attention_layers = nn.ModuleList([
                MultiheadAttention(base_dim, self.num_heads, attn_dropout=self.attn_dropout) 
                for _ in range(self.self_attention_layers)
            ])
        else:
            self.self_attention_layers = None
        
        # 新增：特征融合后的归一化层
        self.post_attention_norm = nn.LayerNorm(base_dim)
        
        # 新增：对比学习投影头
        self.contrastive_proj = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim)
        )
        
        # 新增：温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # BERT文本层
        self.bert_text_layer = nn.Linear(self.t_dim, base_dim)
        self.text_feat_layer = nn.Linear(self.t_dim, base_dim)
        
        # 新增：门控融合机制
        self.text_weight_gate = nn.Linear(base_dim, 1)
        self.video_weight_gate = nn.Linear(base_dim, 1)
        self.audio_weight_gate = nn.Linear(base_dim, 1)
        
        # 新增：特征交互层
        self.feature_interaction = nn.MultiheadAttention(base_dim, self.num_heads, dropout=self.attn_dropout)
        
        # 新增：注意力池化层
        self.attention_pooling_layer = nn.MultiheadAttention(base_dim, self.num_heads, dropout=self.attn_dropout)

        # 新增：聚类优化开关
        self.enable_clustering_optimization = getattr(args, 'enable_clustering_optimization', True)
        
        # 新增：对比学习损失开关
        self.enable_contrastive_loss = getattr(args, 'enable_contrastive_loss', True)  # 默认开启
            
        # 新增：聚类损失权重
        self.clustering_weight = getattr(args, 'clustering_weight', 1.0)
        self.contrastive_weight = getattr(args, 'contrastive_weight', 0.5)
        
        # 新增：对比学习损失
        if self.enable_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(temperature=getattr(args, 'contrastive_temperature', 0.07))
        else:
            self.contrastive_loss = None
        
        # 新增：聚类投影器
        if self.use_clustering_projector:
            self.clustering_projector = ClusteringProjector(base_dim, base_dim, dropout=args.hidden_dropout_prob)
        
        # 新增：聚类融合器
        if self.use_clustering_fusion:
            self.clustering_fusion = ClusteringFusion(base_dim, args.num_labels, dropout=args.hidden_dropout_prob)
        
        # 新增：聚类损失函数
        self.clustering_loss_fn = ClusteringLoss(
            compactness_weight=getattr(args, 'compactness_weight', 1.0),
            separation_weight=getattr(args, 'separation_weight', 1.0)
        )

        # 确保模型参数为float32
        self.float()

    def get_transformer_encoder(self, base_dim, encoder_layers):
        """获取Transformer编码器"""
        return TransformerEncoder(
            base_dim, 
            encoder_layers, 
            self.num_heads, 
            self.attn_dropout, 
            self.relu_dropout, 
            self.embed_dropout, 
            self.res_dropout, 
            self.attn_mask
        )

    def attention_pooling(self, features, query):
        """注意力池化"""
        # features: (L, B, D), query: (B, D)
        query = query.unsqueeze(0)  # (1, B, D)
        pooled, _ = self.attention_pooling_layer(query, features, features)
        return pooled.squeeze(0)  # (B, D)

    def gated_fusion(self, text_feat, video_feat, audio_feat):
        """门控融合机制"""
        # 计算门控权重
        text_weight = torch.sigmoid(self.text_weight_gate(text_feat))
        video_weight = torch.sigmoid(self.video_weight_gate(video_feat))
        audio_weight = torch.sigmoid(self.audio_weight_gate(audio_feat))
        
        # 归一化权重
        total_weight = text_weight + video_weight + audio_weight + 1e-8
        text_weight = text_weight / total_weight
        video_weight = video_weight / total_weight
        audio_weight = audio_weight / total_weight
        
        # 加权融合
        fused_feat = text_weight * text_feat + video_weight * video_feat + audio_weight * audio_feat
        return fused_feat




    def forward(self, text_feats, video_feats, audio_feats, mode=None, labels=None):
        """前向传播"""
        # 确保所有输入数据类型为float32
        if video_feats.dtype != torch.float32:
            video_feats = video_feats.float()
        if audio_feats.dtype != torch.float32:
            audio_feats = audio_feats.float()
        
        # 确保模型在正确的设备上
        device = next(self.parameters()).device
        video_feats = video_feats.to(device)
        audio_feats = audio_feats.to(device)
        
        # 1. 文本、视频、音频特征提取 + LayerNorm
        text_bert = self.text_embedding(text_feats)  # (B, L, t_dim)
        text_feat = self.text_feat_layer(text_bert)  # (B, L, base_dim)
        text_feat = self.ln_text(text_feat)
        video_seq = self.ln_video(self.video_layer(video_feats))    # (B, L, base_dim)
        audio_seq = self.ln_audio(self.audio_layer(audio_feats))    # (B, L, base_dim)
        
        # 2. 创新点一：视频投影处理 - 消融实验控制
        if self.enable_video_dual:
            # 使用双投影机制（ConFEDE）
            video_seq = self.video_dual_projector(video_seq)
        elif self.use_simple_linear_projection:
            # 替代方案：使用简单线性映射
            video_seq = self.video_simple_projector(video_seq)
        # else: 跳过投影，直接使用原始视频特征
        
        # 3. 创新点一：音频投影处理 - 消融实验控制  
        if self.enable_audio_dual:
            # 使用双投影机制（ConFEDE）
            audio_seq = self.audio_dual_projector(audio_seq)
        elif self.use_simple_linear_projection:
            # 替代方案：使用简单线性映射
            audio_seq = self.audio_simple_projector(audio_seq)
        # else: 跳过投影，直接使用原始音频特征
        
        # 4. 多层cross-attention - 消融实验控制
        text_feat_t = text_feat.permute(1, 0, 2)     # (L, B, base_dim)
        video_seq_t = video_seq.permute(1, 0, 2)     # (L, B, base_dim)
        audio_seq_t = audio_seq.permute(1, 0, 2)     # (L, B, base_dim)
        
        x_video = text_feat_t
        x_audio = text_feat_t
        
        if self.enable_cross_attention:
            for layer in self.cross_attn_video_layers:
                x_video, _ = layer(x_video, video_seq_t, video_seq_t)
            for layer in self.cross_attn_audio_layers:
                x_audio, _ = layer(x_audio, audio_seq_t, audio_seq_t)
        else:
            # 跳过Cross-Attention，直接使用原始特征
            x_video = video_seq_t
            x_audio = audio_seq_t
        
        # 5. 创新点二：以文本为锚点的交叉注意力 - 消融实验控制
        if self.enable_text_guided_attention:
            text_guided_video, _ = self.text_guided_video_attn(
                text_feat_t, x_video, x_video
            )
            text_guided_audio, _ = self.text_guided_audio_attn(
                text_feat_t, x_audio, x_audio
            )
        else:
            # 跳过文本引导注意力，直接使用交叉注意力结果
            text_guided_video = x_video
            text_guided_audio = x_audio
        
        # 6. 创新点二：自注意力层 - 消融实验控制
        combined_features = torch.cat([text_feat_t, text_guided_video, text_guided_audio], dim=0)
        
        attended_features = combined_features
        if self.self_attention_layers:
            for self_attn_layer in self.self_attention_layers:
                attended_features, _ = self_attn_layer(
                    attended_features, attended_features, attended_features
                )
                attended_features = attended_features + combined_features
        # else: 跳过自注意力，直接使用拼接后的特征
        
        # 分离三个模态的特征
        seq_len = text_feat_t.shape[0]
        enhanced_text_feat = attended_features[:seq_len]
        enhanced_video_feat = attended_features[seq_len:2*seq_len]
        enhanced_audio_feat = attended_features[2*seq_len:]
        
        # 归一化
        enhanced_text_feat = self.post_attention_norm(enhanced_text_feat)
        enhanced_video_feat = self.post_attention_norm(enhanced_video_feat)
        enhanced_audio_feat = self.post_attention_norm(enhanced_audio_feat)
        
        # 7. 文本BERT的CLS特征
        text_bert_cls = text_bert[:, 0]  # (B, t_dim)
        text_bert_proj = self.bert_text_layer(text_bert_cls)  # (B, base_dim)
        
        # 创新点三：使用注意力池化 - 消融实验控制
        if self.use_attention_pooling:
            text_video_enh_pooled = self.attention_pooling(enhanced_video_feat, text_bert_proj)
            text_audio_enh_pooled = self.attention_pooling(enhanced_audio_feat, text_bert_proj)
        else:
            # 简单平均池化（跳过注意力池化）
            text_video_enh_pooled = enhanced_video_feat.mean(dim=0)  # (B, base_dim)
            text_audio_enh_pooled = enhanced_audio_feat.mean(dim=0)  # (B, base_dim)
        
        # 特征交互 - 消融实验控制
        if self.enable_feature_interaction:
            interaction_input = torch.stack([text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled], dim=1)
            interaction_input = interaction_input.transpose(0, 1)
            
            interacted_features, _ = self.feature_interaction(interaction_input, interaction_input, interaction_input)
            interacted_features = interacted_features.transpose(0, 1)
            
            # 分离交互后的特征
            text_bert_proj = interacted_features[:, 0]
            text_video_enh_pooled = interacted_features[:, 1]
            text_audio_enh_pooled = interacted_features[:, 2]
        # else: 跳过特征交互，直接使用池化后的特征
        
        # 使用门控融合
        if self.enable_gated_fusion:
            enhanced_text = self.gated_fusion(text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled)
        else:
            # 简单拼接融合
            enhanced_text = torch.cat([text_bert_proj, text_video_enh_pooled, text_audio_enh_pooled], dim=-1)
            # 需要将拼接后的特征投影回base_dim维度
            enhanced_text = self.fusion_layer(enhanced_text)
        
        if self.use_layer_norm:
            enhanced_text = self.ln_fusion(enhanced_text)
        
        # 计算最终特征
        features = enhanced_text
        
        # 创新点三：聚类优化路径 - 消融实验控制
        clustering_features = features
        clustering_loss = None
        cluster_weights = None
        
        if self.enable_clustering_optimization:
            # 创新点三：使用聚类投影器
            if self.use_clustering_projector:
                clustering_features, center_features = self.clustering_projector(features)
            # else: 跳过聚类投影器，直接使用原始特征
            
            # 创新点三：使用聚类融合器
            if self.use_clustering_fusion:
                clustering_features, cluster_weights = self.clustering_fusion(clustering_features)
            # else: 跳过聚类融合器
            
            # 创新点三：计算聚类损失
            if labels is not None:
                clustering_loss, compactness_loss, separation_loss = self.clustering_loss_fn(
                    clustering_features, labels
                )
        # else: 跳过整个聚类优化路径
        
        # 对比学习特征和损失 - 消融实验控制
        contrastive_loss = None
        if self.enable_contrastive_loss:
            contrastive_features = self.contrastive_proj(features)
            # 计算对比学习损失
            if labels is not None:
                contrastive_loss = self.contrastive_loss(contrastive_features, labels)
        # else: 跳过对比学习损失
        
        # 返回结果
        if mode == 'features':
            return features
        elif mode == 'contrastive':
            return features, contrastive_loss
        elif mode == 'train-mm':
            mlp_output = self.shared_embedding_layer(features)
            if contrastive_loss is not None and clustering_loss is not None:
                return features, mlp_output, contrastive_loss, clustering_loss
            elif contrastive_loss is not None:
                return features, mlp_output, contrastive_loss
            elif clustering_loss is not None:
                return features, mlp_output, clustering_loss
            else:
                return features, mlp_output
        elif mode == 'pretrain-mm':
            mlp_output = self.shared_embedding_layer(features)
            if contrastive_loss is not None and clustering_loss is not None:
                return mlp_output, contrastive_loss, clustering_loss
            elif contrastive_loss is not None:
                return mlp_output, contrastive_loss
            elif clustering_loss is not None:
                return mlp_output, clustering_loss
            else:
                return mlp_output
        else:
            return features

def umc(args):
    return UMC(args)
