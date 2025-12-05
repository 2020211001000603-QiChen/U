
class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        
        # 处理消融实验配置
        ablation_config = self._get_ablation_config(args)
        
        if args.multimodal_method == 'umc':
            hyper_parameters = {
                'pretrained_bert_model': 'uncased_L-12_H-768_A-12',
                'pretrain_batch_size': 128,
                'train_batch_size': 128,
                'eval_batch_size': 128,
                'test_batch_size': 128,
                'num_pretrain_epochs': 100,
                'num_train_epochs': 100,
                'pretrain': [True],
                'aligned_method': 'ctc',
                'need_aligned': False,
                'freeze_pretrain_bert_parameters': [True],
                'freeze_train_bert_parameters': [True],
                'pretrain_temperature': [0.2],
                'train_temperature_sup': [1.4],
                'train_temperature_unsup': [1],
                'activation': 'tanh',
                'lr_pre': 2e-5,
                'lr': [3e-4],
                'delta': [0.02],     # 降低增长步长
                'thres': [0.05],     # 降低初始阈值
                'topk': [5],
                'weight_decay': 0.01,
                'feat_dim': 768,
                'hidden_size': 768,
                'grad_clip': -1.0,
                'warmup_proportion': 0.1,
                'hidden_dropout_prob': 0.1,
                'weight': 1.0,
                'loss_mode': 'rdrop',
                'base_dim': 256,
                'nheads': 8,
                'attn_dropout': 0.1,
                'relu_dropout': 0.1,
                'embed_dropout': 0.1,
                'res_dropout': 0.0,
                'attn_mask': True,
                'encoder_layers_1': 1,
                'fusion_act': 'tanh',
                
                # 新增：视频双投影开关（只对视频模态应用ConFEDE机制）
                'enable_video_dual': ablation_config.get('enable_video_dual', True),
                'enable_audio_dual': ablation_config.get('enable_audio_dual', True),
                # 替代性消融实验：双投影策略选择
                'dual_projection_strategy': ablation_config.get('dual_projection_strategy', 'dual_projection'),
                # 新增：注意力机制增强开关
                'enable_text_guided_attention': ablation_config.get('enable_text_guided_attention', True),
                'enable_self_attention': ablation_config.get('enable_self_attention', True),
                'self_attention_layers': ablation_config.get('self_attention_layers', 2),
                
                # 聚类损失参数：专门提升聚类效果
                'enable_clustering_optimization': ablation_config.get('enable_clustering_optimization', True),  # 启用聚类优化路径
                'enable_clustering_loss': ablation_config.get('enable_clustering_loss', True), 
                
                # 渐进式学习优化参数
                'max_threshold': 0.5,              # 最大阈值
                'min_threshold': 0.05,             # 最小阈值
                'performance_window': 3,            # 性能分析窗口大小
                'patience': 5,                     # 早停耐心值
                'enable_early_stop': ablation_config.get('enable_early_stop', True),         # 启用早停
                'enable_adaptive_threshold': ablation_config.get('enable_adaptive_threshold', True), # 启用自适应阈值

                # 消融实验配置
                'ablation_config': ablation_config,

                # 损失权重配置
                'clustering_weight': ablation_config.get('clustering_weight', 1.0),
                'contrastive_weight': ablation_config.get('contrastive_weight', 0.5),
                'compactness_weight': ablation_config.get('compactness_weight', 0.3),
                'separation_weight': ablation_config.get('separation_weight', 0.2),

                # 固定训练轮数（用于消融实验）
                'fixed_train_epochs': ablation_config.get('fixed_train_epochs', 25),  # 减少训练轮数
            }
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters
    
    def _get_ablation_config(self, args):
        """获取消融实验配置"""
        # 默认配置
        default_config = {
            # 创新点一：双投影机制 (ConFEDE)
            'enable_dual_projection': True,
            'enable_video_dual': True,
            'enable_audio_dual': True,
            
            # 创新点二：文本引导注意力和自注意力机制
            'enable_text_guided_attention': True,
            'enable_self_attention': True,
            'self_attention_layers': 2,
            
            # 创新点三：聚类优化架构（包含聚类损失）
            'enable_clustering_loss': True,
            'enable_contrastive_loss': True,
            'enable_compactness_loss': True,
            'enable_separation_loss': True,
            
            # 渐进式学习
            'enable_progressive_learning': True,
            'enable_adaptive_threshold': True,
            'enable_performance_monitoring': True,
            'enable_early_stop': True,
            
            # 注意力机制（已移到创新点二）
            
            # 聚类优化
            'enable_clustering_optimization': True,
            
            # 损失权重
            'clustering_weight': 1.0,
            'contrastive_weight': 0.5,
            'compactness_weight': 0.3,
            'separation_weight': 0.2,
            
            # 训练参数
            'fixed_train_epochs': 25,
        }
        
        # 根据消融实验名称调整配置
        if hasattr(args, 'ablation_experiment') and args.ablation_experiment:
            experiment_name = args.ablation_experiment.lower()
            
            if experiment_name == 'full_confede':
                # full_confede: 启用所有ConFEDE相关功能
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                })
            elif experiment_name == 'no_dual_projection':
                # 禁用双投影机制 - 测试第一个创新点的贡献
                # 关闭创新点一，保留创新点二和创新点三
                # 同时关闭基础组件（Cross-Attention、特征交互、门控融合、对比学习损失）
                config = default_config.copy()
                config.update({
                    # 创新点一：关闭双投影机制
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    
                    # 创新点二：明确启用文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    
                    # 创新点三：明确启用渐进式学习和聚类优化
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                    'enable_clustering_optimization': True,
                    'enable_clustering_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    
                    # 基础组件：关闭（更彻底的消融）
                    'enable_cross_attention': False,  # 关闭多层Cross-Attention
                    'enable_feature_interaction': False,  # 关闭特征交互层
                    'enable_gated_fusion': False,  # 关闭门控融合
                    'enable_contrastive_loss': False,  # 关闭对比学习损失
                    
                    # 其他可选组件：明确关闭（确保不影响实验）
                    'use_clustering_projector': False,  # 关闭聚类投影器
                    'use_clustering_fusion': False,  # 关闭聚类融合器
                    'use_attention_pooling': False,  # 关闭注意力池化（使用简单平均池化）
                    'use_layer_norm': False,  # 关闭融合层归一化
                })
            elif experiment_name == 'only_innovation1':
                # 仅保留创新点一（双投影机制）- 测试第一个创新点的独立贡献
                # 保留创新点一，关闭创新点二和创新点三
                config = default_config.copy()
                config.update({
                    # 创新点一：保留双投影机制
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    
                    # 创新点二：关闭文本引导注意力和自注意力机制
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    
                    # 创新点三：关闭渐进式学习和聚类优化
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
            elif experiment_name == 'only_innovation2':
                # 仅保留创新点二（文本引导注意力）- 测试第二个创新点的独立贡献
                # 保留创新点二，关闭创新点一和创新点三
                config = default_config.copy()
                config.update({
                    # 创新点一：关闭双投影机制
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    
                    # 创新点二：保留文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    
                    # 创新点三：关闭渐进式学习和聚类优化
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
            elif experiment_name == 'only_innovation3':
                # 仅保留创新点三（渐进式学习策略）- 测试第三个创新点的独立贡献
                # 保留创新点三，关闭创新点一和创新点二
                config = default_config.copy()
                config.update({
                    # 创新点一：关闭双投影机制
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    
                    # 创新点二：关闭文本引导注意力和自注意力机制
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    
                    # 创新点三：保留渐进式学习和聚类优化
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
            elif experiment_name == 'no_dual_projection_strict_ablation':
                # 严格消融实验：关闭创新点一，开启创新点二和三，关闭基础组件
                # 关闭创新点一，保留创新点二和创新点三
                # 同时关闭多层Cross-Attention、特征交互层、门控融合
                config = default_config.copy()
                config.update({
                    # 创新点一：关闭双投影机制
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    
                    # 创新点二：明确启用文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    
                    # 创新点三：明确启用渐进式学习和聚类优化
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                    'enable_clustering_optimization': True,
                    'enable_clustering_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,
                    
                    # 关闭基础组件（更彻底的消融）
                    'enable_cross_attention': False,  # 关闭多层Cross-Attention
                    'enable_feature_interaction': False,  # 关闭特征交互层
                    'enable_gated_fusion': False,  # 关闭门控融合
                    'enable_contrastive_loss': False,  # 关闭对比学习损失（重要！）
                })
            elif experiment_name == 'no_text_guided_attention':
                # 禁用文本引导注意力机制 - 测试第二个创新点的贡献
                # 关闭创新点二，保留创新点一和创新点三
                config = default_config.copy()
                config.update({
                    # 创新点一：保留双投影机制
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    
                    # 创新点二：关闭文本引导注意力和自注意力机制
                    'enable_text_guided_attention': False,
                    'enable_self_attention': False,
                    
                    # 创新点三：明确启用渐进式学习和聚类优化
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
            elif experiment_name == 'no_clustering_loss':
                # 禁用聚类损失
                config = default_config.copy()
                config.update({
                    'enable_clustering_loss': False,
                    'enable_contrastive_loss': False,
                    'enable_compactness_loss': False,
                    'enable_separation_loss': False,
                })
            elif experiment_name == 'no_progressive_learning':
                # 禁用渐进式学习策略 - 测试第三个创新点的贡献
                # 关闭创新点三，保留创新点一和创新点二
                config = default_config.copy()
                config.update({
                    # 创新点一：保留双投影机制
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    
                    # 创新点二：保留文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    
                    # 创新点三：关闭渐进式学习和聚类优化
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
            elif experiment_name == 'dual_projection_full':
                # 明确开启双投影机制，作为与替代方案对比的标准配置
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': True,
                    'enable_video_dual': True,
                    'enable_audio_dual': True,
                    'dual_projection_strategy': 'dual_projection',
                    'use_simple_linear_projection': False,
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
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
            elif experiment_name == 'dual_projection_simple_linear':
                # 替代性消融实验：用简单线性映射替代双投影机制
                # 关闭双投影，使用简单线性映射作为替代方案
                config = default_config.copy()
                config.update({
                    # 创新点一：关闭双投影，使用简单线性映射替代
                    'enable_dual_projection': False,
                    'enable_video_dual': False,  # 关闭双投影
                    'enable_audio_dual': False,  # 关闭双投影
                    'use_simple_linear_projection': True,  # 使用简单线性映射替代
                    'dual_projection_strategy': 'simple_linear',
                    
                    # 创新点二：保留文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,
                    
                    # 创新点三：保留渐进式学习和聚类优化
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
            elif experiment_name == 'dual_projection_simple_linear_dim128':
                # 子空间维度敏感性实验：简单线性映射 + base_dim = 128
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'use_simple_linear_projection': True,
                    'dual_projection_strategy': 'simple_linear',

                    # 创新点二：文本引导注意力和自注意力机制
                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,

                    # 创新点三：渐进式学习和聚类优化
                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                    'enable_clustering_optimization': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,

                    # 关键超参数：子空间维度
                    'base_dim': 128,
                })
            elif experiment_name == 'dual_projection_simple_linear_dim256':
                # 子空间维度敏感性实验：简单线性映射 + base_dim = 256（默认对照）
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'use_simple_linear_projection': True,
                    'dual_projection_strategy': 'simple_linear',

                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,

                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                    'enable_clustering_optimization': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,

                    'base_dim': 256,
                })
            elif experiment_name == 'dual_projection_simple_linear_dim384':
                # 子空间维度敏感性实验：简单线性映射 + base_dim = 384
                config = default_config.copy()
                config.update({
                    'enable_dual_projection': False,
                    'enable_video_dual': False,
                    'enable_audio_dual': False,
                    'use_simple_linear_projection': True,
                    'dual_projection_strategy': 'simple_linear',

                    'enable_text_guided_attention': True,
                    'enable_self_attention': True,
                    'self_attention_layers': 2,

                    'enable_progressive_learning': True,
                    'enable_adaptive_threshold': True,
                    'enable_performance_monitoring': True,
                    'enable_early_stop': True,
                    'enable_clustering_optimization': True,
                    'enable_clustering_loss': True,
                    'enable_contrastive_loss': True,
                    'enable_compactness_loss': True,
                    'enable_separation_loss': True,

                    'base_dim': 384,
                })
            else:
                # 未知的消融实验，使用默认配置
                config = default_config.copy()
        else:
            # 没有指定消融实验，使用默认配置
            config = default_config.copy()
            
        return config
    