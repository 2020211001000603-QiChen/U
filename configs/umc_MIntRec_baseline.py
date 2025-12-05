
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
                'delta': [0.05],
                'thres': [0.1],
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
                'enable_video_dual': True,
                # 新增：音频双投影开关（只对音频模态应用ConFEDE机制）
                'enable_audio_dual': True,
                # 新增：注意力机制增强开关
                'enable_text_guided_attention': True,  # 以文本为锚点的交叉注意力
                'enable_self_attention': True,        # 自注意力层
                'self_attention_layers': 2,           # 自注意力层数
                # 对比学习参数：提升聚类效果
                'contrastive_weight': 0.1,          # 对比学习损失权重
                'contrastive_temperature': 0.07,    # 对比学习温度参数
                'enable_contrastive': True,         # 是否启用对比学习
                
                # 聚类损失参数：专门提升聚类效果
                'clustering_weight': 0.2,           # 聚类损失权重
                'compactness_weight': 1.0,          # 紧密度损失权重
                'separation_weight': 1.0,           # 分离度损失权重
                'enable_clustering_loss': True,     # 是否启用聚类损失

                # 在UMC-main/configs/umc_MIntRec.py文件中，修改第76-81行：

# 聚类损失参数：专门提升聚类效果
                'enable_clustering_optimization': True,  # 启用聚类优化路径
                'clustering_weight': 1.0,               # 聚类损失权重
                'contrastive_weight': 0.5,              # 对比学习权重
                'compactness_weight': 1.0,              # 紧密度损失权重
                'separation_weight': 1.0,               # 分离度损失权重
                'enable_clustering_loss': True, 
                # 渐进式学习优化参数
                'max_threshold': 0.5,              # 最大阈值
                'min_threshold': 0.05,             # 最小阈值
                'performance_window': 3,            # 性能分析窗口大小
                'patience': 5,                     # 早停耐心值
                'enable_early_stop': True,         # 启用早停
                'enable_adaptive_threshold': True, # 启用自适应阈值
            }
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters
    