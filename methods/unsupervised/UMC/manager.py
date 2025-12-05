import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time 
import copy

from sklearn.cluster import KMeans
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, set_torch_seed
from transformers import BertTokenizer

from backbones.base import freeze_bert_parameters
from sklearn.neighbors import NearestNeighbors, KDTree
from utils.neighbor_dataset import NeighborsDataset
from torch.utils.data import DataLoader

from utils.metrics import clustering_score
from .pretrain import PretrainUMCManager

from data.utils import get_dataloader
from .utils import *

# 渐进式学习优化类
class AdaptiveProgressiveLearning:
    def __init__(self, 
                 initial_threshold=0.1, 
                 max_threshold=0.5, 
                 min_threshold=0.05,
                 performance_window=3,
                 patience=5):
        self.initial_threshold = initial_threshold
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.performance_window = performance_window
        self.patience = patience
        
        # 性能历史记录
        self.performance_history = []
        self.threshold_history = []
        self.loss_history = []
        self.epoch_history = []
        
        # 统计信息
        self.best_performance = 0.0
        self.best_epoch = 0
        self.total_improvements = 0
        self.total_degradations = 0
        
    def compute_threshold(self, epoch, total_epochs, current_performance, current_loss):
        """计算优化的阈值"""
        # 记录历史
        self._record_history(epoch, current_performance, current_loss)
        
        # 更新最佳性能
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_epoch = epoch
            self.total_improvements += 1
        elif current_performance < self.best_performance * 0.95:
            self.total_degradations += 1
        
        # 1. 计算基础阈值（S型曲线增长）
        base_threshold = self._compute_base_threshold(epoch, total_epochs)
        
        # 2. 性能自适应调整
        performance_adjustment = self._compute_performance_adjustment()
        
        # 3. 损失自适应调整
        loss_adjustment = self._compute_loss_adjustment()
        
        # 4. 稳定性调整
        stability_adjustment = self._compute_stability_adjustment()
        
        # 5. 综合计算
        adaptive_threshold = (base_threshold + 
                            performance_adjustment + 
                            loss_adjustment + 
                            stability_adjustment)
        
        # 6. 边界约束
        adaptive_threshold = np.clip(adaptive_threshold, self.min_threshold, self.max_threshold)
        self.threshold_history.append(adaptive_threshold)
        
        return adaptive_threshold
    
    def _compute_base_threshold(self, epoch, total_epochs):
        """计算基础阈值（S型曲线增长）"""
        progress = epoch / total_epochs
        threshold_range = self.max_threshold - self.initial_threshold
        
        if progress < 0.2:
            # 早期缓慢增长阶段
            phase_progress = progress / 0.2
            base_threshold = self.initial_threshold + threshold_range * 0.2 * (phase_progress ** 2)
        elif progress < 0.6:
            # 中期快速增长阶段
            phase_progress = (progress - 0.2) / 0.4
            base_threshold = self.initial_threshold + threshold_range * (0.2 + 0.5 * phase_progress)
        elif progress < 0.9:
            # 后期稳定增长阶段
            phase_progress = (progress - 0.6) / 0.3
            base_threshold = self.initial_threshold + threshold_range * (0.7 + 0.25 * np.log(1 + 3 * phase_progress) / np.log(4))
        else:
            # 最终阶段
            phase_progress = (progress - 0.9) / 0.1
            base_threshold = self.initial_threshold + threshold_range * (0.95 + 0.05 * phase_progress)
        
        return base_threshold
    
    def _compute_performance_adjustment(self):
        """基于性能趋势计算阈值调整"""
        if len(self.performance_history) < self.performance_window:
            return 0.0
        
        recent_performance = self.performance_history[-self.performance_window:]
        performance_trend = self._analyze_performance_trend(recent_performance)
        
        adjustment = 0.0
        if performance_trend == 'improving':
            improvement_rate = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
            if improvement_rate > 0.02:
                adjustment = 0.03
            elif improvement_rate > 0.01:
                adjustment = 0.02
            else:
                adjustment = 0.01
        elif performance_trend == 'declining':
            decline_rate = abs(recent_performance[-1] - recent_performance[0]) / len(recent_performance)
            if decline_rate > 0.02:
                adjustment = -0.02
            elif decline_rate > 0.01:
                adjustment = -0.01
            else:
                adjustment = -0.005
        
        return adjustment
    
    def _compute_loss_adjustment(self):
        """基于损失变化计算阈值调整"""
        if len(self.loss_history) < 2:
            return 0.0
        
        recent_losses = self.loss_history[-3:]
        if len(recent_losses) < 2:
            return 0.0
        
        loss_change_rate = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
        
        adjustment = 0.0
        if loss_change_rate < -0.05:
            adjustment = 0.015
        elif loss_change_rate < -0.01:
            adjustment = 0.01
        elif loss_change_rate > 0.05:
            adjustment = -0.02
        elif loss_change_rate > 0.01:
            adjustment = -0.01
        
        return adjustment
    
    def _compute_stability_adjustment(self):
        """基于训练稳定性计算阈值调整"""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent_performance = self.performance_history[-5:]
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        
        if performance_mean > 0:
            cv = performance_std / performance_mean
            if cv > 0.1:
                return -0.01
            elif cv < 0.02:
                return 0.005
        
        return 0.0
    
    def should_early_stop(self, min_epochs=10):
        """判断是否应该早停"""
        if len(self.performance_history) < min_epochs:
            return False
        
        # 性能早停
        if len(self.performance_history) >= self.patience:
            recent_performance = self.performance_history[-self.patience:]
            if all(perf <= self.best_performance for perf in recent_performance):
                return True
        
        # 损失早停
        if len(self.loss_history) >= self.patience:
            recent_losses = self.loss_history[-self.patience:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                return True
        
        # 性能退化早停
        if self.total_degradations > 3 and len(self.performance_history) > 10:
            return True
        
        return False
    
    def _record_history(self, epoch, performance, loss):
        """记录历史信息"""
        self.performance_history.append(performance)
        self.loss_history.append(loss)
        self.epoch_history.append(epoch)
    
    def _analyze_performance_trend(self, performance_list):
        """分析性能趋势"""
        if len(performance_list) < 2:
            return 'stable'
        
        increasing_count = sum(1 for i in range(1, len(performance_list)) 
                              if performance_list[i] > performance_list[i-1])
        total_changes = len(performance_list) - 1
        
        if total_changes == 0:
            return 'stable'
        
        increasing_ratio = increasing_count / total_changes
        
        if increasing_ratio > 0.7:
            return 'improving'
        elif increasing_ratio < 0.3:
            return 'declining'
        else:
            return 'oscillating'
    
    def get_training_statistics(self):
        """获取训练统计信息"""
        return {
            'best_performance': self.best_performance,
            'best_epoch': self.best_epoch,
            'total_improvements': self.total_improvements,
            'total_degradations': self.total_degradations,
            'performance_trend': self._analyze_performance_trend(self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history),
            'stability_score': self._compute_stability_score()
        }
    
    def _compute_stability_score(self):
        """计算稳定性分数"""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent_performance = self.performance_history[-5:]
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        
        if performance_mean > 0:
            cv = performance_std / performance_mean
            stability_score = max(0, 1 - cv * 10)
        else:
            stability_score = 0.0
        
        return stability_score

class UMCManager:
    
    def __init__(self, args, data, model):

        pretrain_manager = PretrainUMCManager(args, data, model)

        set_torch_seed(args.seed)

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model

        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.test_dataloader = mm_dataloader['train'], mm_dataloader['test']
        self.train_outputs = data.train_outputs
        
        self.criterion = loss_map['CrossEntropyLoss']
        self.contrast_criterion = loss_map['SupConLoss']
        self.mse_criterion = loss_map['MSELoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)
        self.centroids = None

        # 创新点三：自适应渐进式学习策略
        # 配置文件中 thres / delta 可能是单元素列表，这里统一转为标量
        thres_value = args.thres
        if isinstance(thres_value, (list, tuple)):
            thres_value = thres_value[0]
        delta_value = getattr(args, 'delta', 0.02)
        if isinstance(delta_value, (list, tuple)):
            delta_value = delta_value[0]

        if getattr(args, 'enable_progressive_learning', True):
            self.progressive_learner = AdaptiveProgressiveLearning(
                initial_threshold=thres_value,
                max_threshold=0.5,
                min_threshold=0.05,
                performance_window=3,
                patience=5
            )
        else:
            self.progressive_learner = None

        if args.pretrain:
            self.pretrained_model = pretrain_manager.model

            self.num_labels = args.num_labels
            self.load_pretrained_model(self.pretrained_model)
            
        else:
            self.num_labels = args.num_labels
            # 如果pretrain=False，检查是否存在预训练模型
            pretrain_model_path = os.path.join(args.model_output_path, 'pretrain', 'pytorch_model.bin')
            if os.path.exists(pretrain_model_path):
                self.logger.info('Loading pretrained model from %s', pretrain_model_path)
                self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.model_output_path, 'pretrain'), self.device)
            else:
                self.logger.info('Pretrain is False and no pretrained model found, using random initialization')
                self.pretrained_model = pretrain_manager.model  # 使用随机初始化的模型   
            
        if args.train:
            # 使用标量形式的 thres / delta 计算训练轮数
            args.num_train_epochs = (1 - thres_value) / delta_value
            self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr)

            if args.freeze_train_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                self.model = freeze_bert_parameters(self.model, args.multimodal_method)
            
            # 只有在pretrain=True或存在预训练模型时才加载
            if args.pretrain or (hasattr(self, 'pretrained_model') and self.pretrained_model is not None):
                self.load_pretrained_model(self.pretrained_model)
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)   

    def clustering(self, args, init = 'k-means++', threshold = 0.25):
        
        outputs = self._get_outputs(args, mode = 'train', return_feats = True)
        feats = outputs['feats']
        y_true = outputs['y_true']
        
        if init == 'k-means++':
            
            self.logger.info('Initializing centroids with K-means++...')
            start = time.time()
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = 'k-means++').fit(feats) 
            
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            end = time.time()
            self.logger.info('K-means++ used %s s', round(end - start, 2))   
            
        elif init == 'centers':
            
            start = time.time()
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = self.centroids).fit(feats)
            km_centroids, assign_labels = km.cluster_centers_, km.labels_ 
            end = time.time()
            self.logger.info('K-means used %s s', round(end - start, 2))

        self.centroids = km_centroids

        select_ids = []
        
        # 针对类别不平衡：确保每个类别至少选择最小样本数
        # 对于IEMOCAP-DA等极端不平衡数据集，这很重要
        min_samples_per_class = max(5, int(len(feats) / self.num_labels * 0.1))  # 至少5个或总数的10%
        self.logger.info(f"Minimum samples per class: {min_samples_per_class}")

        for cluster_id in range(self.num_labels):
            cluster_samples = feats[assign_labels == cluster_id]
            pos = list(np.where(assign_labels == cluster_id)[0])
            
            # 改进：确保每个类别至少选择最小样本数（针对类别不平衡）
            # 对于极少数类（少于5个样本），选择所有样本
            if len(cluster_samples) <= 5:
                cutoff = len(cluster_samples)
                self.logger.info(f"Cluster {cluster_id}: Very small cluster ({len(cluster_samples)} samples), selecting all")
            else:
                # 确保至少选择最小样本数
                cutoff = max(
                    int(len(cluster_samples) * threshold), 
                    min(min_samples_per_class, len(cluster_samples))
                )
            
            k_candidate_proportions = np.arange(0.1, 0.32, 0.02).tolist()

            if cutoff == 1:
                select_ids.extend(pos)

            else:
                
                best_sorted_indices = None
                best_eval_score = 0
                best_k_cand = None

                for k_cand in k_candidate_proportions:

                    k = max(int(len(cluster_samples) * k_cand), 1)
                    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(cluster_samples)
                    distances, indices = nbrs.kneighbors(cluster_samples)
                    
                    reachable_distances = np.mean(distances[:, 1:], axis=1) 
                    density = 1 / reachable_distances  
                    sorted_indices = np.argsort(density)

                    tmp_select_indices = sorted_indices[-cutoff:]
                    tmp_select_pos = [pos[i] for i in tmp_select_indices] 
                    
                    tmp_feats = feats[tmp_select_pos]
                    tmp_assign_labels = assign_labels[tmp_select_pos]

                    tree = KDTree(tmp_feats)
                    _, ind = tree.query(tmp_feats, k=2)  
                    nearest_neighbor_distances = np.array([tmp_feats[i] - tmp_feats[ind[i, 1]] for i in range(len(tmp_feats))])
                    tmp_eval_score = np.mean(np.linalg.norm(nearest_neighbor_distances, axis=1))

                    if tmp_eval_score > best_eval_score:
               
                        best_eval_score = tmp_eval_score
                        best_k_cand = k_cand
                        best_sorted_indices = sorted_indices

                select_indices = best_sorted_indices[-cutoff:]
                select_pos = [pos[i] for i in select_indices] 
                select_ids.extend(select_pos)
        

        return np.array(assign_labels), select_ids, feats
                  
    def _train(self, args): 
        
        self.model.to(self.device)
        
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):

            # 使用自适应阈值计算
            if self.progressive_learner is not None:
                # 计算当前性能（简化版本，实际应该使用验证集）
                current_performance = self._evaluate_current_performance(args)
                current_loss = self._get_current_loss(args)
                
                threshold = self.progressive_learner.compute_threshold(
                    epoch, args.num_train_epochs, current_performance, current_loss
                )
                
                # 检查早停
                if self.progressive_learner.should_early_stop():
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # 记录训练统计
                if epoch % 5 == 0:
                    stats = self.progressive_learner.get_training_statistics()
                    self.logger.info(f"Training Statistics: {stats}")
            else:
                # 不使用渐进式学习时，使用固定阈值（同样使用标量 thres_value / delta_value）
                base_threshold = thres_value + delta_value * epoch
                threshold = min(base_threshold, 0.3)  # 限制最大阈值为0.3
                self.logger.info(f"Using fixed threshold: {threshold}")
                
                # 添加简单的早停机制
                if epoch > 10:  # 至少训练10个epoch
                    if hasattr(self, 'recent_losses') and len(self.recent_losses) >= 5:
                        recent_avg = np.mean(self.recent_losses[-5:])
                        older_avg = np.mean(self.recent_losses[-10:-5]) if len(self.recent_losses) >= 10 else recent_avg
                        
                        # 添加调试信息
                        self.logger.info(f"Recent loss avg: {recent_avg:.4f}, Older loss avg: {older_avg:.4f}")
                        self.logger.info(f"Loss increase ratio: {recent_avg/older_avg:.4f}")
                        
                        # 更宽松的早停条件：损失上升超过20%且连续5轮
                        if recent_avg > older_avg * 1.20:  # 从1.15改为1.20
                            self.logger.info(f"Loss significantly increasing, early stopping at epoch {epoch}")
                            break

            init_mechanism = 'k-means++' if epoch == 0 else 'centers'
            pseudo_labels, select_ids, feats = self.clustering(args, init = init_mechanism, threshold = threshold)

            if epoch > 0: 
                self.logger.info("***** Epoch: %s *****", str(epoch))
                self.logger.info('Supervised Training Loss: %f', np.round(tr_sup_loss, 5))
                if len(non_select_ids) != 0:
                    self.logger.info('Unsupervised Training Loss: %f', np.round(tr_unsup_loss, 5))

            self.train_outputs['select_ids'] = select_ids
            
            _, pseudo_sup_train_dataloader = get_pseudo_dataloader(args=args, \
                train_outputs=self.train_outputs, mode='pretrain')
            
            
            self.model.train()
            tr_sup_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            
            for batch_sup in tqdm(pseudo_sup_train_dataloader, desc = 'Iteration'):

                text_feats = batch_sup['text_feats'].to(self.device)
                video_feats = batch_sup['video_feats'].to(self.device)
                audio_feats = batch_sup['audio_feats'].to(self.device)

                label_ids = batch_sup['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    outputs_a = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats, mode='train-mm', labels=label_ids)
                    outputs_b = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device), mode='train-mm', labels=label_ids)
                    outputs_c = self.model(text_feats, video_feats, audio_feats, mode='train-mm', labels=label_ids)
                    
                    # 处理输出 - 支持聚类损失
                    if outputs_a is not None and len(outputs_a) >= 4:
                        _, mlp_output_a, contrastive_loss_a, clustering_loss_a = outputs_a[:4]
                    elif outputs_a is not None and len(outputs_a) >= 3:
                        _, mlp_output_a, contrastive_loss_a = outputs_a[:3]
                        clustering_loss_a = None
                    elif outputs_a is not None and len(outputs_a) >= 2:
                        _, mlp_output_a = outputs_a[:2]
                        contrastive_loss_a = None
                        clustering_loss_a = None
                    else:
                        # 如果输出为None，跳过这个batch
                        continue
                        
                    if outputs_b is not None and len(outputs_b) >= 4:
                        _, mlp_output_b, contrastive_loss_b, clustering_loss_b = outputs_b[:4]
                    elif outputs_b is not None and len(outputs_b) >= 3:
                        _, mlp_output_b, contrastive_loss_b = outputs_b[:3]
                        clustering_loss_b = None
                    elif outputs_b is not None and len(outputs_b) >= 2:
                        _, mlp_output_b = outputs_b[:2]
                        contrastive_loss_b = None
                        clustering_loss_b = None
                    else:
                        # 如果输出为None，跳过这个batch
                        continue
                        
                    if outputs_c is not None and len(outputs_c) >= 4:
                        _, mlp_output_c, contrastive_loss_c, clustering_loss_c = outputs_c[:4]
                    elif outputs_c is not None and len(outputs_c) >= 3:
                        _, mlp_output_c, contrastive_loss_c = outputs_c[:3]
                        clustering_loss_c = None
                    elif outputs_c is not None and len(outputs_c) >= 2:
                        _, mlp_output_c = outputs_c[:2]
                        contrastive_loss_c = None
                        clustering_loss_c = None
                    else:
                        # 如果输出为None，跳过这个batch
                        continue

                    norm_mlp_output_a = F.normalize(mlp_output_a)
                    norm_mlp_output_b = F.normalize(mlp_output_b)
                    norm_mlp_output_c = F.normalize(mlp_output_c)

                    contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1), norm_mlp_output_c.unsqueeze(1)), dim = 1)
                    loss_sup = self.contrast_criterion(contrastive_logits, labels = label_ids, temperature = args.train_temperature_sup, device = self.device)
                    
                    # 计算总损失
                    loss = loss_sup
                    
                    # 添加聚类损失
                    if clustering_loss_a is not None or clustering_loss_b is not None or clustering_loss_c is not None:
                        clustering_loss_total = 0.0
                        count = 0
                        
                        if clustering_loss_a is not None:
                            clustering_loss_total += clustering_loss_a
                            count += 1
                        if clustering_loss_b is not None:
                            clustering_loss_total += clustering_loss_b
                            count += 1
                        if clustering_loss_c is not None:
                            clustering_loss_total += clustering_loss_c
                            count += 1
                        
                        if count > 0:
                            clustering_loss_avg = clustering_loss_total / count
                            loss += getattr(args, 'clustering_weight', 1.0) * clustering_loss_avg
                    
                    # 添加对比学习损失
                    if contrastive_loss_a is not None or contrastive_loss_b is not None or contrastive_loss_c is not None:
                        contrastive_loss_total = 0.0
                        count = 0
                        
                        if contrastive_loss_a is not None:
                            contrastive_loss_total += contrastive_loss_a
                            count += 1
                        if contrastive_loss_b is not None:
                            contrastive_loss_total += contrastive_loss_b
                            count += 1
                        if contrastive_loss_c is not None:
                            contrastive_loss_total += contrastive_loss_c
                            count += 1
                        
                        if count > 0:
                            contrastive_loss_avg = contrastive_loss_total / count
                            loss += getattr(args, 'contrastive_weight', 0.5) * contrastive_loss_avg

                    self.optimizer.zero_grad()
                    loss.backward()


                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_sup_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()

                    torch.cuda.empty_cache()


            tr_sup_loss /= nb_tr_steps

            # 记录损失用于早停
            if not hasattr(self, 'recent_losses'):
                self.recent_losses = []
            self.recent_losses.append(tr_sup_loss)
            self.logger.info(f"Epoch {epoch}: Loss = {tr_sup_loss:.4f}, Recent losses: {self.recent_losses[-3:]}")

            # 保持最近20个epoch的损失记录
            if len(self.recent_losses) > 20:
                self.recent_losses = self.recent_losses[-20:]

            non_select_ids = [i for i in range(len(pseudo_labels)) if i not in select_ids]
            unsup_pseudo_labels = pseudo_labels[non_select_ids]

            if len(non_select_ids) != 0:

                self.train_outputs['select_ids'] = non_select_ids
                _, pseudo_unsup_train_dataloader = get_pseudo_dataloader(args=args, \
                    train_outputs=self.train_outputs, mode='pretrain')

                tr_unsup_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                
                for batch_unsup in tqdm(pseudo_unsup_train_dataloader, desc = 'Iteration'):

                    unsup_text_feats = batch_unsup['text_feats'].to(self.device)
                    unsup_video_feats = batch_unsup['video_feats'].to(self.device)
                    unsup_audio_feats = batch_unsup['audio_feats'].to(self.device)

                    with torch.set_grad_enabled(True):
    
                        _, mlp_output_a = self.model(unsup_text_feats, torch.zeros_like(unsup_video_feats).to(self.device), unsup_audio_feats, mode='train-mm')
                        _, mlp_output_b = self.model(unsup_text_feats, unsup_video_feats, torch.zeros_like(unsup_audio_feats).to(self.device), mode='train-mm')
                        _, mlp_output_c = self.model(unsup_text_feats, unsup_video_feats, unsup_audio_feats, mode='train-mm')

                        norm_mlp_output_a = F.normalize(mlp_output_a)
                        norm_mlp_output_b = F.normalize(mlp_output_b)
                        norm_mlp_output_c = F.normalize(mlp_output_c)

                        contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1), norm_mlp_output_c.unsqueeze(1)), dim = 1)
                        loss_unsup = self.contrast_criterion(contrastive_logits, temperature = args.train_temperature_unsup, device = self.device)
                    
                        loss = loss_unsup

                        self.optimizer.zero_grad()
                        loss.backward()


                        if args.grad_clip != -1.0:
                            torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                        tr_unsup_loss += loss.item()
                        nb_tr_steps += 1

                        self.optimizer.step()
                        self.scheduler.step()

                        torch.cuda.empty_cache()
                
                tr_unsup_loss /= nb_tr_steps

        if args.save_model:
            save_model(self.model, args.model_output_path)

    def _test(self, args):
        
        self.model.to(self.device)

        outputs = self._get_outputs(args, mode = 'test', return_feats = True)
        feats = outputs['feats']
        y_true = outputs['y_true']

        km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, \
                    init = 'k-means++').fit(feats) 
       
        y_pred = km.labels_
        
        test_results = clustering_score(y_true, y_pred)

        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def _get_outputs(self, args, mode, return_feats = False, modality = 'tva'):
        
        if mode == 'test':
   
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Get Outputs"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                if modality == 'tva':
                    features, _ = self.model(text_feats, video_feats, audio_feats, mode='train-mm')
                elif modality == 't0a':
                    features, _ = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats, mode='train-mm')
                elif modality == 'tv0':
                    features, _ = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device), mode='train-mm')


                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
        
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        if return_feats:
            outputs = {
                'y_true': y_true,
                'feats': feats
            }

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_pred = total_preds.cpu().numpy()
            
            y_logits = total_logits.cpu().numpy()
            
            outputs = {
                'y_true': y_true,
                'y_pred': y_pred,
                'logits': y_logits,
                'feats': feats
            }

        return outputs

    def load_pretrained_model(self, pretrained_model):
        
        pretrained_dict = pretrained_model.state_dict()
        mlp_params = ['method_model.mlp_head_train.2.weight', 'method_model.mlp_head_train.2.bias', 'method_model.classifier.weight', 'method_model.classifier.bias']
  
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in mlp_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def _evaluate_current_performance(self, args):
        """评估当前性能"""
        try:
            # 获取当前的特征和标签
            outputs = self._get_outputs(args, mode='train', return_feats=True)
            feats = outputs['feats']
            y_true = outputs['y_true']
            
            # 使用K-means进行聚类
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state=args.seed, init='k-means++')
            y_pred = km.fit_predict(feats)
            
            # 计算NMI作为性能指标
            from sklearn.metrics import normalized_mutual_info_score
            nmi = normalized_mutual_info_score(y_true, y_pred)
            return nmi
        except Exception as e:
            print(f"评估性能时出错: {e}")
            return 0.0

    def _get_current_loss(self, args):
        """获取当前损失"""
        try:
            # 如果有最近的损失记录，返回平均值
            if hasattr(self, 'recent_losses') and len(self.recent_losses) > 0:
                return np.mean(self.recent_losses[-5:])  # 返回最近5个epoch的平均损失
            else:
                return 1.0  # 默认损失值
        except Exception as e:
            print(f"获取损失时出错: {e}")
            return 1.0