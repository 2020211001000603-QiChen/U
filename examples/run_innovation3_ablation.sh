#!/usr/bin/bash

# 创新点三：聚类优化架构消融实验
# 运行所有创新点三的消融实验

echo "开始运行创新点三：聚类优化架构消融实验"

# 定义创新点三的消融实验列表
innovation3_experiments=(
    "baseline_no_clustering"
    "clustering_projector_only" 
    "clustering_fusion_only"
    "clustering_loss_only"
    "clustering_projector_fusion"
    "clustering_projector_loss"
    "clustering_fusion_loss"
    "full_clustering_optimization"
)

# 运行每个消融实验
for experiment in "${innovation3_experiments[@]}"
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
        --logger_name umc \
        --multimodal_method umc \
        --method umc \
        --train \
        --tune \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'swin_feats.pkl' \
        --audio_feats_path 'wavlm_feats.pkl' \
        --text_backbone bert-base-uncased \
        --config_file_name umc_MIntRec \
        --ablation_experiment $experiment \
        --results_file_name "results_innovation3_${experiment}.csv" \
        --output_path "outputs/MIntRec/innovation3_${experiment}"
        
        echo "种子 $seed 完成"
    done
    
    echo "实验 $experiment 完成"
    echo ""
done

echo "所有创新点三聚类优化架构消融实验完成！"