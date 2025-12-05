#!/usr/bin/bash

# 创新点一：ConFEDE双投影机制消融实验
# 运行所有创新点一的消融实验

echo "开始运行创新点一：ConFEDE双投影机制消融实验"

# 定义创新点一的消融实验列表
innovation1_experiments=(
    "no_dual_projection"
    "video_dual_only" 
    "audio_dual_only"
    "full_dual_projection"
)

# 运行每个消融实验
for experiment in "${innovation1_experiments[@]}"
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
        --results_file_name "results_innovation1_${experiment}.csv" \
        --output_path "outputs/MIntRec/innovation1_${experiment}"
        
        echo "种子 $seed 完成"
    done
    
    echo "实验 $experiment 完成"
    echo ""
done

echo "所有创新点一双投影消融实验完成！"
