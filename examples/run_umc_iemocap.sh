#!/usr/bin/bash

# IEMOCAP-DA 数据集 - 基线实验
# 运行完整的基线模型（不使用消融实验参数）

echo "运行 IEMOCAP-DA 数据集基线实验"

for seed in 0 1 2 3 4
do
    echo "运行种子: $seed"
    
    python run.py \
    --dataset IEMOCAP-DA \
    --data_path 'Datasets' \
    --logger_name umc \
    --multimodal_method umc \
    --method umc \
    --train \
    --tune \
    --save_model \
    --save_results \
    --seed $seed \
    --gpu_id '0' \
    --num_workers 2 \
    --video_feats_path 'swin_feats.pkl' \
    --audio_feats_path 'wavlm_feats.pkl' \
    --text_backbone bert-base-uncased \
    --config_file_name umc_IEMOCAP-DA \
    --results_file_name "results_baseline_IEMOCAP-DA.csv" \
    --output_path "outputs/IEMOCAP-DA/baseline"
    
    echo "种子 $seed 完成"
done

echo "基线实验完成！"

