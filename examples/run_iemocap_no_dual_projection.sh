#!/usr/bin/bash

# IEMOCAP-DA 数据集 - 消融实验：关闭创新点一（双投影/ConFEDE）
# 关闭创新点一，保留创新点二（三）用于评估双投影机制贡献

echo "运行 IEMOCAP-DA 数据集消融实验：关闭创新点一"

for seed in 0 1 2 3 4
do
    echo "=========================================="
    echo "运行种子: $seed"
    echo "=========================================="
    
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
    --ablation_experiment no_dual_projection \
    --results_file_name "iemocap_no_dual_projection_seed${seed}.csv" \
    --output_path "outputs/IEMOCAP-DA/ablation_no_dual_projection/seed${seed}"
    
    echo "种子 $seed 完成"
    echo ""
done

echo "=========================================="
echo "消融实验完成！"
echo "=========================================="
echo "实验结果保存在: outputs/IEMOCAP-DA/ablation_no_dual_projection/"
echo ""
echo "实验配置："
echo "  - 创新点一（双投影机制）：关闭"
echo "  - 创新点二（文本引导注意力）：开启"
echo "  - 创新点三（聚类/渐进式学习）：开启"

