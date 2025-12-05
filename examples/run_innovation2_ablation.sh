#!/usr/bin/bash

# 创新点二：文本引导注意力和自注意力机制消融实验
# 运行所有创新点二的消融实验

echo "开始运行创新点二：文本引导注意力和自注意力机制消融实验"

# 定义创新点二的消融实验列表
innovation2_experiments=(
    "no_text_guided_attention"
    "text_guided_only" 
    "self_attention_only"
    "full_attention_mechanism"
)

# 运行每个消融实验
for experiment in "${innovation2_experiments[@]}"
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
        --results_file_name "results_innovation2_${experiment}.csv" \
        --output_path "outputs/MIntRec/innovation2_${experiment}"
        
        echo "种子 $seed 完成"
    done
    
    echo "实验 $experiment 完成"
    echo ""
done

echo "所有创新点二注意力机制消融实验完成！"
