#!/usr/bin/bash

# UMC内存优化训练脚本
# 使用内存优化配置进行训练

for seed in 0 1 2 3 4
do
    for multimodal_method in 'umc'
    do
        for method in 'umc'
        do 
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in 'MIntRec' # 'MELD-DA' 'IEMOCAP-DA'
                do
                    echo "🚀 开始训练 UMC (内存优化版本) - Seed: $seed"
                    
                    python run.py \
                    --dataset $dataset \
                    --data_path 'Datasets' \
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method\
                    --train \
                    --tune \
                    --save_results \
                    --seed $seed \
                    --gpu_id '0' \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name umc_MIntRec_memory_optimized \
                    --results_file_name "results_umc_memory_optimized.csv" \
                    --output_path "outputs/${dataset}_memory_optimized" \
                    --num_workers 2 \
                    --log_path 'logs_memory_optimized'
                    
                    echo "✅ 训练完成 - Seed: $seed"
                done
            done
        done
    done
done

echo "🎉 所有训练任务完成！"
