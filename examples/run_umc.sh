#!/usr/bin/bash

# MIntRec 数据集 - 创新点一机制对比实验
# 对比：双投影机制 (dual_projection_full) vs 简单线性映射 (dual_projection_simple_linear)
# 目的：在相同的其余配置下，仅改变投影机制，证明显式语义/环境解耦的优势

echo "运行 MIntRec 数据集：创新点一（子空间维度敏感性，简单线性基线）"
echo "=========================================="
echo "实验配置："
echo "  - 仅使用简单线性映射替代双投影（dual_projection_strategy=simple_linear）"
echo "  - 维度扫描：base_dim ∈ {128, 256, 384}"
echo "  - 创新点二：文本引导注意力（开启）"
echo "  - 创新点三：渐进式学习（开启）"
echo "提示：请先 cd 到包含 run_single_ablation.py 的项目根目录（例如 UMC-main），再执行：sh examples/run_umc.sh"
echo "=========================================="

# 如需更多 seed，可自行扩展此列表
for seed in 0 1 2
do
    for dim in 128 256 384
    do
        echo "=========================================="
        echo "运行 seed=${seed}, base_dim=${dim} (dual_projection_simple_linear_dim${dim})"
        echo "=========================================="

        python run_single_ablation.py \
            --experiment_name dual_projection_simple_linear_dim${dim} \
            --dataset MIntRec \
            --data_path 'Datasets' \
            --gpu '0' \
            --seed $seed \
            --train \
            --save_results \
            --save_model \
            --output_dir "ablation_results" \
            --log_path "logs" \
            --results_path "results" \
            --model_path "models" \
            --num_workers 1 \
            --config_file_name umc_MIntRec.py \
            --results_file_name "mintrec_simple_linear_dim${dim}_seed${seed}.csv"

        echo "seed ${seed}, base_dim ${dim} 完成"
        echo ""
    done
done

echo "=========================================="
echo "子空间维度敏感性（简单线性基线）实验完成！"
echo "结果输出目录：ablation_results/ （按文件名区分不同 dim 和 seed）"
echo "=========================================="
echo "  - 预期结果：双投影机制应在 NMI/ARI 上显著优于简单线性映射"
