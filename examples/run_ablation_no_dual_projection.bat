@echo off
REM MIntRec 数据集 - 关闭创新点一（双投影机制）的消融实验
REM 关闭创新点一，保留创新点二和创新点三

echo ==========================================
echo 运行 MIntRec 数据集消融实验：关闭创新点一（双投影机制）
echo ==========================================

cd /d "d:\下载\UMC-main (9)\UMC-main"

for /L %%i in (0,1,4) do (
    echo ==========================================
    echo 运行种子: %%i
    echo ==========================================
    
    python run.py ^
    --dataset MIntRec ^
    --data_path "Datasets" ^
    --logger_name umc ^
    --multimodal_method umc ^
    --method umc ^
    --train ^
    --tune ^
    --save_model ^
    --save_results ^
    --seed %%i ^
    --gpu_id "0" ^
    --num_workers 2 ^
    --video_feats_path "swin_feats.pkl" ^
    --audio_feats_path "wavlm_feats.pkl" ^
    --text_backbone bert-base-uncased ^
    --config_file_name umc_MIntRec ^
    --ablation_experiment no_dual_projection ^
    --results_file_name "results_no_dual_projection_seed%%i.csv" ^
    --output_path "outputs/MIntRec/ablation_no_dual_projection/seed%%i"
    
    echo 种子 %%i 完成
    echo.
)

echo ==========================================
echo 消融实验完成！
echo ==========================================
echo 实验结果保存在: outputs/MIntRec/ablation_no_dual_projection/
echo.
echo 实验配置：
echo   - 创新点一（双投影机制）：关闭
echo   - 创新点二（文本引导注意力）：开启
echo   - 创新点三（渐进式学习）：开启
echo.
pause








