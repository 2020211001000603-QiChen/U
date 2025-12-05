@echo off
echo 运行创新点三消融实验 - 基线实验

cd /d "d:\下载\UMC-main (9)\UMC-main"

for /L %%i in (0,1,4) do (
    echo 运行种子: %%i
    
    python run.py ^
    --dataset MIntRec ^
    --data_path "Datasets" ^
    --logger_name umc ^
    --multimodal_method umc ^
    --method umc ^
    --train ^
    --tune ^
    --save_results ^
    --seed %%i ^
    --gpu_id "0" ^
    --video_feats_path "swin_feats.pkl" ^
    --audio_feats_path "wavlm_feats.pkl" ^
    --text_backbone bert-base-uncased ^
    --config_file_name umc_MIntRec ^
    --ablation_experiment baseline_no_clustering ^
    --results_file_name "results_innovation3_baseline_no_clustering.csv" ^
    --output_path "outputs/MIntRec/innovation3_baseline_no_clustering"
    
    echo 种子 %%i 完成
)

echo 基线实验完成！
pause
