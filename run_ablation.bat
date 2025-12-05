@echo off
REM UMC消融实验批处理脚本

echo ========================================
echo UMC消融实验运行脚本
echo ========================================

REM 设置默认参数
set DATASET=MIntRec
set GPU=0
set CONFIG_FILE=umc_MIntRec.py

echo 可用的实验:
echo 1. baseline_traditional
echo 2. confede_only
echo 3. text_guided_only
echo 4. gated_fusion_only
echo 5. full_confede
echo 6. fixed_strategy
echo 7. progressive_only
echo 8. enhanced_loss_only
echo 9. full_progressive
echo 10. standard_architecture
echo 11. clustering_projector_only
echo 12. full_optimized_architecture
echo 13. only_confede
echo 14. only_progressive
echo 15. only_architecture
echo 16. full_umc_model

set /p CHOICE="请选择实验编号 (1-16): "

REM 根据选择设置实验名称
if "%CHOICE%"=="1" set EXPERIMENT_NAME=baseline_traditional
if "%CHOICE%"=="2" set EXPERIMENT_NAME=confede_only
if "%CHOICE%"=="3" set EXPERIMENT_NAME=text_guided_only
if "%CHOICE%"=="4" set EXPERIMENT_NAME=gated_fusion_only
if "%CHOICE%"=="5" set EXPERIMENT_NAME=full_confede
if "%CHOICE%"=="6" set EXPERIMENT_NAME=fixed_strategy
if "%CHOICE%"=="7" set EXPERIMENT_NAME=progressive_only
if "%CHOICE%"=="8" set EXPERIMENT_NAME=enhanced_loss_only
if "%CHOICE%"=="9" set EXPERIMENT_NAME=full_progressive
if "%CHOICE%"=="10" set EXPERIMENT_NAME=standard_architecture
if "%CHOICE%"=="11" set EXPERIMENT_NAME=clustering_projector_only
if "%CHOICE%"=="12" set EXPERIMENT_NAME=full_optimized_architecture
if "%CHOICE%"=="13" set EXPERIMENT_NAME=only_confede
if "%CHOICE%"=="14" set EXPERIMENT_NAME=only_progressive
if "%CHOICE%"=="15" set EXPERIMENT_NAME=only_architecture
if "%CHOICE%"=="16" set EXPERIMENT_NAME=full_umc_model

if "%EXPERIMENT_NAME%"=="" (
    echo 无效的选择！
    pause
    exit /b 1
)

echo.
echo 运行实验: %EXPERIMENT_NAME%
echo 数据集: %DATASET%
echo GPU: %GPU%
echo.

REM 运行实验
python run_single_ablation.py ^
    --experiment_name %EXPERIMENT_NAME% ^
    --dataset %DATASET% ^
    --gpu %GPU% ^
    --train ^
    --save_results ^
    --config_file_name %CONFIG_FILE%

if %ERRORLEVEL%==0 (
    echo.
    echo ✅ 实验成功完成！
) else (
    echo.
    echo ❌ 实验失败！
)

pause
