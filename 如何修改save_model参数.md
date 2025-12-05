# 如何修改 save_model 参数

## 一、方法1：命令行直接添加（推荐）

### 在运行命令时直接添加参数

```bash
# Windows PowerShell 或 CMD
python run.py --dataset MELD-DA --train --save_model --gpu_id '0'

# Linux/Mac
python run.py --dataset MELD-DA --train --save_model --gpu_id '0'
```

**优点**：
- ✅ 灵活，可以随时决定是否保存
- ✅ 不需要修改文件
- ✅ 推荐方式

---

## 二、方法2：修改运行脚本（.sh 文件）

### 修改示例脚本

编辑 `examples/run_umc.sh` 或其他 `.sh` 脚本：

```bash
# 修改前
python run.py \
    --dataset MIntRec \
    --train \
    --tune \
    --save_results \
    --seed $seed \
    ...

# 修改后（添加 --save_model）
python run.py \
    --dataset MIntRec \
    --train \
    --tune \
    --save_model \        # ← 添加这一行
    --save_results \
    --seed $seed \
    ...
```

### 完整示例

```bash
#!/usr/bin/bash

for seed in 0 1 2 3 4
do
    python run.py \
    --dataset MELD-DA \
    --data_path 'Datasets' \
    --train \
    --save_model \        # ← 添加这里
    --save_results \
    --seed $seed \
    --gpu_id '0' \
    --config_file_name umc_MELD-DA
done
```

---

## 三、方法3：修改批处理文件（.bat 文件，Windows）

### 修改 .bat 文件

编辑 `run_ablation.bat` 或其他 `.bat` 文件：

```batch
@echo off
python run.py ^
    --dataset MELD-DA ^
    --train ^
    --save_model ^        REM ← 添加这一行
    --save_results ^
    --gpu_id 0
```

**注意**：Windows 批处理文件使用 `^` 作为换行符，使用 `REM` 添加注释。

---

## 四、方法4：修改代码默认值（不推荐）

### 如果想默认启用 save_model

修改 `run.py` 文件：

```python
# 修改前
parser.add_argument("--save_model", action="store_true", 
                   help="save trained-model for multimodal intent recognition.")

# 修改后（不推荐，因为会强制所有实验都保存）
# 这种方式不推荐，因为会强制所有运行都保存模型
```

**为什么不推荐**：
- ❌ 会强制所有实验都保存模型
- ❌ 占用大量磁盘空间
- ❌ 不够灵活

---

## 五、实际使用示例

### 示例1：命令行直接使用

```bash
# 基本训练并保存
python run.py \
    --dataset MELD-DA \
    --config_file_name umc_MELD-DA \
    --train \
    --save_model \
    --gpu_id '0'
```

### 示例2：修改现有脚本

#### 修改 `examples/run_umc.sh`

```bash
# 打开文件
vim examples/run_umc.sh
# 或
nano examples/run_umc.sh
# 或直接用编辑器打开

# 找到这一行（大约第18行）
    --train \

# 在下面添加
    --save_model \
```

修改后的片段：
```bash
    python run.py \
    --dataset MIntRec \
    --data_path 'Datasets' \
    --train \
    --save_model \        # ← 新添加的
    --tune \
    --save_results \
    --seed $seed \
    ...
```

### 示例3：创建新的脚本文件

创建 `run_meld_da.sh`：

```bash
#!/usr/bin/bash

echo "运行 MELD-DA 数据集实验"

python run.py \
    --dataset MELD-DA \
    --data_path 'Datasets' \
    --train \
    --save_model \        # 保存模型
    --save_results \      # 保存结果
    --seed 0 \
    --gpu_id '0' \
    --config_file_name umc_MELD-DA \
    --video_feats_path 'swin_feats.pkl' \
    --audio_feats_path 'wavlm_feats.pkl' \
    --text_backbone bert-base-uncased

echo "实验完成！"
```

然后运行：
```bash
chmod +x run_meld_da.sh
./run_meld_da.sh
```

---

## 六、检查是否生效

### 方法1：查看输出目录

运行后检查是否有模型文件生成：

```bash
# Linux/Mac
ls outputs/*/models/pytorch_model.bin

# Windows PowerShell
Get-ChildItem -Recurse -Filter "pytorch_model.bin" outputs/
```

### 方法2：查看日志

训练日志中会显示：
```
Trained models are saved in outputs/...
```

### 方法3：运行时检查

如果添加了 `--save_model`，训练完成后会看到：
```
Trained models are saved in outputs/umc_umc_MELD-DA_bert-base-uncased_0/models
```

---

## 七、快速参考

### 命令行方式（最简单）

```bash
# 添加 --save_model 参数即可
python run.py --dataset MELD-DA --train --save_model --gpu_id '0'
```

### 脚本方式（适合批量实验）

```bash
# 在脚本中添加
--save_model \
```

### 完整命令示例

```bash
python run.py \
    --dataset MELD-DA \
    --config_file_name umc_MELD-DA \
    --data_path 'Datasets' \
    --train \
    --save_model \        # ← 添加这个
    --save_results \
    --seed 0 \
    --gpu_id '0'
```

---

## 八、常见问题

### Q1: 添加了参数但还是没保存？

**检查**：
1. 确认参数拼写正确：`--save_model`（两个短横线）
2. 确认训练完成（模型只在训练完成后保存）
3. 检查输出目录权限

### Q2: 想禁用 save_model？

**方法**：不添加 `--save_model` 参数即可（默认就是 False）

### Q3: 批量实验时如何控制？

**方法**：在循环脚本中添加或移除 `--save_model` 参数

```bash
# 保存所有实验的模型
for seed in 0 1 2 3 4
do
    python run.py --train --save_model --seed $seed ...
done

# 只保存特定实验的模型
python run.py --train --save_model --seed 0 ...
python run.py --train --seed 1 ...  # 这个不保存
```

---

## 总结

### 推荐方式

1. **命令行直接添加**（最灵活）
   ```bash
   python run.py --train --save_model
   ```

2. **修改脚本文件**（适合重复运行）
   ```bash
   # 在 .sh 或 .bat 文件中添加
   --save_model \
   ```

### 修改位置总结

| 方式 | 修改位置 | 适用场景 |
|------|---------|---------|
| 命令行 | 直接添加参数 | 单次运行 |
| 脚本文件 | `examples/*.sh` | 批量实验 |
| 批处理文件 | `*.bat` | Windows批量运行 |
| 代码默认值 | `run.py` | 不推荐 |

**最简单的方法**：在运行命令时直接添加 `--save_model` 参数！

