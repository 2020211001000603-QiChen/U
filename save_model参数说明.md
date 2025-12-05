# save_model 参数说明

## 一、什么是 save_model？

`save_model` 是一个**布尔标志参数**（flag），用于控制是否在训练完成后保存训练好的模型。

### 参数定义

```python
parser.add_argument("--save_model", action="store_true", 
                   help="save trained-model for multimodal intent recognition.")
```

- **类型**：`action="store_true"` - 布尔标志，添加参数即为 True，不添加为 False
- **默认值**：False（如果不指定该参数，模型不会被保存）
- **作用**：训练完成后，将模型权重保存到磁盘

## 二、save_model 的工作原理

### 1. 触发时机

`save_model` 在训练完成后被检查：

```python
# 在 methods/unsupervised/UMC/manager.py 中
def _train(self, args):
    # ... 训练过程 ...
    
    if args.save_model:  # 训练完成后检查
        save_model(self.model, args.model_output_path)
```

### 2. 保存的内容

保存的是**模型的权重参数**（state_dict），而不是完整的模型对象：

```python
# utils/functions.py
def save_model(model, model_dir):
    # 如果是DataParallel模型，提取实际模型
    save_model = model.module if hasattr(model, 'module') else model 
    model_file = os.path.join(model_dir, 'pytorch_model.bin')
    
    # 只保存模型权重，不保存优化器、调度器等
    torch.save(save_model.state_dict(), model_file)
```

### 3. 保存位置

模型保存路径由以下参数决定：

```
{output_path}/{save_model_name}/{model_path}/pytorch_model.bin
```

其中：
- `output_path`：通过 `--output_path` 参数指定（默认空字符串）
- `save_model_name`：自动生成，格式为：
  - 正常训练：`{method}_{multimodal_method}_{dataset}_{text_backbone}_{seed}`
  - 消融实验：`ablation_{experiment_name}_{dataset}_{seed}`
- `model_path`：通过 `--model_path` 参数指定（默认 `'models'`）

**示例路径**：
```
outputs/umc_umc_MELD-DA_bert-base-uncased_0/models/pytorch_model.bin
```

## 三、save_model 的使用场景

### 1. **需要后续使用训练好的模型**

```bash
# 训练并保存模型
python run.py \
    --dataset MELD-DA \
    --train \
    --save_model \
    --gpu_id '0'
```

### 2. **模型推理/测试**

保存模型后，可以在不重新训练的情况下进行测试：

```python
# 后续可以加载模型进行推理
from utils.functions import restore_model

model = restore_model(model, model_dir, device)
```

### 3. **模型检查点（Checkpoint）**

虽然当前实现只保存最终模型，但可以用于：
- 保存最佳模型
- 模型部署
- 继续训练（需要配合优化器状态）

### 4. **模型对比实验**

保存多个不同配置的模型，便于对比：

```bash
# 实验1
python run.py --dataset MELD-DA --seed 0 --save_model

# 实验2  
python run.py --dataset MELD-DA --seed 1 --save_model

# 每个实验的模型会保存在不同目录
```

## 四、save_model vs save_results

### 区别对比

| 参数 | 保存内容 | 用途 | 文件位置 |
|------|---------|------|---------|
| `--save_model` | 模型权重（`.bin`文件） | 模型推理、部署、继续训练 | `{output_path}/{model_name}/models/pytorch_model.bin` |
| `--save_results` | 测试结果（`.csv`文件） | 性能分析、结果对比 | `{results_path}/{results_file_name}` |

### 同时使用

```bash
# 同时保存模型和结果
python run.py \
    --dataset MELD-DA \
    --train \
    --save_model \      # 保存模型
    --save_results \    # 保存测试结果
    --gpu_id '0'
```

## 五、模型加载和恢复

### 1. 加载保存的模型

```python
from utils.functions import restore_model

# 加载模型权重
model = restore_model(model, model_dir, device)
```

### 2. 代码中的使用

在训练代码中，如果指定了 `--pretrain` 但没有训练，会尝试从保存的模型加载：

```python
# methods/unsupervised/UMC/manager.py
if args.pretrain:
    self.pretrained_model = pretrain_manager.model
else:
    # 从保存的模型恢复
    self.pretrained_model = restore_model(
        pretrain_manager.model, 
        os.path.join(args.model_output_path, 'pretrain'), 
        self.device
    )
```

## 六、注意事项

### 1. **磁盘空间**

模型文件可能很大（几百MB到几GB），取决于：
- 模型架构复杂度
- 参数量
- 是否使用多GPU训练（DataParallel）

**建议**：
- 定期清理不需要的模型文件
- 只保存重要的实验模型

### 2. **模型版本管理**

保存的模型文件名相同，新训练会覆盖旧模型：

```
pytorch_model.bin  # 每次训练都会覆盖
```

**建议**：
- 重要实验使用不同的 `--output_path` 或 `--seed`
- 或者手动重命名模型文件

### 3. **只保存权重**

当前实现**只保存模型权重**，不保存：
- 优化器状态
- 训练进度（epoch数）
- 学习率调度器状态
- 其他训练状态

**影响**：
- 可以用于推理和测试 ✅
- 可以用于微调 ✅
- **不能直接恢复训练进度** ❌（需要重新初始化优化器等）

### 4. **DataParallel 兼容性**

代码已经处理了多GPU训练的情况：

```python
# 如果是DataParallel，会自动提取实际模型
save_model = model.module if hasattr(model, 'module') else model
```

## 七、使用示例

### 示例1：基本训练并保存

```bash
python run.py \
    --dataset MELD-DA \
    --config_file_name umc_MELD-DA \
    --train \
    --save_model \
    --gpu_id '0'
```

### 示例2：保存到指定目录

```bash
python run.py \
    --dataset MELD-DA \
    --train \
    --save_model \
    --output_path "saved_models/MELD-DA_experiment1" \
    --gpu_id '0'
```

### 示例3：消融实验保存

```bash
python run.py \
    --dataset MELD-DA \
    --ablation_experiment no_dual_projection \
    --train \
    --save_model \
    --gpu_id '0'
```

模型会保存到：
```
outputs/ablation_no_dual_projection_MELD-DA_0/models/pytorch_model.bin
```

### 示例4：不保存模型（只训练和测试）

```bash
python run.py \
    --dataset MELD-DA \
    --train \
    --save_results \  # 只保存结果，不保存模型
    --gpu_id '0'
```

## 八、常见问题

### Q1: 保存的模型文件在哪里？

**A**: 默认在 `outputs/{model_name}/models/pytorch_model.bin`

### Q2: 如何知道模型保存成功？

**A**: 检查文件是否存在：
```bash
ls outputs/*/models/pytorch_model.bin
```

### Q3: 可以只保存最佳模型吗？

**A**: 当前实现只保存最终模型。如果需要保存最佳模型，需要修改代码添加验证逻辑。

### Q4: 模型文件太大怎么办？

**A**: 
- 使用模型压缩技术
- 只保存必要的层
- 考虑使用量化模型

### Q5: 保存的模型可以用于其他数据集吗？

**A**: 通常不行，因为：
- 模型架构可能不同（类别数不同）
- 特征维度可能不同
- 但可以作为预训练模型进行迁移学习

## 九、最佳实践

### 1. **实验管理**

```bash
# 为每个重要实验创建独立的输出目录
python run.py --dataset MELD-DA --output_path "exp1_baseline" --save_model
python run.py --dataset MELD-DA --output_path "exp2_ablation" --save_model
```

### 2. **模型命名规范**

使用描述性的输出路径：
```bash
--output_path "MELD-DA_umc_seed0_$(date +%Y%m%d)"
```

### 3. **定期清理**

```bash
# 删除不需要的模型文件
find outputs -name "pytorch_model.bin" -mtime +30 -delete
```

### 4. **版本控制**

重要模型应该：
- 记录训练配置
- 记录训练日志
- 备份到其他位置

## 总结

- **save_model** = 训练完成后保存模型权重
- **作用**：保存训练好的模型供后续使用
- **保存内容**：模型权重（state_dict），不包含优化器状态
- **保存位置**：`{output_path}/{model_name}/models/pytorch_model.bin`
- **使用场景**：模型推理、部署、后续实验
- **注意事项**：磁盘空间、文件覆盖、只保存权重
- **最佳实践**：为重要实验创建独立目录，定期清理不需要的模型

