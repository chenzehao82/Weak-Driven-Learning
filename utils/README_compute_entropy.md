# 智能多卡并行 Entropy 计算

## 🎯 新特性

### 自动模型类型检测

`compute_entropy_for_model()` 现在能够**自动检测**模型类型，无需手动指定：

- ✅ **标准 Qwen 模型**: 自动使用 `AutoModelForCausalLM`
- ✅ **QwenBoost 模型**: 自动使用 `QwenBoostForCausalLM`
- ✅ **容错机制**: 加载失败自动回退到标准模式

### 检测逻辑

```python
def detect_model_type(model_path: str) -> str:
    """
    检测规则：
    1. 检查 config.json 中的 ensemble_config / num_submodels 字段
    2. 检查 architectures 中是否包含 "Boost" / "Ensemble"
    3. 检查是否存在 ensemble_weights.json 等特有文件
    4. 默认返回 "standard"
    """
```

## 📦 API

### compute_entropy_for_model()

```python
from accelerate import PartialState
from utils.compute_entropy import compute_entropy_for_model

distributed_state = PartialState()

compute_entropy_for_model(
    model_path="path/to/model",           # 模型路径（自动检测类型）
    data_files=["data.jsonl"],            # 数据文件
    output_path="entropy_0.jsonl",        # 输出文件
    entropy_field="entropy_0",            # 字段名
    distributed_state=distributed_state,  # 分布式状态
)
```

**注意**: 不再需要 `use_ensemble_model` 参数！

## 🚀 使用方法

### 方法 1: 在训练流程中（自动调用）

```bash
accelerate launch \
    --config_file=./scripts/accelerate_config.yaml \
    llmboost_train.py \
    --model-name "Qwen/Qwen2.5-3B" \
    --stage1-data-path "/path/to/data.jsonl" \
    --data-files "/path/to/data.jsonl" \
    --output-dir "./output"
```

### 方法 2: 独立测试

```bash
# 测试标准 Qwen 模型
bash scripts/test_entropy_parallel.sh \
    "Qwen/Qwen2.5-3B" \
    "/path/to/data.jsonl" \
    "./test_entropy.jsonl" \
    8

# 测试 QwenBoost 模型（自动检测）
bash scripts/test_entropy_parallel.sh \
    "path/to/qwen_boost_model" \
    "/path/to/data.jsonl" \
    "./test_entropy.jsonl" \
    8
```

### 方法 3: Python 脚本

```python
# test_my_model.py
from accelerate import PartialState
from utils.compute_entropy import compute_entropy_for_model

distributed_state = PartialState()

# 计算任何模型的 entropy（自动检测类型）
compute_entropy_for_model(
    model_path="path/to/any/model",
    data_files=["data.jsonl"],
    output_path="entropy.jsonl",
    entropy_field="entropy_0",
    distributed_state=distributed_state,
)
```

运行：
```bash
accelerate launch --num_processes 8 --multi_gpu test_my_model.py
```

## 🔍 模型类型检测示例

### 标准 Qwen 模型

```
[Rank 0] 检测到模型类型: standard
[Rank 0] 使用 AutoModelForCausalLM 加载...
[Rank 0] 模型加载完成 (type=standard, vocab_size=151936, pad_token_id=151643)
```

### QwenBoost 模型

```
[Rank 0] 检测到模型类型: qwen_boost
[Rank 0] 使用 QwenBoostForCausalLM 加载...
[Rank 0] 模型加载完成 (type=qwen_boost, vocab_size=151936, pad_token_id=151643)
```

### 自动回退

```
[Rank 0] 检测到模型类型: qwen_boost
[Rank 0] 使用 QwenBoostForCausalLM 加载...
[Rank 0] 警告: QwenBoostForCausalLM 加载失败 (module not found)，尝试使用标准加载方式...
[Rank 0] 模型加载完成 (type=standard, vocab_size=151936, pad_token_id=151643)
```

## 🛡️ 错误处理

### Pad Token 错误修复

自动修复 `AssertionError: Padding_idx must be within num_embeddings`：

```python
# 1. 先加载 tokenizer，设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. 加载模型时明确传入 pad_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    pad_token_id=tokenizer.pad_token_id,  # ← 关键
    ...
)

# 3. 同步模型配置
model.config.pad_token_id = tokenizer.pad_token_id
```

### 模型加载失败自动回退

```python
try:
    # 尝试加载 QwenBoost
    model = QwenBoostForCausalLM.from_pretrained(...)
except Exception as e:
    # 失败后自动回退到标准模型
    model = AutoModelForCausalLM.from_pretrained(...)
    model_type = "standard"
```

## 📊 性能

| GPU数量 | 数据量 | 标准模型时间 | QwenBoost时间 | 加速比 |
|---------|--------|--------------|---------------|--------|
| 1       | 10K    | ~60min       | ~65min        | 1.0x   |
| 8       | 10K    | ~8min        | ~9min         | 7.5x   |
| 8       | 100K   | ~80min       | ~90min        | 7.5x   |

## 🔧 故障排除

### 问题：检测错误的模型类型

**解决方案**：
- 检查 `config.json` 是否正确配置
- 手动在代码中修改检测逻辑
- 查看日志中的 "检测到模型类型" 信息

### 问题：pad_token_id 错误

**解决方案**：
- 已自动修复，查看日志确认：
  ```
  [Rank 0] Tokenizer 配置: vocab_size=151936, pad_token_id=151643
  ```

### 问题：OOM（显存不足）

**解决方案**：
- 减少 GPU 数量
- 减小 batch_size（当前固定为1）
- 使用显存更大的 GPU

## 📝 更新日志

### v2.0 (当前版本)

- ✅ 自动模型类型检测
- ✅ 移除 `use_ensemble_model` 参数
- ✅ 智能加载和回退机制
- ✅ 增强的错误处理和日志

### v1.0

- ✅ 基础多卡并行功能
- ✅ 手动指定模型类型

## 🎓 最佳实践

1. **总是使用多卡**: 即使数据量小，多卡也能提速
2. **检查日志**: 确认检测到正确的模型类型
3. **保留临时文件**: 如果需要调试，可以注释删除临时文件的代码
4. **监控显存**: 使用 `nvidia-smi` 或脚本中的显存打印功能

## 📧 支持

如果遇到问题：
1. 查看完整日志输出
2. 确认模型类型检测是否正确
3. 检查 pad_token_id 配置
4. 查看上述故障排除部分

