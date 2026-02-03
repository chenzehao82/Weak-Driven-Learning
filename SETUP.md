# Ensemble LLM 独立代码包设置说明

## 目录结构要求

为了运行本代码包，您需要确保以下目录结构：

```
EnsembleLLM_standalone/
├── ensemble_train.py
├── run_ensemble.sh
├── README.md
├── SETUP.md
└── [需要链接或复制的目录]
    ├── utils/              # 从原始项目复制或链接
    ├── Trainer/            # 从原始项目复制或链接
    ├── EnsembleQwen3/      # 从原始项目复制或链接（如果使用 Qwen3）
    ├── EnsembleQwen2/      # 从原始项目复制或链接（如果使用 Qwen2）
    └── run_entropy.py      # 从原始项目复制或链接
```

## 设置方法

### 方法 1: 创建符号链接（推荐）

如果您想保持与原始项目的同步，可以创建符号链接：

```bash
cd /root/buaa/czh/EnsembleLLM_standalone

# 链接必要的目录和文件
ln -s ../EnsembleLLM/utils utils
ln -s ../EnsembleLLM/Trainer Trainer
ln -s ../EnsembleLLM/EnsembleQwen3 EnsembleQwen3
ln -s ../EnsembleLLM/EnsembleQwen2 EnsembleQwen2
ln -s ../EnsembleLLM/run_entropy.py run_entropy.py
```

### 方法 2: 复制文件

如果您想要独立的代码包，可以复制必要的文件：

```bash
cd /root/buaa/czh/EnsembleLLM_standalone

# 复制必要的目录
cp -r ../EnsembleLLM/utils .
cp -r ../EnsembleLLM/Trainer .
cp -r ../EnsembleLLM/EnsembleQwen3 .
cp -r ../EnsembleLLM/EnsembleQwen2 .
cp ../EnsembleLLM/run_entropy.py .
```

### 方法 3: 修改 Python 路径

在运行脚本前，将原始项目路径添加到 PYTHONPATH：

```bash
export PYTHONPATH=/root/buaa/czh/EnsembleLLM:$PYTHONPATH
```

## 必需的依赖模块

### utils/ 目录
- `utils.py`: `load_model_tokenizer`, `load_data`, `load_entropy_df`
- `fuse_models.py`: 所有模型融合相关函数
- `weight_datasets.py`: `compute_sampling_weights_brownboost_style`, `compute_adaboost_sampling_weights`
- `load_dataset.py`: `load_math_dataset_jsonl`

### Trainer/ 目录
- `sft_runner.py`: `run_sft` 函数

### 模型定义
- `EnsembleQwen3/modeling_qwen3.py`: Qwen3 模型定义（如果使用 Qwen3）
- `EnsembleQwen2/`: Qwen2 模型定义（如果使用 Qwen2）

### 其他
- `run_entropy.py`: 熵计算脚本

## 配置文件

### accelerate_config.yaml

需要从原始项目复制 `scripts/accelerate_config.yaml`，或创建新的配置文件。

示例配置（DeepSpeed ZeRO-2）：
```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /path/to/deepspeed_config.json
  zero3_init_flag: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 验证设置

运行以下命令验证设置是否正确：

```bash
cd /root/buaa/czh/EnsembleLLM_standalone

# 检查 Python 导入
python -c "
from utils.utils import load_model_tokenizer
from utils.fuse_models import fuse_submodels
from utils.weight_datasets import compute_sampling_weights_brownboost_style
from Trainer.sft_runner import run_sft
print('✓ 所有模块导入成功')
"
```

如果出现导入错误，请检查：
1. 目录结构是否正确
2. PYTHONPATH 是否设置正确
3. 必要的文件是否存在

## 数据格式要求

### Stage 1 数据格式
JSONL 格式，每行一个 JSON 对象，包含：
- `instruction`: 指令/问题
- `output`: 输出/答案
- 其他字段（根据实际需求）

### Stage 2/3 数据格式
JSONL 格式，每行一个 JSON 对象，必须包含：
- `idx`: 样本索引（用于熵匹配）
- 其他字段（根据实际需求）

### Entropy 文件格式
JSONL 格式，每行一个 JSON 对象，包含：
- `idx`: 样本索引
- `entropy_0`, `entropy_1`, `entropy_2` 等：不同阶段的熵值

## 常见问题

### Q: 导入错误 "Cannot locate Qwen3ForEnsemble"
A: 确保 `EnsembleQwen3/` 或 `EnsembleQwen2/` 目录存在，并且 Python 路径正确。

### Q: 找不到 `run_entropy.py`
A: 确保从原始项目复制或链接了 `run_entropy.py` 文件。

### Q: 找不到 `accelerate_config.yaml`
A: 从原始项目的 `scripts/` 目录复制配置文件，或创建新的配置文件。

### Q: GPU 内存不足
A: 调整以下参数：
- 减小 `--per-device-train-batch-size`
- 增大 `--grad-accum`
- 减小 `--max-seq-length`
- 使用更少的 GPU（修改 `CUDA_VISIBLE_DEVICES`）

## 下一步

设置完成后，请参考 `README.md` 了解如何使用本代码包进行训练。

