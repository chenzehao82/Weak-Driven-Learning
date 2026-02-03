# 快速开始指南

## 1. 设置环境

### 创建符号链接（推荐）
```bash
cd /root/buaa/czh/EnsembleLLM_standalone

# 链接必要的目录和文件
ln -s ../EnsembleLLM/utils utils
ln -s ../EnsembleLLM/Trainer Trainer
ln -s ../EnsembleLLM/EnsembleQwen3 EnsembleQwen3
ln -s ../EnsembleLLM/EnsembleQwen2 EnsembleQwen2
ln -s ../EnsembleLLM/run_entropy.py run_entropy.py
```

### 验证设置
```bash
python -c "
from utils.utils import load_model_tokenizer
from utils.fuse_models import fuse_submodels
print('✓ 设置成功')
"
```

## 2. 修改配置

编辑 `run_ensemble.sh`，修改以下参数：

```bash
# GPU 配置
GPU_USE=0,1,2,3,4,5,6,7

# 模型和数据路径
outdir="weights/ensemble/Qwen2.5-3B"
base_model="Qwen/Qwen2.5-3B"
stage1_data_path="/path/to/stage1_data.jsonl"
data_files="/path/to/data.jsonl"

# 训练参数
stage1_epochs=1
stage2_epochs=1
stage3_epochs=1

# 模型类型
model_type="wmss"  # Weighted Model Selection and Synthesis
```

## 3. 运行完整 Pipeline

```bash
bash run_ensemble.sh
```

## 4. 分阶段运行（可选）

### 仅运行 Stage 1
```bash
conda activate qwen
accelerate launch \
    --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
    ensemble_train.py \
    --stage 1 \
    --model-name "Qwen/Qwen2.5-3B" \
    --stage1-data-path "/path/to/stage1_data.jsonl" \
    --data-files "/path/to/data.jsonl" \
    --output-dir "./weights/ensemble" \
    --stage1-num-epochs 1
```

## 文件说明

- `ensemble_train.py`: 主训练脚本
- `run_ensemble.sh`: 完整 Pipeline 脚本
- `README.md`: 详细文档
- `SETUP.md`: 设置说明
- `QUICKSTART.md`: 本文件

## 需要帮助？

- 详细设置说明：查看 `SETUP.md`
- 完整文档：查看 `README.md`
- 参数说明：运行 `python ensemble_train.py --help`

