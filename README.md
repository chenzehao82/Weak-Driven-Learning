## Ensemble LLM（WMSS）训练与评测框架

本项目提供一个**可一键跑通**的多阶段训练 Pipeline，用于训练/融合两个子模型，并在训练结束后自动**提取第一个子模型**并做**评测**。整体组织方式参考 RAGEN 的“模块化 + 脚本化 Pipeline + 清晰文档”标准（参考：[`https://github.com/mll-lab-nu/RAGEN`](https://github.com/mll-lab-nu/RAGEN)）。

### 核心概念

- **WMSS**：Weighted Model Selection and Synthesis（加权模型选择与合成）。在本项目中表示“vote”融合方式的统一命名。
- **Entropy / BrownBoost 风格加权采样**：通过不同阶段的熵（entropy）差异，给训练样本赋权并重采样，提升后续阶段训练效果。

---

### 目录结构（高层）

（以当前项目为准，关键目录如下）

- **`scripts/`**：一键 Pipeline 脚本（入口）
- **`ensemble/`**：训练/熵计算/提取/评测等主脚本
- **`utils/`**：模型加载、融合、熵计算、数据处理等通用工具
- **`Trainer/`**：SFT 训练 runner 与 trainer
- **`weights/`**：输出权重与 checkpoint（运行后生成）
- **`logs/`**、**`tensorboard_logs/`**：日志（运行后生成）

---

### 快速开始（推荐）

1) **进入项目根目录**（非常重要：避免 `utils` 导入失败）

```bash
cd "/root/buaa/czh/Weak-Driving Learning"
```

2) **修改训练与数据路径**

编辑 `scripts/run_ensemble.sh`，按需修改：
- **`GPU_USE`**
- **`base_model`**
- **`stage1_data_path` / `data_files`**
- **`outdir`**
- 训练超参（epochs、batch、grad-accum、max-seq-length 等）

3) **一键运行**

```bash
bash scripts/run_ensemble.sh
```

---

### Pipeline 做了什么（按顺序）

- **Step 0**：计算 base 模型 `entropy_0`
- **Step 1**：Stage1 训练，得到 `m1`（目录：`$outdir/stage1_m1`）
- **Step 2**：计算 `m1` 的 `entropy_1`
- **Step 3**：合并 `entropy_0` + `entropy_1` → `entropy_merged_stage1.jsonl`
- **Step 4**：制作 `stage0_m0`（复制 base 模型到训练输出目录下，作为融合用的 m0）
- **Step 5**：Stage3 训练（融合 `m0 + m1`，并用加权采样数据继续训练），得到最终模型
- **Step 6**：从最终融合模型中**提取第一个子模型**（`submodel_idx=0`）
- **Step 7**：评测提取出的模型（`eval_vllm_thinking_math.py`）

---

### 常见错误与排查

- **报错：`No module named 'utils'`**
  - **原因**：不是在项目根目录运行、或 Python 路径未包含项目根目录。
  - **解决**：
    - 确保先 `cd` 到项目根目录再跑：`cd "/root/buaa/czh/Weak-Driving Learning"`
    - `scripts/run_ensemble.sh` 内已使用相对路径调用 `../ensemble/*.py`，请不要手动从其他目录运行那些脚本。

- **报错：`python: command not found`**
  - **原因**：系统没有 `python` 软链。
  - **解决**：用 `python3` 或在 conda 环境中使用 `python`。

---

### 评测结果输出

评测脚本会将结果写到 `results/` 目录（如果脚本中配置如此），同时日志会写到 `logs/`。

---

### 参考

- 组织与文档风格参考：[`mll-lab-nu/RAGEN`](https://github.com/mll-lab-nu/RAGEN)
