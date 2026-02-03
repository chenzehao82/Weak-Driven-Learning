# Weak-Driven Learning

Weak-Driven Learning leverages entropy-based weighted sampling and ensemble model fusion to train powerful language models through multi-stage training pipelines. This framework provides a **one-command** training pipeline that automatically trains and fuses sub-models, extracts the enhanced sub-model, and evaluates its performance.

The project follows the modular design principles of [RAGEN](https://github.com/mll-lab-nu/RAGEN) with a "modular + scripted pipeline + clear documentation" approach.

## Overview

Weak-Driven Learning implements a multi-stage training framework that:

- **Trains ensemble models** using Weighted Model Selection and Synthesis (WMSS)
- **Applies entropy-based weighted sampling** inspired by BrownBoost to focus training on challenging samples
- **Automatically extracts and evaluates** the enhanced sub-model after training

### Core Concepts

- **WMSS (Weighted Model Selection and Synthesis)**: A unified naming for the "vote" fusion method that combines multiple sub-models with learned weights
- **Entropy-based Weighted Sampling**: Uses entropy differences across training stages to assign weights and resample training data, improving subsequent stage performance
- **Multi-stage Training Pipeline**: A three-stage process that progressively refines model capabilities through entropy-guided training

## Framework Overview

The following diagram illustrates the overall paradigm of Weak-Driven Learning:

<p align="center">
  <img src="pics/weak-drivenlearning.png" alt="Weak-Driven Learning Framework" width="800"/>
</p>

## Model Architecture

The model architecture diagram shows the ensemble structure:

<p align="center">
  <img src="pics/pivotFig.pdf" alt="Model Architecture" width="600"/>
</p>

## Quick Start

### Prerequisites

- Python >= 3.8
- CUDA-capable GPUs (recommended: 8 GPUs for full pipeline)
- Conda environment (recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/chenzehao82/Weak-Driven-Learning.git
cd Weak-Driven-Learning
```

2. **Set up the environment**

```bash
# Create conda environment (example)
conda create -n weak_driven python=3.10
conda activate weak_driven

# Install dependencies
pip install torch transformers accelerate deepspeed
pip install vllm  # For evaluation
# Add other dependencies as needed
```

3. **Configure training parameters**

Edit `scripts/run_ensemble.sh` and modify:
- `GPU_USE`: GPU device IDs (e.g., `0,1,2,3,4,5,6,7`)
- `base_model`: Base model path (e.g., `Qwen/Qwen3-4B-Base`)
- `stage1_data_path`: Path to Stage 1 training data (JSONL format)
- `data_files`: Path to training data for subsequent stages
- `outdir`: Output directory for checkpoints
- Training hyperparameters (epochs, batch size, gradient accumulation, max sequence length, etc.)

4. **Run the complete pipeline**

```bash
# Important: Run from project root directory
cd "/path/to/Weak-Driven-Learning"
bash scripts/run_ensemble.sh
```

The script will automatically:
- Compute base model entropy
- Train Stage 1 model
- Compute Stage 1 entropy
- Merge entropy files
- Train ensemble model (Stage 3)
- Extract the enhanced sub-model
- Evaluate the extracted model

## Training Pipeline

The complete pipeline consists of the following steps:

### Step 0: Compute Base Model Entropy
Computes `entropy_0` for the base model on the training dataset.

### Step 1: Stage 1 Training
Trains the first sub-model `m1` using the base model and Stage 1 training data. Output: `$outdir/stage1_m1`

### Step 2: Compute Stage 1 Entropy
Computes `entropy_1` for the Stage 1 model.

### Step 3: Merge Entropy Files
Combines `entropy_0` and `entropy_1` → `entropy_merged_stage1.jsonl`

### Step 4: Prepare Base Model for Fusion
Copies the base model to `$outdir/stage0_m0` for ensemble fusion.

### Step 5: Stage 3 Training (Ensemble)
Fuses `m0 + m1` and continues training with entropy-weighted sampling data. This produces the final ensemble model.

### Step 6: Extract Sub-model
Extracts the first sub-model (`submodel_idx=0`) from the ensemble model, which contains the enhanced capabilities.

### Step 7: Evaluation
Evaluates the extracted model using `eval_vllm_thinking_math.py` on reasoning tasks.

## Project Structure

```
Weak-Driven-Learning/
├── scripts/              # One-command pipeline scripts (entry point)
│   └── run_ensemble.sh  # Complete training pipeline
├── ensemble/             # Core training, entropy computation, extraction, and evaluation
│   ├── ensemble_train.py
│   ├── run_entropy.py
│   ├── extract_submodel.py
│   ├── copymodel.py
│   └── eval_vllm_thinking_math.py
├── utils/                # Model loading, fusion, entropy computation, data processing
│   ├── utils.py
│   ├── fuse_models.py
│   ├── compute_entropy.py
│   ├── weight_datasets.py
│   └── load_dataset.py
├── Trainer/              # SFT training runners and trainers
│   ├── sft_runner.py
│   ├── sft_trainer.py
│   └── ensemble_sft_trainer.py
├── EnsembleQwen3/        # Qwen3 ensemble model definitions
│   ├── configuration_qwen3.py
│   └── modeling_qwen3.py
├── docs/                 # Additional documentation
├── pics/                 # Figures and diagrams
│   ├── weak-drivenlearning.png  # Framework overview
│   ├── pivotFig.pdf            # Model architecture
│   └── 结果图.png              # Results
├── weights/              # Model checkpoints (generated, gitignored)
├── logs/                 # Training logs (generated, gitignored)
├── tensorboard_logs/     # TensorBoard logs (generated, gitignored)
├── LICENSE
├── README.md
├── QUICKSTART.md
└── SETUP.md
```

## Modular System Design

Weak-Driven Learning is implemented as a modular system with clear separation of concerns:

### Core Modules

1. **Ensemble Training Module** (`ensemble/ensemble_train.py`)
   - Implements the multi-stage training pipeline
   - Manages model fusion and weighted sampling
   - Coordinates training stages

2. **Entropy Computation Module** (`utils/compute_entropy.py`, `ensemble/run_entropy.py`)
   - Computes entropy for models at different stages
   - Merges entropy files for weighted sampling
   - Supports BrownBoost-style weighting

3. **Model Fusion Module** (`utils/fuse_models.py`)
   - Implements WMSS fusion method
   - Handles ensemble model creation and sub-model extraction
   - Manages model checkpointing

4. **Training Runner** (`Trainer/sft_runner.py`)
   - Handles distributed training with DeepSpeed
   - Manages training loops and optimization
   - Supports gradient accumulation and mixed precision

5. **Evaluation Module** (`ensemble/eval_vllm_thinking_math.py`)
   - Evaluates models on reasoning tasks
   - Uses vLLM for efficient inference
   - Outputs results to `results/` directory

## Configuration

### Training Configuration

Key parameters in `scripts/run_ensemble.sh`:

```bash
# GPU Configuration
GPU_USE=0,1,2,3,4,5,6,7

# Model and Data Paths
base_model="Qwen/Qwen3-4B-Base"
stage1_data_path="/path/to/stage1_data.jsonl"
data_files="/path/to/data.jsonl"
outdir="weights/ensemble/Qwen3-4B-Base"

# Training Hyperparameters
stage1_epochs=1
stage3_epochs=1
per_device_train_batch_size=4
gradient_accumulation_steps=4
max_seq_length=2048

# BrownBoost Parameters
alpha=0.1
beta=0.8
```

### Data Format

**Stage 1 Data** (JSONL format):
```json
{"instruction": "...", "output": "..."}
```

**Stage 2/3 Data** (JSONL format):
```json
{"idx": 0, "instruction": "...", "output": "..."}
```

**Entropy Files** (JSONL format):
```json
{"idx": 0, "entropy_0": 2.5, "entropy_1": 2.1, ...}
```

## Common Issues and Troubleshooting

### Error: `No module named 'utils'`

**Cause**: Not running from project root directory, or Python path doesn't include project root.

**Solution**:
- Always `cd` to the project root directory before running scripts:
  ```bash
  cd "/path/to/Weak-Driven-Learning"
  ```
- The `scripts/run_ensemble.sh` uses relative paths. Don't run scripts from other directories.

### Error: `python: command not found`

**Cause**: System doesn't have `python` symlink.

**Solution**: Use `python3` or activate your conda environment where `python` is available.

### Error: GPU Out of Memory

**Solution**: Adjust the following parameters:
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Use fewer GPUs (modify `GPU_USE`)

### Error: Cannot locate Qwen3ForEnsemble

**Cause**: `EnsembleQwen3/` directory missing or Python path incorrect.

**Solution**: Ensure `EnsembleQwen3/` exists and is in the Python path.

## Evaluation Results

Evaluation results are saved to the `results/` directory (if configured in the evaluation script). Training logs are written to `logs/`.

Example results visualization:

<p align="center">
  <img src="pics/结果图.png" alt="Evaluation Results" width="600"/>
</p>

## Advanced Usage

### Running Individual Stages

You can run individual stages by modifying `scripts/run_ensemble.sh` or calling the Python scripts directly:

```bash
# Compute entropy only
python ensemble/run_entropy.py --model-path <path> --data-path <path>

# Train Stage 1 only
python ensemble/ensemble_train.py --stage 1 --model-name <model> --stage1-data-path <path>

# Extract sub-model
python ensemble/extract_submodel.py --ensemble-path <path> --submodel-idx 0
```

### Custom Model Types

To add support for new model types:

1. Add model definition to `EnsembleQwen3/` or create a new model directory
2. Add fusion logic to `utils/fuse_models.py`
3. Update `ensemble/ensemble_train.py` to support the new type

## Citation

If you find Weak-Driven Learning useful, please consider citing:

```bibtex
@misc{weak-driven-learning,
  title={Weak-Driven Learning: Multi-Stage Training with Entropy-Based Weighted Sampling},
  author={Your Name},
  year={2025},
  url={https://github.com/chenzehao82/Weak-Driven-Learning}
}
```

## Acknowledgments

- Project organization and documentation style inspired by [RAGEN](https://github.com/mll-lab-nu/RAGEN)
- Model architecture based on Qwen models
- Training framework built on Hugging Face Transformers and DeepSpeed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## References

- [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://github.com/mll-lab-nu/RAGEN)
- [Qwen Models](https://github.com/QwenLM/Qwen)

---

For detailed setup instructions, see [SETUP.md](SETUP.md).  
For a quick start guide, see [QUICKSTART.md](QUICKSTART.md).
