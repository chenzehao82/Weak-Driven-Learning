# Project Structure

This document describes the organization of the Ensemble LLM project, following best practices for modular ML frameworks.

## Directory Overview

```
EnsembleLLM_standalone_refactored/
├── config/                 # Configuration files (YAML)
│   ├── base.yaml          # Base configuration with defaults
│   └── train.yaml         # Training-specific settings
│
├── ensemble/               # Core training and evaluation modules
│   ├── __init__.py        # Package initialization
│   ├── ensemble_train.py  # Main training script (3-stage pipeline)
│   ├── run_entropy.py     # Entropy computation utilities
│   ├── extract_submodel.py # Extract submodels from ensemble
│   ├── copymodel.py       # Model copying utility
│   └── eval_vllm_thinking_math.py # Evaluation script
│
├── scripts/                # Shell scripts for automation
│   └── run_ensemble.sh    # Complete training pipeline script
│
├── utils/                  # Utility functions and helpers
│   ├── __init__.py
│   ├── utils.py           # Model loading and data utilities
│   ├── fuse_models.py     # Model fusion functions (WMSS)
│   ├── weight_datasets.py # Dataset weighting (BrownBoost)
│   ├── compute_entropy.py  # Entropy computation
│   ├── load_dataset.py    # Dataset loading
│   └── prompts.py         # Prompt templates
│
├── Trainer/                # Training runners
│   ├── sft_runner.py      # SFT training runner
│   ├── sft_trainer.py     # Base SFT trainer
│   └── ensemble_sft_trainer.py # Ensemble-specific trainer
│
├── EnsembleQwen3/          # Qwen3 ensemble model definitions
│   ├── __init__.py
│   ├── configuration_qwen3.py
│   └── modeling_qwen3.py
│
├── examples/               # Example configurations and scripts
│   └── example_config.yaml # Example configuration file
│
├── docs/                   # Documentation
│   └── STRUCTURE.md       # This file
│
├── tests/                  # Unit tests (to be added)
│
├── weights/                # Model checkpoints (generated, gitignored)
├── logs/                   # Training logs (generated, gitignored)
├── tensorboard_logs/       # TensorBoard logs (generated, gitignored)
│
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation script
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
├── README.md             # Main documentation
├── QUICKSTART.md         # Quick start guide
└── SETUP.md              # Setup instructions
```

## Module Descriptions

### `config/`
Configuration files in YAML format for easy modification and version control.

- **base.yaml**: Contains all default settings for models, data, training, and evaluation
- **train.yaml**: Training-specific overrides and stage configurations

### `ensemble/`
Core training and evaluation modules. These are the main entry points for training.

- **ensemble_train.py**: Implements the 3-stage training pipeline
- **run_entropy.py**: Computes entropy for models at different stages
- **extract_submodel.py**: Extracts individual submodels from ensemble
- **eval_vllm_thinking_math.py**: Evaluates models on math reasoning datasets

### `scripts/`
Shell scripts for automation and pipeline execution.

- **run_ensemble.sh**: Orchestrates the complete training pipeline from entropy computation to evaluation

### `utils/`
Reusable utility functions used across the project.

- **fuse_models.py**: WMSS model fusion implementation
- **weight_datasets.py**: BrownBoost-style dataset weighting
- **compute_entropy.py**: Entropy computation algorithms

### `Trainer/`
Training runners that handle the actual training loops.

- **sft_runner.py**: Supervised fine-tuning runner with distributed training support

## Design Principles

1. **Modularity**: Each component has a clear, single responsibility
2. **Configuration-driven**: Training parameters are externalized to YAML files
3. **Separation of concerns**: Training, evaluation, and utilities are clearly separated
4. **Extensibility**: Easy to add new models, datasets, or training stages

## Adding New Components

### Adding a New Model Type

1. Add model definition to `EnsembleQwen3/` or create new model directory
2. Add fusion logic to `utils/fuse_models.py`
3. Update `ensemble/ensemble_train.py` to support the new type
4. Add configuration options to `config/base.yaml`

### Adding a New Training Stage

1. Add stage function to `ensemble/ensemble_train.py`
2. Update `scripts/run_ensemble.sh` to include the new stage
3. Add stage configuration to `config/train.yaml`

### Adding a New Dataset

1. Add dataset loader to `utils/load_dataset.py`
2. Add evaluation script if needed
3. Update `config/base.yaml` with dataset paths

