##!/usr/bin/env python3
## -*- coding: utf-8 -*-
"""
Ensemble LLM training script
Support three-stage training process: Stage1 (m1) -> Stage2 (m2) -> Stage3 (fuse m1+m2)
"""

import argparse
import os
import sys
import torch
import json
import datetime
import re
import numpy as np
import pandas as pd
from pathlib import Path

from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import AutoTokenizer
from accelerate import PartialState

## Add parent directory to path to import utils
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

## Import utility functions
try:
    from utils.utils import load_model_tokenizer, load_data, load_entropy_df
    from utils.fuse_models import (
        fuse_submodels,
        load_fuse_model_tokenizer_vote
    )
    from utils.weight_datasets import (
        compute_sampling_weights_brownboost_style, compute_adaboost_sampling_weights
    )
    from utils.load_dataset import load_math_dataset_jsonl
except ImportError as e:
    print(f"Error: Cannot import utility functions: {e}")
    print("Please ensure:")
    print("  1. utils/ Directory exists in project root")
    print("  2. Python path is correctly set")
    print("  3. All necessary module files exist")
    raise

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    """Get latest checkpoint"""
    if not os.path.isdir(folder):
        return None
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def build_weighted_dataset(
    dataset: Dataset,
    entropy_df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    sample_multiplier: float,
    easy_quantile: float,
    hard_quantile: float,
    patience: int,
    easy_patience: int,
    lambda_time: float,
    lambda_easy: float,
    seed: int = 42,
    dataset_type: str = "brownboost",
) -> Dataset:
    """
    Build weighted dataset based on entropy information
    """
    if dataset_type == "adaboost":
        entropy_df = compute_adaboost_sampling_weights(
            entropy_df, alpha=alpha, beta=beta, gamma=gamma
        )
    elif dataset_type == "brownboost":
        entropy_df = compute_sampling_weights_brownboost_style(
            entropy_df,
            alpha=alpha, beta=beta, gamma=gamma,
            easy_quantile=easy_quantile, hard_quantile=hard_quantile,
            patience=patience, easy_patience=easy_patience,
            lambda_time=lambda_time, lambda_easy=lambda_easy,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    weight_map = dict(
        zip(entropy_df["idx"].tolist(), entropy_df["sampling_weight"].tolist())
    )

    weights = []
    miss_cnt = 0
    for ex in dataset:
        idx = ex.get("idx", None)
        if idx is None:
            raise ValueError("dataset sample missing 'idx' field")
        w = weight_map.get(idx, 0.0)
        if idx not in weight_map:
            miss_cnt += 1
        weights.append(w)

    weights = np.asarray(weights, dtype=np.float64)
    if miss_cnt > 0:
        print(f"[WARN] {miss_cnt} samples in dataset have no matching entropy idx, weight=0.")

    w_sum = weights.sum()
    if w_sum <= 0:
        print("[WARN] all weights are zero, falling back to uniform sampling.")
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / w_sum

    n_orig = len(dataset)
    n_new = int(n_orig * sample_multiplier)
    if n_new <= 0:
        raise ValueError(f"sample_multiplier too small, n_new={n_new}")

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(n_orig, size=n_new, replace=True, p=weights)

    counts = np.bincount(sampled_indices, minlength=n_orig)
    num_unique = (counts > 0).sum()
    num_duplicated_samples = (counts > 1).sum()
    total_duplicate_occurrences = n_new - num_unique

    print(f"[Resample stats]")
    print(f"  original dataset size       : {n_orig}")
    print(f"  new dataset size (resampled): {n_new}")
    print(f"  unique samples used         : {num_unique}")
    print(f"  samples appearing >1 times  : {num_duplicated_samples}")
    print(f"  total duplicate occurrences : {total_duplicate_occurrences}")

    new_dataset = dataset.select(sampled_indices.tolist())
    return new_dataset


def train_stage1(args, distributed_state: PartialState):
    """Stage 1: Regular SFT training to get m1"""
    stage1_out = os.path.join(args.output_dir, "stage1_m1")
    if distributed_state.is_main_process:
        os.makedirs(stage1_out, exist_ok=True)
        print("\n=========== Stage1 Configuration ParametersParameters ===========")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("======================================\n")

    model, tokenizer = load_model_tokenizer(args.model_name)

    use_chat_template = None
    if args.use_chat_template.lower() == "auto":
        use_chat_template = True
    elif args.use_chat_template.lower() == "true":
        use_chat_template = True
    elif args.use_chat_template.lower() == "false":
        use_chat_template = False

    dataset = load_math_dataset_jsonl(
        args.stage1_data_path,
        tokenizer,
        use_chat_template=use_chat_template,
    )

    if distributed_state.is_main_process:
        print(f"Stage1 data loading complete, total samples read:{len(dataset)}")

    dataset = dataset.shuffle(seed=42)
    max_step = (
        len(dataset)
        * args.stage1_num_epochs
        // (
            distributed_state.num_processes
            * args.per_device_train_batch_size
            * args.grad_accum
        )
    )

    iterable_dataset = dataset.shuffle(seed=0).to_iterable_dataset(
        num_shards=distributed_state.num_processes * 2
    )

    distributed_state.wait_for_everyone()
    checkpoint = get_last_checkpoint(stage1_out)

    config = SFTConfig(
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.stage1_num_epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="epoch",
        output_dir=stage1_out,
        report_to="tensorboard",
        logging_dir=(
            f'tensorboard_logs/{args.wandb_run_name or "stage1"}_'
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        run_name=(args.wandb_run_name or "stage1"),
        gradient_checkpointing=True,
        max_steps=max_step,
        accelerator_config={"dispatch_batches": False},
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=iterable_dataset,
    )

    if distributed_state.is_main_process:
        print("\n=========== Starting Stage1 training (m1) ===========\n")

    trainer.train(resume_from_checkpoint=checkpoint)
    distributed_state.wait_for_everyone()
    return stage1_out


def train_stage2(args, distributed_state: PartialState, m1_dir: str):
    """Stage 2: Use entropy-weighted dataset, continue training from m1 to get m2"""
    from Trainer.sft_runner import run_sft

    if distributed_state.is_main_process:
        print("\n=========== Stage2:  weighted dataset &  m1  m2 ===========")

    dataset = load_data(args.data_files)
    entropy_df = load_entropy_df(args.entropy_results)

    weighted_dataset = build_weighted_dataset(
        dataset=dataset,
        entropy_df=entropy_df,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        sample_multiplier=args.sample_multiplier_stage2,
        easy_quantile=args.easy_quantile,
        hard_quantile=args.hard_quantile,
        patience=args.patience,
        easy_patience=args.easy_patience,
        lambda_time=args.lambda_time,
        lambda_easy=args.lambda_easy,
        dataset_type="adaboost",
    )

    model, tokenizer = load_model_tokenizer(m1_dir)

    stage2_out = os.path.join(args.output_dir, "stage2_m2")
    if distributed_state.is_main_process:
        os.makedirs(stage2_out, exist_ok=True)
    distributed_state.wait_for_everyone()

    run_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=weighted_dataset,
        output_dir=stage2_out,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        grad_accum=args.grad_accum,
        num_epochs=args.stage2_num_epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        wandb_run_name=(
            f"{args.wandb_run_name}_stage2" if args.wandb_run_name else "stage2"
        ),
    )

    if distributed_state.is_main_process:
        print(f"Stage2 training complete, m2 saved in: {stage2_out}")
    distributed_state.wait_for_everyone()
    return stage2_out


def train_stage3(args, distributed_state: PartialState, m1_dir: str, m2_dir: str, dataset_type: str = "adaboost"):
    """Stage 3: Fuse m1 and m2, then train with weighted dataset"""
    from Trainer.sft_runner import run_sft

    ##  m1  m2
    ensemble_dir = os.path.join(args.output_dir, "ensemble_m1_m2")
    if distributed_state.is_main_process:
        print("\n=========== Stage3: Fuse m1 & m2 ==========")
        print(f"Ensemble model:")
        print(f"  - m1: {m1_dir}")
        print(f"  - m2: {m2_dir}")
        print(f"  - Save to: {ensemble_dir}")
        
        if args.model_type == "wmss":
            print("Model Type: WMSS (Weighted Model Selection and Synthesis)")
            fuse_submodels(
                model_list=[m1_dir, m2_dir],
                save_dir=ensemble_dir,
                fusion_lambda=args.fusion_lambda if hasattr(args, 'fusion_lambda') and args.fusion_lambda is not None else 0.5,
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        print(f"✓ Fusion complete, ensemble initial weights saved in: {ensemble_dir}")
    distributed_state.wait_for_everyone()

    dataset = load_data(args.data_files)
    entropy_df = load_entropy_df(args.entropy_results)

    weighted_dataset = build_weighted_dataset(
        dataset=dataset,
        entropy_df=entropy_df,
        dataset_type=dataset_type,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        sample_multiplier=args.sample_multiplier_stage3,
        easy_quantile=args.easy_quantile,
        hard_quantile=args.hard_quantile,
        patience=args.patience,
        easy_patience=args.easy_patience,
        lambda_time=args.lambda_time,
        lambda_easy=args.lambda_easy,
    )

    print("args.freeze_first_model", args.freeze_first_model)
    if args.model_type == "wmss":
        model, tokenizer = load_fuse_model_tokenizer_vote(ensemble_dir, args.freeze_first_model)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    stage3_out = os.path.join(args.output_dir, args.stage3_name)
    if distributed_state.is_main_process:
        os.makedirs(stage3_out, exist_ok=True)
    distributed_state.wait_for_everyone()

    run_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=weighted_dataset,
        output_dir=stage3_out,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        grad_accum=args.grad_accum,
        num_epochs=args.stage3_num_epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        wandb_run_name=(
            f"{args.wandb_run_name}_stage3" if args.wandb_run_name else "stage3"
        ),
        resume_from_checkpoint=False,
    )

    if distributed_state.is_main_process:
        print(f"Stage3 training complete, final model saved in: {stage3_out}")
    distributed_state.wait_for_everyone()
    return stage3_out


def _str2bool(v):
    """Parse boolean string"""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble LLM three-stage training")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    ## Base model # Base model  &  common training hyperparameters common training hyperparameters
    parser.add_argument("--model-name", type=str, required=True, help="base HF model name or local path")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--use-chat-template", type=str, default="auto", help="auto/true/false")

    ## Stage1: Regular SFT
    parser.add_argument("--stage1-data-path", type=str, required=True, help="jsonl for math dataset used in stage1")
    parser.add_argument("--stage1-num-epochs", type=int, default=1)

    ## Stage2/3: Entropy sampling data
    parser.add_argument("--data-files", type=str, required=True, help="training data with `idx` for entropy sampling")
    parser.add_argument("--entropy-results", type=str, default=None, help="entropy file path")

    parser.add_argument("--stage2-num-epochs", type=int, default=1)
    parser.add_argument("--stage3-num-epochs", type=int, default=1)

    parser.add_argument("--sample-multiplier-stage2", type=float, default=1.0)
    parser.add_argument("--sample-multiplier-stage3", type=float, default=1.0)

    ## BrownBoost hyperparameters
    parser.add_argument("--alpha", type=float, default=0.2, help="weight for H0 term")
    parser.add_argument("--beta", type=float, default=0.7, help="weight for -ΔH term (improvement)")
    parser.add_argument("--gamma", type=float, default=0.1, help="weight for +ΔH term (rebound)")
    parser.add_argument("--easy-quantile", type=float, default=0.2)
    parser.add_argument("--hard-quantile", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--easy-patience", type=int, default=2)
    parser.add_argument("--lambda-time", type=float, default=1.0)
    parser.add_argument("--lambda-easy", type=float, default=1.0)

    ## Stage selection
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True, help="Specify which stage to run")
    parser.add_argument("--m1-path", type=str, default=None, help="m1 model path needed for Stage 2/3")
    parser.add_argument("--m2-path", type=str, default=None, help="m2 model path needed for Stage 3")

    parser.add_argument("--model-type", type=str, default="wmss",
                        choices=["wmss"],
                        help="Specify which model type to use (wmss: Weighted Model Selection and Synthesis)")
    parser.add_argument("--stage3-name", type=str, default="stage3_fused_brownboost", help="Stage 3 final model save directory name")
    parser.add_argument("--freeze-first-model", type=_str2bool, default=False, help="Whether to freeze the first model")
    parser.add_argument("--fusion-lambda", type=float, default=0.5, help="Fusion weight")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown parameters: {unknown}")
    return args


def main():
    args = parse_args()
    distributed_state = PartialState()

    ## Stage 1: SFT (m1)
    if args.stage == 1:
        if distributed_state.is_main_process:
            print("\n" + "="*60)
            print("Stage 1: Regular SFT training -> m1")
            print("="*60 + "\n")
        
        m1_dir = train_stage1(args, distributed_state)
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if distributed_state.is_main_process:
            print(f"\n✓ Stage 1 complete, model saved in: {m1_dir}")

    ## Stage 2: Weighted SFT (m2)
    elif args.stage == 2:
        if not args.m1_path:
            raise ValueError("Stage 2 needs to provide --m1-path parameter")
        if not args.entropy_results:
            raise ValueError("Stage 2 needs to provide --entropy-results parameter")

        if not os.path.isabs(args.m1_path):
            m1_dir = os.path.join(args.output_dir, args.m1_path)
        else:
            m1_dir = args.m1_path
        
        if os.path.isdir(m1_dir):
            m1_checkpoint = get_last_checkpoint(m1_dir)
            if m1_checkpoint:
                m1_dir = m1_checkpoint
        
        if distributed_state.is_main_process:
            print("\n" + "="*60)
            print("Stage 2: Entropy-weighted dataset training -> m2")
            print("="*60 + "\n")
            print(f"Load from m1 path: {m1_dir}")
            print(f"Use entropy file: {args.entropy_results}")
        
        m2_dir = train_stage2(args, distributed_state, m1_dir)
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if distributed_state.is_main_process:
            print(f"\n✓ Stage 2 complete, model saved in: {m2_dir}")

    ## Stage 3: Fuse & BrownBoost
    elif args.stage == 3:
        if not args.m1_path or not args.m2_path:
            raise ValueError("Stage 3 needs to provide --m1-path and --m2-path parameters")
        if not args.entropy_results:
            raise ValueError("Stage 3 needs to provide --entropy-results parameter")

        if not os.path.isabs(args.m1_path):
            m1_dir = os.path.join(args.output_dir, args.m1_path)
        else:
            m1_dir = args.m1_path
        
        if not os.path.isabs(args.m2_path):
            m2_dir = os.path.join(args.output_dir, args.m2_path)
        else:
            m2_dir = args.m2_path
        
        if os.path.isdir(m1_dir):
            m1_checkpoint = get_last_checkpoint(m1_dir)
            if m1_checkpoint:
                m1_dir = m1_checkpoint
        
        if os.path.isdir(m2_dir):
            m2_checkpoint = get_last_checkpoint(m2_dir)
            if m2_checkpoint:
                m2_dir = m2_checkpoint

        if distributed_state.is_main_process:
            print("\n" + "="*60)
            print("Stage 3: Fuse m1 Stage 3: Fuse m1 Stage 3:  m1 & m2，BrownBoost  m2, BrownBoost training m2, BrownBoost training")
            print("="*60 + "\n")
            print(f"m1 path: {m1_dir}")
            print(f"m2 path: {m2_dir}")
            print(f"Use entropy file: {args.entropy_results}")

        final_dir = train_stage3(args, distributed_state, m1_dir, m2_dir)
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if distributed_state.is_main_process:
            print(f"\n✓ Stage 3 complete, final model saved in: {final_dir}")


if __name__ == "__main__":
    main()

