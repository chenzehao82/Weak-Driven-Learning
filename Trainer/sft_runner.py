import argparse
import os
import torch
import json
import datetime
import re

from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import AutoTokenizer
from accelerate import PartialState

from utils.utils import load_model_tokenizer  ## your original function
## ================== Distributed debug information (reserved) ==================
if os.environ.get("RANK", None) is not None:
    rank = os.environ["RANK"]
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[rank {rank}] LOCAL_RANK={local_rank}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

## ================== checkpoint utility functions ==================
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint(folder):
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

## ================== DeepSpeed status check (reserved) ==================
def check_deepspeed_status(trainer):
    """Improved DeepSpeed status check"""
    try:
        ## Method 1: Check if trainer uses deepspeed
        if hasattr(trainer, 'is_deepspeed_enabled') and trainer.is_deepspeed_enabled:
            print("✓ DeepSpeed enabled")
            return True
            
        ## Method 2: Check if model is wrapped by DeepSpeed
        model = trainer.model
        if hasattr(model, 'module') and hasattr(model.module, 'engine'):
            engine = model.module.engine
            print("=== DeepSpeed status ===")
            print(f"ZeRO stage: {engine.zero_optimization_stage()}")
            print(f"Optimizer: {type(engine.optimizer).__name__}")
            return True
            
        ## Method 3: Directly check accelerate status
        from accelerate.utils import is_deepspeed_available
        if is_deepspeed_available():
            try:
                from deepspeed import comm as dist
                if dist.is_initialized():
                    print("✓ DeepSpeed distributed initialized")
                    return True
            except:
                pass
        
        print("✗ DeepSpeed not properly initialized")
        return False
    except Exception as e:
        print(f"✗ Failed to check DeepSpeed status: {e}")
        return False

## ================== General SFT function (core wrapper you need) ==================
def run_sft(
    model,
    tokenizer,
    train_dataset,
    *,
    output_dir: str,
    max_seq_length: int = 4096,
    per_device_train_batch_size: int = 1,
    grad_accum: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    bf16: bool = True,
    wandb_run_name: str | None = None,
    logging_steps: int = 1,
    seed: int = 42,
    resume_from_checkpoint: bool = True,  ## Whether to resume from checkpoint
):
    """
    Perform SFT fine-tuning on given model / tokenizer / dataset.
    
    Parameters:
        model, tokenizer: Loaded model and tokenizer
        train_dataset: HF Dataset or IterableDataset
        output_dir: checkpoint output directory
        Other parameters correspond to your original argparse
    """
    distributed_state = PartialState()

    ## ------ Handle dataset # ------ Handle dataset ------  &  max_steps ------ calculate max_steps ------ calculate max_steps ------
    if isinstance(train_dataset, Dataset):
        ## Map-style Dataset: use your original logic
        dataset = train_dataset.shuffle(seed=seed)
        max_step = (
            len(dataset)
            * num_epochs
            // (
                distributed_state.num_processes
                * per_device_train_batch_size
                * grad_accum
            )
        )
        dataset = dataset.shuffle(seed=seed).to_iterable_dataset(
            num_shards=distributed_state.num_processes * 2
        )
    else:
        ## Already IterableDataset case: don't force change, just give None
        dataset = train_dataset
        max_step = None

    ## ------ Main process print info # ------ Main process print info ------  &  ------ create directory ------ create directory ------
    if distributed_state.is_main_process:
        print("DeepSpeed config:", os.environ.get("DEEPSPEED_CONFIG_FILE", "Not set"))
        os.makedirs(output_dir, exist_ok=True)

        print("\n=========== SFT Configuration ===========")
        print(f"output_dir: {output_dir}")
        print(f"max_seq_length: {max_seq_length}")
        print(f"per_device_train_batch_size: {per_device_train_batch_size}")
        print(f"grad_accum: {grad_accum}")
        print(f"num_epochs: {num_epochs}")
        print(f"learning_rate: {learning_rate}")
        print(f"bf16: {bf16}")
        print(f"wandb_run_name: {wandb_run_name}")
        print("====================================\n")

        print(f"Dataset sample count (estimate): {len(train_dataset) if isinstance(train_dataset, Dataset) else 'IterableDataset'}")

    ## Sync
    distributed_state.wait_for_everyone()

    ## ------ Checkpoint resume ------
    if resume_from_checkpoint:
        checkpoint = get_last_checkpoint(output_dir)
        if checkpoint and distributed_state.is_main_process:
            print(f"📂 Found checkpoint: {checkpoint}, will attempt to resume training")
    else:
        checkpoint = None
        if distributed_state.is_main_process:
            print("⚠️ resume_from_checkpoint=False, start training from scratch")

    ## ------ Construct SFTConfig ------
    logging_dir = os.path.join(
        "tensorboard_logs",
        (wandb_run_name or "run") + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    sft_config = SFTConfig(
        max_length=max_seq_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=bf16,
        logging_steps=logging_steps,
        save_strategy="epoch",
        ## save_strategy="steps",        # Change save strategy from "epoch" to "steps"
        ## save_steps=100,               # Set save every 100 steps
        output_dir=output_dir,
        report_to="tensorboard",
        logging_dir=logging_dir,
        run_name=wandb_run_name,
        gradient_checkpointing=True,
        max_steps=max_step,  ## use your original max_step logic
        accelerator_config={"dispatch_batches": False},
        seed=seed,
        save_only_model=True
    )

    ## ------ Create Trainer ------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  ## Use tokenizer
        args=sft_config,
        train_dataset=dataset,
    )

    ## ------ Check DeepSpeed status ------
    if check_deepspeed_status(trainer):
        print("✓ DeepSpeed working normally")
    else:
        print("⚠ Training in non-DeepSpeed mode")

    ## ------ Start training ------
    if distributed_state.is_main_process:
        print("\n=========== StartingTraining ===========\n")

    trainer.train(resume_from_checkpoint=checkpoint)

    return trainer  ## Convenient for getting log / model etc. later


## ================== A simple wrapper example of original main (optional) ==================
def parse_known_args():
    """Only parse known parameters, ignore unknown parameters"""
    parser = argparse.ArgumentParser(description="Finetune Qwen3 model")
    
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--data-files", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Parameters: {unknown}")
    return args

def load_jsonl_files(data_files):
    """Simple JSONL loading function (extracted from your original main)"""
    all_records = []
    print("Loading data from local JSONL files...")
    for file_path in data_files:
        print(f"Loading file:{file_path}")
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except Exception as e:
                    print(f"[Warning] Skipping bad line({file_path}): {e}")
    print(f"Data loading complete, total samples read:{len(all_records)}")
    return Dataset.from_list(all_records)

def main():
    args = parse_known_args()

    ## 1) Load data
    dataset = load_jsonl_files(args.data_files)

    ## 2) Load model and tokenizer (your utility functions)
    model, tokenizer = load_model_tokenizer(args.model_name)

    ## 3) Call unified run_sft for fine-tuning
    run_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        grad_accum=args.grad_accum,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        wandb_run_name=args.wandb_run_name,
    )

if __name__ == "__main__":
    main()
