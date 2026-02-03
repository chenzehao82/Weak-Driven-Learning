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

from utils.utils import load_model_tokenizer  # 你原来的函数
# ================== 分布式调试信息（保留） ==================
if os.environ.get("RANK", None) is not None:
    rank = os.environ["RANK"]
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[rank {rank}] LOCAL_RANK={local_rank}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

# ================== checkpoint 工具函数 ==================
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

# ================== DeepSpeed 状态检查（保留） ==================
def check_deepspeed_status(trainer):
    """改进的DeepSpeed状态检查"""
    try:
        # 方法1: 检查trainer是否使用了deepspeed
        if hasattr(trainer, 'is_deepspeed_enabled') and trainer.is_deepspeed_enabled:
            print("✓ DeepSpeed已启用")
            return True
            
        # 方法2: 检查模型是否被DeepSpeed包装
        model = trainer.model
        if hasattr(model, 'module') and hasattr(model.module, 'engine'):
            engine = model.module.engine
            print("=== DeepSpeed状态 ===")
            print(f"ZeRO阶段: {engine.zero_optimization_stage()}")
            print(f"优化器: {type(engine.optimizer).__name__}")
            return True
            
        # 方法3: 直接检查accelerate状态
        from accelerate.utils import is_deepspeed_available
        if is_deepspeed_available():
            try:
                from deepspeed import comm as dist
                if dist.is_initialized():
                    print("✓ DeepSpeed分布式已初始化")
                    return True
            except:
                pass
        
        print("✗ DeepSpeed未正确初始化")
        return False
    except Exception as e:
        print(f"✗ 检查DeepSpeed状态失败: {e}")
        return False

# ================== 通用 SFT 函数（你需要的核心封装） ==================
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
    resume_from_checkpoint: bool = True,  # 是否从 checkpoint 恢复
):
    """
    对给定 model / tokenizer / dataset 进行 SFT 微调。
    
    参数:
        model, tokenizer: 已加载好的模型与分词器
        train_dataset: HF Dataset 或 IterableDataset
        output_dir: checkpoint 输出目录
        其他参数与你原来的 argparse 对应
    """
    distributed_state = PartialState()

    # ------ 处理数据集 & 计算 max_steps ------
    if isinstance(train_dataset, Dataset):
        # Map-style Dataset：走你原来的逻辑
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
        # 已经是 IterableDataset 的情况：不强行改，只给个 None
        dataset = train_dataset
        max_step = None

    # ------ 主进程打印信息 & 建目录 ------
    if distributed_state.is_main_process:
        print("DeepSpeed config:", os.environ.get("DEEPSPEED_CONFIG_FILE", "Not set"))
        os.makedirs(output_dir, exist_ok=True)

        print("\n=========== SFT 配置参数 ===========")
        print(f"output_dir: {output_dir}")
        print(f"max_seq_length: {max_seq_length}")
        print(f"per_device_train_batch_size: {per_device_train_batch_size}")
        print(f"grad_accum: {grad_accum}")
        print(f"num_epochs: {num_epochs}")
        print(f"learning_rate: {learning_rate}")
        print(f"bf16: {bf16}")
        print(f"wandb_run_name: {wandb_run_name}")
        print("====================================\n")

        print(f"数据集样本数(估算): {len(train_dataset) if isinstance(train_dataset, Dataset) else 'IterableDataset'}")

    # 同步
    distributed_state.wait_for_everyone()

    # ------ checkpoint 恢复 ------
    if resume_from_checkpoint:
        checkpoint = get_last_checkpoint(output_dir)
        if checkpoint and distributed_state.is_main_process:
            print(f"📂 找到 checkpoint: {checkpoint}，将尝试恢复训练")
    else:
        checkpoint = None
        if distributed_state.is_main_process:
            print("⚠️ resume_from_checkpoint=False，从头开始训练")

    # ------ 构造 SFTConfig ------
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
        # save_strategy="steps",        # 将保存策略从 "epoch" 改为 "steps"
        # save_steps=100,               # 设置每 100 步保存一次
        output_dir=output_dir,
        report_to="tensorboard",
        logging_dir=logging_dir,
        run_name=wandb_run_name,
        gradient_checkpointing=True,
        max_steps=max_step,  # 沿用你原来的 max_step 逻辑
        accelerator_config={"dispatch_batches": False},
        seed=seed,
        save_only_model=True
    )

    # ------ 创建 Trainer ------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # 使用 tokenizer
        args=sft_config,
        train_dataset=dataset,
    )

    # ------ 检查 DeepSpeed 状态 ------
    if check_deepspeed_status(trainer):
        print("✓ DeepSpeed 正常工作")
    else:
        print("⚠ 使用非 DeepSpeed 模式训练")

    # ------ 开始训练 ------
    if distributed_state.is_main_process:
        print("\n=========== 开始训练 ===========\n")

    trainer.train(resume_from_checkpoint=checkpoint)

    return trainer  # 方便后面拿 log / model 等


# ================== 原来 main 的一个简单封装示例（可选） ==================
def parse_known_args():
    """只解析已知参数，忽略未知参数"""
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
        print(f"忽略未知参数: {unknown}")
    return args

def load_jsonl_files(data_files):
    """简单的 JSONL 加载函数（从你原来的 main 中抽出来）"""
    all_records = []
    print("正在从本地 JSONL 文件加载数据...")
    for file_path in data_files:
        print(f"加载文件：{file_path}")
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except Exception as e:
                    print(f"[警告] 跳过坏行({file_path}): {e}")
    print(f"数据加载完成，共读取样本数：{len(all_records)}")
    return Dataset.from_list(all_records)

def main():
    args = parse_known_args()

    # 1) 加载数据
    dataset = load_jsonl_files(args.data_files)

    # 2) 加载模型和 tokenizer（你的工具函数）
    model, tokenizer = load_model_tokenizer(args.model_name)

    # 3) 调用统一的 run_sft 进行微调
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
