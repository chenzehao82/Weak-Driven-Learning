from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Features, Value, concatenate_datasets, Dataset
import argparse
import os
from accelerate import PartialState
from datetime import timedelta
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    GenerationConfig,
    get_scheduler,
)
import json
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm
import datetime


def load_model_tokenizer(model_name):
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    origin_eos_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if origin_eos_token_id != eos_token_id:
        tokenizer.eos_token_id = eos_token_id
        # does not make any influence
        model.config.eos_token_id = eos_token_id
        # NOTE: append new eos token
        model.generation_config.eos_token_id = [origin_eos_token_id, eos_token_id]
    return model, tokenizer


def load_am_dataset(splition_name:str,tokenizer:Qwen2Tokenizer):
    system_prompt = """ You are a helpful assistant. To answer the user’s question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> and <answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    data_list = []
    # check dataset
    print('begin_to_load_dataset')
    with open(f"huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/{splition_name}.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                message = data["messages"]
                assert len(message) == 2 and message[0]["role"] == "user" and message[1]["role"] == "assistant"
                data_list.append({
                    'prompt': tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, message[0]],
                        tokenize=False,
                        add_generation_prompt=True) ,
                        "completion": message[1]["content"] + eos_token})
            except json.JSONDecodeError as e:
                print(f"跳过无效行: {e}")
                continue

    return Dataset.from_list(data_list)

def load_openthought_dataset(splition_name:str,tokenizer:Qwen2Tokenizer):
    from utils.prompts import SFT_SYSTEM_PROMPT
    system_prompt = SFT_SYSTEM_PROMPT
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    data_list = []
    # check dataset
    print('begin_to_load_dataset')
    with open(f"dataset/openthought_{splition_name}.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                message = data["conversations"]
                assert len(message) == 2 and message[0]["from"] == "human" and message[1]["from"] == "gpt"
                data_list.append({
                    'prompt': tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt}, 
                            {'role':'user',"content":message[0]['value']}
                         ],
                        tokenize=False,
                        add_generation_prompt=True) ,
                        "completion": message[1]["value"] + eos_token})
            except json.JSONDecodeError as e:
                print(f"跳过无效行: {e}")
                continue
    return Dataset.from_list(data_list)

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument(
        "--prefix", type=str, default="test", help="Experiment prefix or name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="the name of the model used",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate for the model optimizer",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="The warmup ratio of the scheduler",
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=0.1,
        help="The ratio of scheduler's final minimal learning rate",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="the batch size per device",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--max_length", type=int, default=32768, help="the max length of the tokenizer"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="the gradient accumulation steps",
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='am',
        choices=['am','openthought'],
        help='the name of the dataset'
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="1.4M",
        help="the dataset splition",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="the steps to save",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models/SFT",
        help="the ckpt saved directory"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="the max grad norm"
    )
    parser.add_argument(
        '--save_only_model',
        default=True,
        type=str2bool,
        help='whether to save only model'
    )
    parser.add_argument(
        '--save_strategy',
        default='epoch',
        choices=['epoch','steps'],
        type=str,
        help='the strategy for save checkpoint'
    )

    args = parser.parse_args()
    return args


def train(args):
    model, tokenizer = load_model_tokenizer(args.model_name)
    if args.dataset_name == 'am':
        if args.dataset_split == "1.4M":
                dataset = concatenate_datasets(
                    [
                        load_am_dataset(
                            "am_0.5M",
                            tokenizer=tokenizer,
                        ),
                        load_am_dataset(
                            "am_0.9M",
                            tokenizer=tokenizer,
                        ),
                    ]
                )
        elif args.dataset_split == "0.5M":
            dataset = load_am_dataset(
                "am_0.5M", tokenizer
            )
        elif args.dataset_split == "0.9M":
            dataset = load_am_dataset(
                "am_0.9M", tokenizer=tokenizer
            )
        elif args.dataset_split == "1K":
            dataset = load_am_dataset(
                "am_0.9M_sample_1k",
                tokenizer=tokenizer,
            )
        else:
            dataset = load_am_dataset(
                'am_' + args.dataset_split,
                tokenizer=tokenizer
            )
    elif args.dataset_name == 'openthought':
        dataset = load_openthought_dataset(args.dataset_split,tokenizer)
    else:
        raise NotImplementedError(f"dataset_name {args.dataset_name} is not supported")   


    max_step = (
        len(dataset)
        * args.num_epochs
        // (
            PartialState().num_processes
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
    )
    dataset = dataset.shuffle(seed=0).to_iterable_dataset(num_shards=PartialState().num_processes * 2)
    if PartialState().is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    PartialState().wait_for_everyone()
    checkpoint = get_last_checkpoint(args.output_dir)

    sft_config = SFTConfig(
        seed=0,
        output_dir=args.output_dir,
        save_only_model=args.save_only_model,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=True,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_total_limit=50,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate},
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        # only works when the save_strategy is 'steps', global steps (i.e., the step of param updating)
        save_steps=args.save_steps,
        logging_strategy='steps',
        logging_steps=1,
        report_to="tensorboard",
        logging_dir=f'tensorboard_logs/{args.prefix}_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        gradient_checkpointing=True,
        # if preprocess first, should set a large ddp_timeout
        ddp_timeout=21600,
        # the global step (i.e., all processes perfoorming a backward acount for one step)
        max_steps=max_step,
        accelerator_config={"dispatch_batches": False},
        # must set this, or the base model can not stop
        eos_token="<|im_end|>",
        completion_only_loss=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_liger_kernel=True
    )
    trainer = SFTTrainer(
        args=sft_config,
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    args = parse_args()
    train(args)