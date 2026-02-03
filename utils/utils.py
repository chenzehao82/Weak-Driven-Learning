import torch
import os
import json
import numpy as np
import random
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import pandas as pd
def apply_chat_template(tokenizer, question):
    """将输入问题转换为符合 chat 模板的文本"""
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return text

def unwrap_lm_head(lm_head):
    """
    通用解包函数：支持 Linear / Sequential / CastOutputToFloat 外包等情况
    返回最内部的 Linear 层。
    """
    if hasattr(lm_head, "weight"):  # 直接是 Linear
        return lm_head
    elif hasattr(lm_head, "_modules") and len(lm_head._modules) > 0:
        # 如果是 Sequential 或 CastOutputToFloat 等包装
        first = list(lm_head._modules.values())[0]
        return unwrap_lm_head(first)
    else:
        raise ValueError(f"无法识别 lm_head 结构: {lm_head.__class__.__name__}")


def prepare_model_for_int8_training(
    model, output_embedding_layer_name="lm_head", use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]
):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_8bit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_8bit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32

            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model

def generate_prompt(data_point):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


def load_data(file_path) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def format_max_memory(gpu_idx: int, safety_margin_gb: int = 2) -> str:
        total_gb = torch.cuda.get_device_properties(gpu_idx).total_memory // (1024 ** 3)
        alloc_gb = max(total_gb - safety_margin_gb, 1)
        return f"{alloc_gb}GiB"

def lm_per_sample_ce(
    logits: torch.Tensor,   # [B, T, V]
    labels: torch.Tensor,   # [B, T]
    pad_token_id: int = -100,
) -> torch.Tensor:          # [B] 每个样本的平均CE（忽略pad）
    B, T, V = logits.shape
    # 展平到token级
    ce_token = F.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
        ignore_index=pad_token_id,
        reduction='none'
    ).view(B, T)

    # 有效token掩码
    mask = (labels != pad_token_id).float()  # [B, T]
    # 防止某样本全是pad
    valid_cnt = mask.sum(dim=1).clamp_min(1.0)  # [B]
    ce_per_sample = (ce_token * mask).sum(dim=1) / valid_cnt  # [B]
    return ce_per_sample


def lm_weighted_loss(
    logits: torch.Tensor,   # [B, T, V]
    labels: torch.Tensor,   # [B, T]
    weights: torch.Tensor,  # [B]
    pad_token_id: int = -100,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    带样本权重的LM交叉熵；先做逐样本CE，再按weights做加权聚合。
    """
    ce_per_sample = lm_per_sample_ce(logits, labels, pad_token_id=pad_token_id)  # [B]
    if reduction == 'none':
        return weights * ce_per_sample
    # 归一化的加权平均更稳（不受batch大小影响）
    w = weights.clamp_min(1e-8)
    return (w * ce_per_sample).sum() / w.sum()


# =========================
# 工具函数：由上一模型输出生成样本权重（AdaBoost风格）
# =========================
@torch.no_grad()
def compute_adaboost_weights_from_prev_logits(
    prev_logits: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,       # [B, T]
    pad_token_id: int = -100,
    alpha: float = 2.0,         # 放大系数（1~3常用）
    w_min: float = 0.2,
    w_max: float = 5.0,
    normalize_by_batch_mean: bool = True,
) -> torch.Tensor:              # [B]
    """
    使用上一模型的逐样本平均CE作为“难度”，生成weights。
    公式：w_i = exp(alpha * CE_i / mean(CE))（再clip）
    """
    ce_prev = lm_per_sample_ce(prev_logits, labels, pad_token_id=pad_token_id)  # [B]
    if normalize_by_batch_mean:
        scale = ce_prev.mean().clamp_min(1e-8)
        weights = torch.exp(alpha * (ce_prev / scale))
    else:
        weights = torch.exp(alpha * ce_prev)
    return weights.clamp(w_min, w_max)

def enable_sft_training_optimizations(model):
    """
    正确开启 SFT（全参数训练）所需的所有设置。
    包括：
      - gradient checkpointing（降低激活显存）
      - 关闭 use_cache（必须）
      - 输入梯度（仅全参不需要，LoRA 需要）
    """
    # 关闭 KV cache，否则 checkpointing 会报错
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 打开 gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 有些模型还需要 config.flag
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = True
        
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

def load_model_tokenizer(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,      # ❗❗不要自动放 GPU
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # qwen2.5 use
    if "Qwen2.5" in model_name:
        print("Qwen2.5 model detected")
        origin_eos_token_id = tokenizer.eos_token_id
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        print(f"origin_eos_token_id: {origin_eos_token_id}")
        print(f"eos_token_id: {eos_token_id}")
        if origin_eos_token_id != eos_token_id:
            tokenizer.eos_token_id = eos_token_id
            print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
            model.config.eos_token_id = eos_token_id
            print(f"model.config.eos_token_id: {model.config.eos_token_id}")
            model.generation_config.eos_token_id = [origin_eos_token_id, eos_token_id]
            print(f"model.generation_config.eos_token_id: {model.generation_config.eos_token_id}")

    return model, tokenizer


def load_data(file_path):
    all_records = []
    with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                all_records.append(json.loads(line))
    dataset = Dataset.from_list(all_records)
    return dataset


def load_entropy_df(file_path: str) -> pd.DataFrame:
    """
    读取熵结果 jsonl:
    每行至少包含: { "idx": ..., "entropy_0": ..., "entropy_1": ... }
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    assert "idx" in df.columns, "entropy file must contain 'idx'"

    # 兼容两种格式：
    # 1）stage1/2/3 使用的：包含 entropy_0 / entropy_1 (/ entropy_2)
    # 2）stage4 使用的：仅包含 entropy_2 / entropy_3，此时将其映射为 entropy_0 / entropy_1 以复用采样公式
    
    if "entropy_0" in df.columns and "entropy_1" in df.columns:
        return df

    if "entropy_1" in df.columns and "entropy_2" in df.columns:
        df = df.copy()
        # 映射为 AdaBoost 需要的两个熵值：
        # entropy_0 = entropy_2 (之前的状态)
        # entropy_1 = entropy_3 (当前状态)
        df["entropy_0"] = df["entropy_1"]
        df["entropy_1"] = df["entropy_2"]
        return df
    # 仅有 entropy_2 / entropy_3 的情况（例如 entropy_2_3_merged.jsonl，用于 Stage4 AdaBoost）
    if "entropy_2" in df.columns and "entropy_3" in df.columns:
        df = df.copy()
        # 映射为 AdaBoost 需要的两个熵值：
        # entropy_0 = entropy_2 (之前的状态)
        # entropy_1 = entropy_3 (当前状态)
        df["entropy_0"] = df["entropy_2"]
        df["entropy_1"] = df["entropy_3"]
        return df

    raise AssertionError(
        f"entropy file must contain 'entropy_0'/'entropy_1' "
        f"or 'entropy_2'/'entropy_3', got columns: {list(df.columns)}"
    )
