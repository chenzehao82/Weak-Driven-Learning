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
    """Convert input question to chat template format"""
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return text

def unwrap_lm_head(lm_head):
    """
    Universal unwrapping function: supports Linear / Sequential / CastOutputToFloat wrapping etc.
    Return the innermost Linear layer.
    """
    if hasattr(lm_head, "weight"):  ##  Linear
        return lm_head
    elif hasattr(lm_head, "_modules") and len(lm_head._modules) > 0:
        ##  Sequential  CastOutputToFloat 
        first = list(lm_head._modules.values())[0]
        return unwrap_lm_head(first)
    else:
        raise ValueError(f"Cannot recognize lm_head structure: {lm_head.__class__.__name__}")


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
        ## freeze base model's layers
        param.requires_grad = False

        if loaded_in_8bit:
            ## cast layer norm in fp32 for stability for 8-bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_8bit and use_gradient_checkpointing:
        ## For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        ## enable gradient checkpointing for memory efficiency
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

                #### Instruction:
                {data_point["instruction"]}
                
                #### Response:
                {data_point["output"]}""" ## noqa: E501


def load_data(file_path) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"cannot find dataset file: {file_path}")
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
    logits: torch.Tensor,   ## [B, T, V]
    labels: torch.Tensor,   ## [B, T]
    pad_token_id: int = -100,
) -> torch.Tensor:          ## [B] eachaverage CE per sample (ignore pad)
    B, T, V = logits.shape
    ## token
    ce_token = F.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
        ignore_index=pad_token_id,
        reduction='none'
    ).view(B, T)

    ## Valid token mask
    mask = (labels != pad_token_id).float()  ## [B, T]
    ## Prevent sample from being all pad
    valid_cnt = mask.sum(dim=1).clamp_min(1.0)  ## [B]
    ce_per_sample = (ce_token * mask).sum(dim=1) / valid_cnt  ## [B]
    return ce_per_sample


def lm_weighted_loss(
    logits: torch.Tensor,   ## [B, T, V]
    labels: torch.Tensor,   ## [B, T]
    weights: torch.Tensor,  ## [B]
    pad_token_id: int = -100,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    LM cross-entropy with sample weights; first do per-sample CE, then do weighted aggregation by weights.
    """
    ce_per_sample = lm_per_sample_ce(logits, labels, pad_token_id=pad_token_id)  ## [B]
    if reduction == 'none':
        return weights * ce_per_sample
    ## Normalized weighted average is more stable (unaffected by batch size)
    w = weights.clamp_min(1e-8)
    return (w * ce_per_sample).sum() / w.sum()


## =========================
## Utility function: generate sample weights from previous model output (AdaBoost style)
## =========================
@torch.no_grad()
def compute_adaboost_weights_from_prev_logits(
    prev_logits: torch.Tensor,  ## [B, T, V]
    labels: torch.Tensor,       ## [B, T]
    pad_token_id: int = -100,
    alpha: float = 2.0,         ## Amplification coefficient (1~3 common)
    w_min: float = 0.2,
    w_max: float = 5.0,
    normalize_by_batch_mean: bool = True,
) -> torch.Tensor:              ## [B]
    """
    UsingModelSampleCE“”，weights。
    Formula: w_i = exp(alpha * CE_i / mean(CE)) (then clip)
    """
    ce_prev = lm_per_sample_ce(prev_logits, labels, pad_token_id=pad_token_id)  ## [B]
    if normalize_by_batch_mean:
        scale = ce_prev.mean().clamp_min(1e-8)
        weights = torch.exp(alpha * (ce_prev / scale))
    else:
        weights = torch.exp(alpha * ce_prev)
    return weights.clamp(w_min, w_max)

def enable_sft_training_optimizations(model):
    """
    Correctly enable all settings needed for SFT (full parameter training).
    Including:
      - gradient checkpointing (reduce activation memory)
      - close use_cache (required)
      - input gradients (not needed for full parameter, needed for LoRA)
    """
    ## Close KV cache, otherwise checkpointing will error
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    ## Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ##  config.flag
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = True
        
def check_deepspeed_status(trainer):
    """Improved DeepSpeed status check"""
    try:
        ## Method 1: Check if trainer uses deepspeed
        if hasattr(trainer, 'is_deepspeed_enabled') and trainer.is_deepspeed_enabled:
            print("✓ DeepSpeed")
            return True
            
        ## Method 2: Check if model is wrapped by DeepSpeed
        model = trainer.model
        if hasattr(model, 'module') and hasattr(model.module, 'engine'):
            engine = model.module.engine
            print("=== DeepSpeed ===")
            print(f"ZeRO: {engine.zero_optimization_stage()}")
            print(f": {type(engine.optimizer).__name__}")
            return True
            
        ## Method 3: Directly check accelerate status
        from accelerate.utils import is_deepspeed_available
        if is_deepspeed_available():
            try:
                from deepspeed import comm as dist
                if dist.is_initialized():
                    print("✓ DeepSpeed")
                    return True
            except:
                pass
        
        print("✗ DeepSpeed")
        return False
    except Exception as e:
        print(f"✗ DeepSpeed: {e}")
        return False

def load_model_tokenizer(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,      ## ❗❗ GPU
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ## qwen2.5 use
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
    Read entropy results jsonl:
    Each line must contain at least: { "idx": ..., "entropy_0": ..., "entropy_1": ... }
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

    ## Compatible with two formats:
    ## 1）stage1/2/3 ： entropy_0 / entropy_1 (/ entropy_2)
    ## 2）stage4 used: contains only entropy_2 / entropy_3, map them to entropy_0 / entropy_1 to reuse sampling formula
    
    if "entropy_0" in df.columns and "entropy_1" in df.columns:
        return df

    if "entropy_1" in df.columns and "entropy_2" in df.columns:
        df = df.copy()
        ## Map to two entropy values needed by AdaBoost:
        ## entropy_0 = entropy_2 ()
        ## entropy_1 = entropy_3 ()
        df["entropy_0"] = df["entropy_1"]
        df["entropy_1"] = df["entropy_2"]
        return df
    ## Case with only entropy_2 / entropy_3 (e.g., entropy_2_3_merged.jsonl, for Stage4 AdaBoost)
    if "entropy_2" in df.columns and "entropy_3" in df.columns:
        df = df.copy()
        ## Map to two entropy values needed by AdaBoost:
        ## entropy_0 = entropy_2 ()
        ## entropy_1 = entropy_3 ()
        df["entropy_0"] = df["entropy_2"]
        df["entropy_1"] = df["entropy_3"]
        return df

    raise AssertionError(
        f"entropy file must contain 'entropy_0'/'entropy_1' "
        f"or 'entropy_2'/'entropy_3', got columns: {list(df.columns)}"
    )
