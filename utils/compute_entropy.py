"""
Compute entropy 
Support automatic model type detection (Qwen / QwenBoost)
"""

import json
import os
## os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
## os.environ["HF_HOME"] = "/root/buaa/hf_cache"
import torch
import gc
import time
import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from torch.utils.data import DataLoader
from accelerate import PartialState


def detect_model_type(model_path: str) -> str:
    """
    Detect model type
    
    Returns:
        "qwen_boost"  "standard"
    """
    ##  QwenBoost 
    config_file = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            
            ##  ensemble 
            if "ensemble_config" in config or "num_submodels" in config:
                return "qwen_boost"
            
            ##  architectures 
            if "architectures" in config:
                for arch in config["architectures"]:
                    if "Boost" in arch or "Ensemble" in arch:
                        return "qwen_boost"
        except Exception as e:
            print(f"Warning: Read config.json failed: {e}")
    
    ##  QwenBoost 
    boost_indicator_files = ["ensemble_weights.json", "submodel_weights.json"]
    for indicator in boost_indicator_files:
        if os.path.exists(os.path.join(model_path, indicator)):
            return "qwen_boost"
    
    ## Default to standard model
    return "standard"


def load_model_and_tokenizer(model_path: str, device: torch.device, rank: int, stage: str = None):
    """
    Load model and tokenizer, determine which model type to use based on stage parameter
    
    Args:
        model_path: Model path
        device: 
        rank: rank
        stage: Training stage, if "stage3" then use QwenBoostForCausalLM (fused model), otherwise use AutoModelForCausalLM
    
    Returns:
        model, tokenizer, model_type
    """
    ## Determine model type based on stage parameter
    ## Only stage3 (fused model) uses QwenBoostForCausalLM
    import transformers
    print(f"Process Rank: {rank}, Transformers Version: {transformers.__version__}")
    if stage == "stage3":
        model_type = "qwen_boost"
        print(f"[Rank {rank}] Based on stage={stage} Using QwenBoostForCausalLM")
    else:
        model_type = "standard"
        print(f"[Rank {rank}] Based on stage={stage} Using AutoModelForCausalLM")
    
    ## Loading tokenizer
    print(f"[Rank {rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    ##  pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ## Loading model - force CPU load, avoid DeepSpeed sharding
    print(f"[Rank {rank}] Loading model (CPU)...")
    if model_type == "qwen_boost":
        from EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
        model = QwenBoostForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    
    ##  padding_idx 
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if hasattr(embeddings, "padding_idx") and embeddings.padding_idx is not None:
            num_embeddings = embeddings.weight.size(0)
            if embeddings.padding_idx >= num_embeddings:
                print(f"[Rank {rank}] Fix: padding_idx ({embeddings.padding_idx}) >= num_embeddings ({num_embeddings})，reset to None")
                embeddings.padding_idx = None
                if hasattr(model.config, "pad_token_id"):
                    model.config.pad_token_id = None
    
    ##  GPU
    print(f"[Rank {rank}] Move model to GPU {device}...")
    model = model.to(device)
    
    return model, tokenizer, model_type


def compute_entropy_for_model(
    model_path: str,
    data_files: list,
    output_path: str,
    entropy_field: str = "entropy_0",
    distributed_state: PartialState = None,
    stage: str = None,
) -> str:
    """
    Multi-GPU parallel compute model entropy on dataset and save to jsonl file
     stage Using QwenBoostForCausalLM  AutoModelForCausalLM
    
    Args:
        model_path: Model path
        data_files: Data file path list
        output_path: Output file path (final merged file)
        entropy_field: entropy field name, e.g., entropy_0, entropy_1, entropy_2
        distributed_state: PartialState object for distributed processing
        stage: Training stage， "stage3" Using QwenBoostForCausalLM，Using AutoModelForCausalLM
    
    Returns:
        Saved entropy file path
    """
    
    ##  PartialState（）
    if distributed_state is None:
        distributed_state = PartialState()
    
    rank = distributed_state.process_index
    world_size = distributed_state.num_processes
    is_main = distributed_state.is_main_process
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Starting multi-GPU parallel compute {entropy_field}")
        print(f"Model: {model_path}")
        print(f"Using .* GPU cards parallel compute")
        print(f"{'='*60}\n")
    
    ## Each process uses its own GPU
    device = torch.device(f"cuda:{distributed_state.local_process_index}")
    
    ## Load model and tokenizer (determine type based on stage parameter)
    model, tokenizer, model_type = load_model_and_tokenizer(model_path, device, rank, stage=stage)
    
    ## Load dataset
    all_records = []
    if isinstance(data_files, str):
        data_files = [data_files]
    
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
    
    dataset = Dataset.from_list(all_records)
    
    if is_main:
        print(f"Total dataset size: {len(dataset)}")
    
    ## Data sharding: each process processes a part of data
    dataset_shard = dataset.shard(num_shards=world_size, index=rank)
    print(f"[Rank {rank}] Processing data shard size: {len(dataset_shard)}")
    
    ## collate_fn
    def collate_fn(batch):
        full_texts = []
        prompt_texts = []
        orig_indices = []
        
        for item in batch:
            orig_indices.append(item["idx"])
            
            msgs = item["messages"]
            user_msg = None
            assistant_msg = None
            
            for m in msgs:
                if m.get("role") == "user" and user_msg is None:
                    user_msg = {"role": "user", "content": m["content"]}
                if m.get("role") == "assistant" and assistant_msg is None:
                    assistant_msg = {"role": "assistant", "content": m["content"]}
            
            if user_msg is None:
                user_msg = {"role": "user", "content": ""}
            if assistant_msg is None:
                assistant_msg = {"role": "assistant", "content": ""}
            
            prompt_text = tokenizer.apply_chat_template(
                [user_msg],
                tokenize=False,
                add_generation_prompt=False,
            )
            full_text = tokenizer.apply_chat_template(
                [user_msg, assistant_msg],
                tokenize=False,
                add_generation_prompt=False,
            )
            
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
        
        enc = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        
        prompt_ids = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )["input_ids"]
        
        labels = enc["input_ids"].clone()
        prompt_lens = []
        
        ##  pad_token_id 
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        
        for i in range(len(labels)):
            prompt_len = (prompt_ids[i] != pad_id).sum().item()
            prompt_lens.append(prompt_len)
            labels[i, :prompt_len] = -100
        
        enc["labels"] = labels
        enc["prompt_lens"] = torch.tensor(prompt_lens)
        enc["orig_idx"] = torch.tensor(orig_indices)
        
        return enc
    
    ## DataLoader
    dataloader = DataLoader(
        dataset_shard,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    results = []
    model.eval()
    
    ## 
    desc = f"[Rank {rank}] Computing {entropy_field}"
    for batch in tqdm.tqdm(dataloader, desc=desc, disable=not is_main):
        B = batch["input_ids"].shape[0]
        
        ## 
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=True
                )
            except Exception as e:
                print(f"[Rank {rank}] Error: Modelfailed - {e}")
                print(f"[Rank {rank}] input_ids shape: {batch['input_ids'].shape}")
                print(f"[Rank {rank}] input_ids max: {batch['input_ids'].max()}, min: {batch['input_ids'].min()}")
                raise
        
        logits = outputs.logits
        orig_idx = batch["orig_idx"].tolist()
        
        ## 
        for i in range(B):
            true_idx = orig_idx[i]
            answer_mask = batch["labels"][i] != -100
            token_ids = torch.where(answer_mask)[0]
            
            if len(token_ids) == 0:
                entropy_value = 0.0
            else:
                token_logits = logits[i, token_ids, :]
                probs = torch.softmax(token_logits, dim=-1)
                entropy_tokens = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                entropy_value = entropy_tokens.mean().item()
            
            results.append({
                "idx": true_idx,
                entropy_field: entropy_value
            })
    
    ## Each process saves its own results to temp file
    temp_output_dir = os.path.dirname(output_path)
    temp_output_name = os.path.basename(output_path).replace(".jsonl", f"_rank{rank}.jsonl")
    temp_output_path = os.path.join(temp_output_dir, temp_output_name)
    
    os.makedirs(temp_output_dir, exist_ok=True)
    with open(temp_output_path, "w", encoding="utf-8") as f:
        for it in results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    
    print(f"[Rank {rank}] Compute complete, saved to temp file: {temp_output_path}")
    print(f"[Rank {rank}] Starting GPU memory cleanup...")
    
    ## 
    try:
        if hasattr(model, 'hf_device_map'):
            for param in model.parameters():
                param.grad = None
    except Exception as e:
        print(f"[Rank {rank}] Warning when cleaning model parameters (can ignore): {e}")
    
    ## Delete all objects
    del results
    del dataloader
    del dataset_shard
    del dataset
    del all_records
    del tokenizer
    del model
    
    ## Force garbage collection
    gc.collect()
    gc.collect()
    
    ## Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        time.sleep(0.5)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[Rank {rank}] Cleaned GPU {distributed_state.local_process_index}: "
              f"Allocated {mem_allocated:.2f}GB, Reserved {mem_reserved:.2f}GB")
    
    print(f"[Rank {rank}] GPU memory cleanup complete")
    
    ## Wait for all processes to complete
    distributed_state.wait_for_everyone()
    
    ## 
    if is_main:
        print(f"\n{'='*60}")
        print(f"Starting to merge results from all processes...")
        print(f"{'='*60}")
        
        merged_results = {}
        
        for i in range(world_size):
            temp_file = os.path.join(
                temp_output_dir,
                os.path.basename(output_path).replace(".jsonl", f"_rank{i}.jsonl")
            )
            
            if os.path.exists(temp_file):
                print(f"Read results from rank: {temp_file}")
                with open(temp_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            idx = item["idx"]
                            merged_results[idx] = item
                
                ## Delete temp file
                os.remove(temp_file)
                print(f"Delete temp file: {temp_file}")
        
        ## 
        print(f"\nSave merged results to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for idx in sorted(merged_results.keys()):
                f.write(json.dumps(merged_results[idx], ensure_ascii=False) + "\n")
        
        print(f"✓ {entropy_field} Compute complete! Total")
        print(f"{'='*60}\n")
    
    ## Sync again
    distributed_state.wait_for_everyone()
    
    return output_path


def merge_entropy_files(entropy_files: list, output_path: str) -> str:
    """
    Merge multiple entropy files into one file
    
    Args:
        entropy_files: entropy file path list
        output_path: Output file path
    
    Returns:
        Merged file path
    """
    print(f"\nMerge entropy files to: {output_path}")
    
    ##  entropy 
    idx_to_entropy = {}
    for filepath in entropy_files:
        if not os.path.exists(filepath):
            print(f"  : ，: {filepath}")
            continue
        
        print(f"  Read: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    idx = item["idx"]
                    if idx not in idx_to_entropy:
                        idx_to_entropy[idx] = {"idx": idx}
                    idx_to_entropy[idx].update(item)
    
    ## 
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in sorted(idx_to_entropy.keys()):
            f.write(json.dumps(idx_to_entropy[idx], ensure_ascii=False) + "\n")
    
    print(f"✓ Merge complete, Total\n")
    return output_path

