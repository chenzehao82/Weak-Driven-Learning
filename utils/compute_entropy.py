"""
多卡并行计算 entropy 的工具模块
支持自动检测模型类型（Qwen / QwenBoost）
"""

import json
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/root/buaa/hf_cache"
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
    检测模型类型
    
    Returns:
        "qwen_boost" 或 "standard"
    """
    # 检查是否存在 QwenBoost 特有的配置文件
    config_file = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            
            # 检查是否有 ensemble 相关的配置
            if "ensemble_config" in config or "num_submodels" in config:
                return "qwen_boost"
            
            # 检查 architectures 字段
            if "architectures" in config:
                for arch in config["architectures"]:
                    if "Boost" in arch or "Ensemble" in arch:
                        return "qwen_boost"
        except Exception as e:
            print(f"警告: 读取 config.json 失败: {e}")
    
    # 检查是否存在 QwenBoost 特有的文件
    boost_indicator_files = ["ensemble_weights.json", "submodel_weights.json"]
    for indicator in boost_indicator_files:
        if os.path.exists(os.path.join(model_path, indicator)):
            return "qwen_boost"
    
    # 默认为标准模型
    return "standard"


def load_model_and_tokenizer(model_path: str, device: torch.device, rank: int, stage: str = None):
    """
    加载模型和 tokenizer，根据 stage 参数决定使用哪种模型类型
    
    Args:
        model_path: 模型路径
        device: 设备
        rank: 进程rank
        stage: 训练阶段，如果为 "stage3" 则使用 QwenBoostForCausalLM（融合模型），否则使用 AutoModelForCausalLM
    
    Returns:
        model, tokenizer, model_type
    """
    # 根据 stage 参数决定模型类型
    # 只有 stage3（融合后的模型）才使用 QwenBoostForCausalLM
    import transformers
    print(f"Process Rank: {rank}, Transformers Version: {transformers.__version__}")
    if stage == "stage3":
        model_type = "qwen_boost"
        print(f"[Rank {rank}] 根据 stage={stage} 使用 QwenBoostForCausalLM")
    else:
        model_type = "standard"
        print(f"[Rank {rank}] 根据 stage={stage} 使用 AutoModelForCausalLM")
    
    # 加载 tokenizer
    print(f"[Rank {rank}] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 - 强制 CPU 加载，避开 DeepSpeed 分片
    print(f"[Rank {rank}] 加载模型 (CPU)...")
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
    
    # 修复 padding_idx 问题
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if hasattr(embeddings, "padding_idx") and embeddings.padding_idx is not None:
            num_embeddings = embeddings.weight.size(0)
            if embeddings.padding_idx >= num_embeddings:
                print(f"[Rank {rank}] 修复: padding_idx ({embeddings.padding_idx}) >= num_embeddings ({num_embeddings})，重置为 None")
                embeddings.padding_idx = None
                if hasattr(model.config, "pad_token_id"):
                    model.config.pad_token_id = None
    
    # 移动到 GPU
    print(f"[Rank {rank}] 将模型移动到 GPU {device}...")
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
    多卡并行计算模型在数据集上的 entropy，并保存到 jsonl 文件
    根据 stage 参数决定使用 QwenBoostForCausalLM 或 AutoModelForCausalLM
    
    Args:
        model_path: 模型路径
        data_files: 数据文件路径列表
        output_path: 输出文件路径（最终合并后的文件）
        entropy_field: entropy 字段名，如 entropy_0, entropy_1, entropy_2
        distributed_state: PartialState 对象，用于分布式处理
        stage: 训练阶段，如果为 "stage3" 则使用 QwenBoostForCausalLM，否则使用 AutoModelForCausalLM
    
    Returns:
        保存的 entropy 文件路径
    """
    
    # 创建 PartialState（如果没有提供）
    if distributed_state is None:
        distributed_state = PartialState()
    
    rank = distributed_state.process_index
    world_size = distributed_state.num_processes
    is_main = distributed_state.is_main_process
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"开始多卡并行计算 {entropy_field}")
        print(f"模型: {model_path}")
        print(f"使用 {world_size} 张卡并行计算")
        print(f"{'='*60}\n")
    
    # 每个进程使用自己的 GPU
    device = torch.device(f"cuda:{distributed_state.local_process_index}")
    
    # 加载模型和 tokenizer（根据 stage 参数决定类型）
    model, tokenizer, model_type = load_model_and_tokenizer(model_path, device, rank, stage=stage)
    
    # 加载数据集
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
        print(f"总数据集大小: {len(dataset)}")
    
    # 数据分片：每个进程处理一部分数据
    dataset_shard = dataset.shard(num_shards=world_size, index=rank)
    print(f"[Rank {rank}] 处理数据分片大小: {len(dataset_shard)}")
    
    # collate_fn
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
        
        # 使用安全的 pad_token_id 检查
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
    
    # DataLoader
    dataloader = DataLoader(
        dataset_shard,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    results = []
    model.eval()
    
    # 计算熵
    desc = f"[Rank {rank}] Computing {entropy_field}"
    for batch in tqdm.tqdm(dataloader, desc=desc, disable=not is_main):
        B = batch["input_ids"].shape[0]
        
        # 移动数据到设备
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
                print(f"[Rank {rank}] 错误: 模型前向传播失败 - {e}")
                print(f"[Rank {rank}] input_ids shape: {batch['input_ids'].shape}")
                print(f"[Rank {rank}] input_ids max: {batch['input_ids'].max()}, min: {batch['input_ids'].min()}")
                raise
        
        logits = outputs.logits
        orig_idx = batch["orig_idx"].tolist()
        
        # 计算熵
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
    
    # 每个进程保存自己的结果到临时文件
    temp_output_dir = os.path.dirname(output_path)
    temp_output_name = os.path.basename(output_path).replace(".jsonl", f"_rank{rank}.jsonl")
    temp_output_path = os.path.join(temp_output_dir, temp_output_name)
    
    os.makedirs(temp_output_dir, exist_ok=True)
    with open(temp_output_path, "w", encoding="utf-8") as f:
        for it in results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    
    print(f"[Rank {rank}] 计算完成，保存到临时文件: {temp_output_path}")
    print(f"[Rank {rank}] 开始清理 GPU 内存...")
    
    # 彻底清理内存
    try:
        if hasattr(model, 'hf_device_map'):
            for param in model.parameters():
                param.grad = None
    except Exception as e:
        print(f"[Rank {rank}] 清理模型参数时出现警告（可忽略）: {e}")
    
    # 删除所有对象
    del results
    del dataloader
    del dataset_shard
    del dataset
    del all_records
    del tokenizer
    del model
    
    # 强制垃圾回收
    gc.collect()
    gc.collect()
    
    # 清空 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        time.sleep(0.5)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[Rank {rank}] 清理后 GPU {distributed_state.local_process_index}: "
              f"已分配 {mem_allocated:.2f}GB, 已保留 {mem_reserved:.2f}GB")
    
    print(f"[Rank {rank}] GPU 内存清理完成")
    
    # 等待所有进程完成
    distributed_state.wait_for_everyone()
    
    # 主进程合并所有结果
    if is_main:
        print(f"\n{'='*60}")
        print(f"开始合并所有进程的结果...")
        print(f"{'='*60}")
        
        merged_results = {}
        
        for i in range(world_size):
            temp_file = os.path.join(
                temp_output_dir,
                os.path.basename(output_path).replace(".jsonl", f"_rank{i}.jsonl")
            )
            
            if os.path.exists(temp_file):
                print(f"读取 rank {i} 的结果: {temp_file}")
                with open(temp_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            idx = item["idx"]
                            merged_results[idx] = item
                
                # 删除临时文件
                os.remove(temp_file)
                print(f"删除临时文件: {temp_file}")
        
        # 保存合并后的结果
        print(f"\n保存合并结果到: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for idx in sorted(merged_results.keys()):
                f.write(json.dumps(merged_results[idx], ensure_ascii=False) + "\n")
        
        print(f"✓ {entropy_field} 计算完成！共 {len(merged_results)} 条记录")
        print(f"{'='*60}\n")
    
    # 再次同步
    distributed_state.wait_for_everyone()
    
    return output_path


def merge_entropy_files(entropy_files: list, output_path: str) -> str:
    """
    合并多个 entropy 文件到一个文件中
    
    Args:
        entropy_files: entropy 文件路径列表
        output_path: 输出文件路径
    
    Returns:
        合并后的文件路径
    """
    print(f"\n合并 entropy 文件到: {output_path}")
    
    # 读取所有 entropy 文件
    idx_to_entropy = {}
    for filepath in entropy_files:
        if not os.path.exists(filepath):
            print(f"  警告: 文件不存在，跳过: {filepath}")
            continue
        
        print(f"  读取: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    idx = item["idx"]
                    if idx not in idx_to_entropy:
                        idx_to_entropy[idx] = {"idx": idx}
                    idx_to_entropy[idx].update(item)
    
    # 保存合并结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in sorted(idx_to_entropy.keys()):
            f.write(json.dumps(idx_to_entropy[idx], ensure_ascii=False) + "\n")
    
    print(f"✓ 合并完成，共 {len(idx_to_entropy)} 条记录\n")
    return output_path

