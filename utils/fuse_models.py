import os
import json
import sys
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Handle execution both as package (`EnsembleLLM.utils`) and as top-level script (`utils`)
try:
    from EnsembleLLM.EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
except ImportError:
    try:
        from EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
    except ImportError as exc:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        try:
            from EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
        except ImportError:
            raise ImportError("Cannot locate `Qwen3ForEnsemble`. Please check PYTHONPATH.") from exc

def extract_submodel(ensemble_model_path, submodel_idx, save_dir, torch_dtype=torch.bfloat16):
    """
    从融合模型中提取指定的子模型并保存为独立模型。

    Args:
        ensemble_model_path (str): 融合模型路径
        submodel_idx (int): 要提取的子模型索引（0表示第一个，1表示第二个）
        save_dir (str): 保存的目录
        torch_dtype: 保存的权重精度 (默认 bfloat16)

    Returns:
        str: 保存路径
    """
    print(f"🔹 从融合模型中提取子模型:")
    print(f"   - 融合模型: {ensemble_model_path}")
    print(f"   - 子模型索引: {submodel_idx}")
    print(f"   - 保存到: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # 加载融合模型
    print("\n🔹 加载融合模型...")
    ensemble_model = QwenBoostForCausalLM.from_pretrained(
        ensemble_model_path,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="flash_attention_2"
    )
    ensemble_state_dict = ensemble_model.state_dict()

    # 提取子模型的权重
    submodel_prefix = f"sub_models.{submodel_idx}."
    extracted_state_dict = {}
    
    print(f"\n🔹 提取 {submodel_prefix} 的权重...")
    for key, value in ensemble_state_dict.items():
        if key.startswith(submodel_prefix):
            # 移除 sub_models.{idx}. 前缀
            new_key = key[len(submodel_prefix):]
            # 跳过 prev_*_proj 权重（这些是用于gate的，不属于原始模型）
            if "prev_" not in new_key:
                extracted_state_dict[new_key] = value.to(torch_dtype)
    
    print(f"   ✅ 提取了 {len(extracted_state_dict)} 个权重张量")

    # 保存 tokenizer
    print("\n🔹 保存 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(ensemble_model_path)
    tokenizer.save_pretrained(save_dir)

    # 加载基础模型结构（用于保存配置）
    print("\n🔹 保存 config.json ...")
    base_config = AutoConfig.from_pretrained(ensemble_model_path)
    # 从融合模型的配置中获取子模型配置
    if hasattr(base_config, "sub_model_cfgs") and base_config.sub_model_cfgs:
        # 复制子模型配置字典（避免修改原始配置）
        submodel_config_dict = dict(base_config.sub_model_cfgs[submodel_idx])
        # 移除融合相关的配置字段
        submodel_config_dict.pop("num_sub_models", None)
        submodel_config_dict.pop("is_llmboost", None)
        submodel_config_dict.pop("sub_model_cfgs", None)
        # 恢复为标准的 Qwen3 配置
        if "architectures" not in submodel_config_dict or not submodel_config_dict.get("architectures"):
            submodel_config_dict["architectures"] = ["Qwen3ForCausalLM"]
    else:
        # 如果没有子模型配置，使用基础配置
        submodel_config_dict = base_config.to_dict()
        submodel_config_dict.pop("num_sub_models", None)
        submodel_config_dict.pop("is_llmboost", None)
        submodel_config_dict.pop("sub_model_cfgs", None)
        if "architectures" not in submodel_config_dict or not submodel_config_dict.get("architectures"):
            submodel_config_dict["architectures"] = ["Qwen3ForCausalLM"]

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(submodel_config_dict, f, indent=2, ensure_ascii=False)

    # 保存权重
    print("\n🔹 保存权重...")
    final_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(extracted_state_dict, final_path)

    print(f"\n🎉 提取完成! 保存到: {final_path}")
    return final_path

def fuse_submodels(model_list, save_dir, torch_dtype=torch.bfloat16, fusion_lambda=0.5):
    """
    将多个 CausalLM 模型合并为一个 Ensemble 模型的权重格式。

    Args:
        model_list (List[str]): 模型路径列表或 HF 名称列表
        save_dir (str): 保存的目录
        torch_dtype: 保存的权重精度 (默认 bfloat16)
        fusion_lambda (float): 融合权重，前面的模型占 (1-lambda)，后面的模型占 lambda。默认 0.5 表示平均融合

    Returns:
        str: 保存路径
    """

    print(f"🔹 Loading {len(model_list)} models:")
    for m in model_list:
        print(f"   - {m}")

    os.makedirs(save_dir, exist_ok=True)

    print("\n🔹 Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_list[0])
    tokenizer.save_pretrained(save_dir)

    print("\n🔹 Loading model weights ...")
    models = [
        AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch_dtype)
        for m in model_list
    ]

    print("\n🔹 Saving config.json ...")
    config = AutoConfig.from_pretrained(model_list[0])
    config_dict = config.to_dict()

    # 确保保存 model_type，这对于正确加载模型很重要
    if "model_type" not in config_dict:
        # 如果 config 中没有 model_type，尝试从模型路径或配置推断
        if hasattr(config, "model_type"):
            config_dict["model_type"] = config.model_type
        elif "qwen2" in model_list[0].lower():
            config_dict["model_type"] = "qwen2"
        elif "qwen3" in model_list[0].lower():
            config_dict["model_type"] = "qwen3"
        else:
            pass

    sub_model_cfgs = []
    for idx, m in enumerate(model_list):
        sub_cfg = AutoConfig.from_pretrained(m)
        sub_cfg_dict = sub_cfg.to_dict()
        # 确保每个子配置也有 model_type
        if "model_type" not in sub_cfg_dict:
            if hasattr(sub_cfg, "model_type"):
                sub_cfg_dict["model_type"] = sub_cfg.model_type
            elif "qwen2" in m.lower():
                sub_cfg_dict["model_type"] = "qwen2"
            elif "qwen3" in m.lower():
                sub_cfg_dict["model_type"] = "qwen3"
        sub_model_cfgs.append(sub_cfg_dict)
    
    config_dict.update({
        "num_sub_models": len(models),
        # "num_hidden_layers": config.num_hidden_layers * len(models),
        # "layer_types": config.layer_types * len(models),
        "architectures": ["QwenBoostForCausalLM"],   # 让你的 Ensemble 类被识别
        "is_llmboost": False,
        "sub_model_cfgs": sub_model_cfgs,
        "fusion_lambda": fusion_lambda,  # 融合权重：前面的模型占 (1-lambda)，后面的模型占 lambda
    })
    
    print(f"🔹 保存的 model_type: {config_dict.get('model_type', '未设置')}")

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    merged_state_dict = {}

    for idx, model in enumerate(models):
        sub_prefix = f"sub_models.{idx}"
        print(f"   ➤ Processing {sub_prefix} ...")

        # 遍历每个 weight
        for k, v in model.state_dict().items():
            merged_state_dict[f"{sub_prefix}.{k}"] = v.to(torch_dtype)

        # lm_head （某些模型没有单独 lm_head）
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            merged_state_dict[f"{sub_prefix}.lm_head.weight"] = model.lm_head.weight.to(torch_dtype)

    print(f"   ✅ Collected {len(merged_state_dict)} tensors")

    final_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(merged_state_dict, final_path)

    print(f"\n🎉 Fusion complete! Saved to: {final_path}")
    return final_path

def load_fuse_model_tokenizer(model_name):
    model = QwenBoostForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,      # ❗❗不要自动放 GPU
        # low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_fuse_model_tokenizer_vote(model_name, freeze_first_model):
    # 判断是 qwen2 还是 qwen3：先尝试加载 config 检查
    is_qwen2 = False
    try:
        config = AutoConfig.from_pretrained(model_name)
        # 优先检查 config 中的 model_type
        if hasattr(config, 'model_type') and config.model_type:
            model_type_str = str(config.model_type).lower()
            is_qwen2 = 'qwen2' in model_type_str
            print(f"🔹 从 config.model_type 识别: {config.model_type} -> {'Qwen2' if is_qwen2 else 'Qwen3'}")
        # 如果 model_type 无法确定，检查 hidden_size（qwen2.5-3b 通常是 2048）
        if not is_qwen2 and hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
            # qwen2.5-3b 的 hidden_size 是 2048，qwen3 通常是 2560 或更大
            if hidden_size == 2048:
                is_qwen2 = True
                print(f"🔹 从 hidden_size={hidden_size} 推断为 Qwen2")
            elif hidden_size >= 2560:
                is_qwen2 = False
                print(f"🔹 从 hidden_size={hidden_size} 推断为 Qwen3")
        # 如果还是无法确定，检查 architectures
        if not is_qwen2 and hasattr(config, 'architectures') and config.architectures:
            is_qwen2 = any('qwen2' in str(arch).lower() for arch in config.architectures)
            if is_qwen2:
                print(f"🔹 从 architectures 识别为 Qwen2")
        # 最后检查模型路径
        if not is_qwen2:
            is_qwen2 = 'qwen2' in model_name.lower()
            if is_qwen2:
                print(f"🔹 从模型路径识别为 Qwen2")
    except Exception as e:
        # 如果加载 config 失败，根据路径判断
        print(f"⚠️  加载 config 失败: {e}，使用路径判断")
        is_qwen2 = 'qwen2' in model_name.lower()
    
    # 根据判断结果导入相应的模块
    if is_qwen2:
        try:
            from EnsembleLLM.EnsembleQwen2LLMBOOST.modeling_qwen2 import QwenBoostForCausalLM
        except ImportError:
            try:
                from EnsembleQwen2LLMBOOST.modeling_qwen2 import QwenBoostForCausalLM
            except ImportError as exc:
                _current_dir = os.path.dirname(os.path.abspath(__file__))
                _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
                if _project_root not in sys.path:
                    sys.path.insert(0, _project_root)
                try:
                    from EnsembleQwen2LLMBOOST.modeling_qwen2 import QwenBoostForCausalLM
                except ImportError:
                    raise ImportError("Cannot locate `Qwen2ForEnsemble`. Please check PYTHONPATH.") from exc
        print("🔹 使用 Qwen2 模型 (vote-base)")
    else:
        try:
            from EnsembleLLM.EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
        except ImportError:
            try:
                from EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
            except ImportError as exc:
                _current_dir = os.path.dirname(os.path.abspath(__file__))
                _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
                if _project_root not in sys.path:
                    sys.path.insert(0, _project_root)
                try:
                    from EnsembleQwen3.modeling_qwen3 import QwenBoostForCausalLM
                except ImportError:
                    raise ImportError("Cannot locate `Qwen3ForEnsemble`. Please check PYTHONPATH.") from exc
        print("🔹 使用 Qwen3 模型 (vote-base)")
    
    model = QwenBoostForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,      # ❗❗不要自动放 GPU
        # low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 如果需要冻结第一个模型（sub_models.0），则将其所有参数的 requires_grad 设置为 False
    # 处理 freeze_first_model 参数，支持 bool 类型和字符串类型
    should_freeze = False
    if isinstance(freeze_first_model, bool):
        should_freeze = freeze_first_model
    elif isinstance(freeze_first_model, str):
        should_freeze = freeze_first_model.lower() in ("true", "1", "yes", "on")
    else:
        should_freeze = bool(freeze_first_model)
    
    if should_freeze:
        frozen_count = 0
        for name, param in model.named_parameters():
            if name.startswith("sub_models.0"):
                param.requires_grad = False
                frozen_count += 1
        print(f"✅ 已冻结第一个模型 (sub_models.0) 的 {frozen_count} 个参数，这些参数将不参与训练")
    else:
        print("ℹ️  未启用冻结第一个模型，所有参数都将参与训练")
    return model, tokenizer
