import os
import json
import sys
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

## Handle execution both as package (`EnsembleLLM.utils`) and as top-level script (`utils`)
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
    Extract specified sub-model from ensemble model and save as independent model.

    Args:
        ensemble_model_path (str): Ensemble model path
        submodel_idx (int): Sub-model index to extract (0 for first, 1 for second)
        save_dir (str): Save directory
        torch_dtype: Saved weight precision (default bfloat16)

    Returns:
        str: Save path
    """
    print(f"Extracting sub-model from ensemble:")
    print(f"   - Ensemble model: {ensemble_model_path}")
    print(f"   - Sub-model index: {submodel_idx}")
    print(f"   - Save to: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    ## Loading ensemble model
    print("\n🔹 Loading ensemble model...")
    ensemble_model = QwenBoostForCausalLM.from_pretrained(
        ensemble_model_path,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="flash_attention_2"
    )
    ensemble_state_dict = ensemble_model.state_dict()

    ## Extract sub-model weights
    submodel_prefix = f"sub_models.{submodel_idx}."
    extracted_state_dict = {}
    
    print(f"\n🔹 Extracting {submodel_prefix}  weights...")
    for key, value in ensemble_state_dict.items():
        if key.startswith(submodel_prefix):
            ## Remove sub_models.{idx}. prefix
            new_key = key[len(submodel_prefix):]
            ## Skip prev_*_proj weights (these are for gate, don't belong to original model)
            if "prev_" not in new_key:
                extracted_state_dict[new_key] = value.to(torch_dtype)
    
    print(f"   ✅ Extracting {len(extracted_state_dict)}  weight tensors")

    ## Saving tokenizer
    print("\n🔹 Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(ensemble_model_path)
    tokenizer.save_pretrained(save_dir)

    ## Load base model structure (for saving config)
    print("\n🔹 Save config.json ...")
    base_config = AutoConfig.from_pretrained(ensemble_model_path)
    ## Get sub-model config from ensemble model config
    if hasattr(base_config, "sub_model_cfgs") and base_config.sub_model_cfgs:
        ## Copy sub-model config dict (avoid modifying original config)
        submodel_config_dict = dict(base_config.sub_model_cfgs[submodel_idx])
        ## Remove ensemble-related config fields
        submodel_config_dict.pop("num_sub_models", None)
        submodel_config_dict.pop("is_llmboost", None)
        submodel_config_dict.pop("sub_model_cfgs", None)
        ## Restore to standard Qwen3 config
        if "architectures" not in submodel_config_dict or not submodel_config_dict.get("architectures"):
            submodel_config_dict["architectures"] = ["Qwen3ForCausalLM"]
    else:
        ## If no sub-model config, use base config
        submodel_config_dict = base_config.to_dict()
        submodel_config_dict.pop("num_sub_models", None)
        submodel_config_dict.pop("is_llmboost", None)
        submodel_config_dict.pop("sub_model_cfgs", None)
        if "architectures" not in submodel_config_dict or not submodel_config_dict.get("architectures"):
            submodel_config_dict["architectures"] = ["Qwen3ForCausalLM"]

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(submodel_config_dict, f, indent=2, ensure_ascii=False)

    ## Saving weights
    print("\n🔹 Saving weights...")
    final_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(extracted_state_dict, final_path)

    print(f"\n🎉 Extracting! Save to: {final_path}")
    return final_path

def fuse_submodels(model_list, save_dir, torch_dtype=torch.bfloat16, fusion_lambda=0.5):
    """
     CausalLM Model Ensemble Model weights。

    Args:
        model_list (List[str]): Model path list or HF name list
        save_dir (str): Save directory
        torch_dtype: Saved weight precision (default bfloat16)
        fusion_lambda (float): Fusion weight, previous model accounts for (1-lambda), later model accounts for lambda. Default 0.5 means average fusion

    Returns:
        str: Save path
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

    ## Ensure model_type is saved, which is important for correct model loading
    if "model_type" not in config_dict:
        ## If no model_type in config, try to infer from model path or config
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
        ## Ensure each sub-config also has model_type
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
        ## "num_hidden_layers": config.num_hidden_layers * len(models),
        ## "layer_types": config.layer_types * len(models),
        "architectures": ["QwenBoostForCausalLM"],   ## Make your Ensemble class recognized
        "is_llmboost": False,
        "sub_model_cfgs": sub_model_cfgs,
        "fusion_lambda": fusion_lambda,  ## Fusion weight: previous model accounts for (1-lambda), later model accounts for lambda
    })
    
    print(f"🔹 Saved model_type: {config_dict.get('model_type', '')}")

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    merged_state_dict = {}

    for idx, model in enumerate(models):
        sub_prefix = f"sub_models.{idx}"
        print(f"   ➤ Processing {sub_prefix} ...")

        ## Iterate each weight
        for k, v in model.state_dict().items():
            merged_state_dict[f"{sub_prefix}.{k}"] = v.to(torch_dtype)

        ## lm_head （ lm_head）
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
        device_map=None,      ## ❗❗ GPU
        ## low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_fuse_model_tokenizer_vote(model_name, freeze_first_model):
    ##  qwen2  qwen3： config 
    is_qwen2 = False
    try:
        config = AutoConfig.from_pretrained(model_name)
        ##  config  model_type
        if hasattr(config, 'model_type') and config.model_type:
            model_type_str = str(config.model_type).lower()
            is_qwen2 = 'qwen2' in model_type_str
            print(f"🔹  config.model_type : {config.model_type} -> {'Qwen2' if is_qwen2 else 'Qwen3'}")
        ##  model_type ， hidden_size（qwen2.5-3b  2048）
        if not is_qwen2 and hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
            ## qwen2.5-3b  hidden_size  2048，qwen3  2560 
            if hidden_size == 2048:
                is_qwen2 = True
                print(f"🔹  hidden_size={hidden_size}  Qwen2")
            elif hidden_size >= 2560:
                is_qwen2 = False
                print(f"🔹  hidden_size={hidden_size}  Qwen3")
        ## ， architectures
        if not is_qwen2 and hasattr(config, 'architectures') and config.architectures:
            is_qwen2 = any('qwen2' in str(arch).lower() for arch in config.architectures)
            if is_qwen2:
                print(f"🔹  architectures  Qwen2")
        ## 
        if not is_qwen2:
            is_qwen2 = 'qwen2' in model_name.lower()
            if is_qwen2:
                print(f"🔹 Model Qwen2")
    except Exception as e:
        ##  config ，
        print(f"⚠️   config : {e}，Using")
        is_qwen2 = 'qwen2' in model_name.lower()
    
    ## 
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
        print("🔹 Using Qwen2 Model (vote-base)")
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
        print("🔹 Using Qwen3 Model (vote-base)")
    
    model = QwenBoostForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,      ## ❗❗ GPU
        ## low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## （sub_models.0）， requires_grad  False
    ## Processing freeze_first_model ， bool 
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
        print(f"✅ Model (sub_models.0)  {frozen_count} ，")
    else:
        print("ℹ️  Model，")
    return model, tokenizer
