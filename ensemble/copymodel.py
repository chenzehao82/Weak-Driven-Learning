#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
os.environ["HF_HOME"] = "/root/buaa/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 避免当前项目根目录里的本地 `transformers/` 包抢占同名 pip 包，导致循环导入
_this_dir = os.path.dirname(os.path.abspath(__file__))              # .../EnsembleLLM/weights
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))  # /root/buaa/czh
if _project_root in sys.path:
    sys.path.remove(_project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen3-4B-Base"
    # 使用相对路径，基于脚本所在位置
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    target_dir = os.path.join(_project_root, "weights/ensemble/Qwen3-4B-Base/stage0_m0")

    os.makedirs(target_dir, exist_ok=True)
    print(f"🌐 从 HuggingFace 加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )

    print(f"💾 保存到: {target_dir}")
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    print("✅ 完成：Qwen3-4B-Base 已复制到 stage0_m0")

if __name__ == "__main__":
    main()