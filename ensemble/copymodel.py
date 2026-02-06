#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
os.environ["HF_HOME"] = "/root/buaa/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 避免当前项目根目录里的本地 `transformers/` 包抢占同名 pip 包，导致循环导入
_this_dir = os.path.dirname(os.path.abspath(__file__))              # .../EnsembleLLM/weights
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))  # /root/buaa/czh
if _project_root in sys.path:
    sys.path.remove(_project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="复制 base 模型到 stage0_m0")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace 模型名称，如 Qwen/Qwen3-8B-Base")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录，如 /root/buaa/czh/weights/ensemble/Qwen3-8B-Base")
    args = parser.parse_args()
    
    model_name = args.model_name
    target_dir = os.path.join(args.output_dir, "stage0_m0")

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

    print(f"✅ 完成：{model_name} 已复制到 {target_dir}")

if __name__ == "__main__":
    main()