#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
os.environ["HF_HOME"] = "/root/buaa/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Avoid local `transformers/` package in project root from conflicting with pip package, causing circular import
_this_dir = os.path.dirname(os.path.abspath(__file__))              # .../EnsembleLLM/weights
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))  # /root/buaa/czh
if _project_root in sys.path:
    sys.path.remove(_project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Copy base model to stage0_m0")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name, e.g., Qwen/Qwen3-8B-Base")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory, e.g., /root/buaa/czh/weights/ensemble/Qwen3-8B-Base")
    args = parser.parse_args()
    
    model_name = args.model_name
    target_dir = os.path.join(args.output_dir, "stage0_m0")

    os.makedirs(target_dir, exist_ok=True)
    print(f"Loading model from HuggingFace: {model_name}")

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

    print(f"Saving to: {target_dir}")
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    print(f"Completed: {model_name} copied to {target_dir}")

if __name__ == "__main__":
    main()