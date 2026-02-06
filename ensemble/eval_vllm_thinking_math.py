#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
import argparse, json, os, re, time, random, sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import re
from pathlib import Path

# Add parent directory to path to import utils
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# -----------------------------
# Import Latex verification dependencies
# -----------------------------
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import datetime
from utils.prompts import SYSTEM_PROMPT

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


BOXED_PAT = re.compile(r"\\boxed\{([^{}]+)\}")
INT_PAT   = re.compile(r"(-?\d+)")

def apply_chat_template(tokenizer, question, thinking: bool = True) -> str:
    text = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )
    return text


def load_general_dataset(path: str) -> List[Dict[str, Any]]:
    data = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                data = obj
            elif isinstance(obj, dict) and "data" in obj:
                data = obj["data"]
            else:
                raise ValueError("JSON format not recognized.")
    else:
        raise ValueError("dataset must be .json or .jsonl")

    norm = []
    for i, ex in enumerate(data):
        # 尝试多个可能的问题字段名
        q = ex.get("question") or ex.get("instruction") or ex.get("problem") or ex.get("prompt") or ex.get("input")
        # 尝试多个可能的答案字段名
        a = ex.get("answer") or ex.get("label") or ex.get("solution") or ex.get("output")
        if q is None or a is None:
            continue
        norm.append({"id": ex.get("id", i), "question": q, "answer": str(a).strip()})
    return norm

# -----------------------------
# 简单数字/字母提取方式（用于简单数学数据集）
# -----------------------------
def extract_answer_number(sentence: str) -> float:
    """从文本中提取数字答案（取最后一个数字）"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    try:
        pred_answer = float(pred[-1])
    except ValueError:
        pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """从文本中提取字母答案（A-E）"""
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        return pred_answers[-1]
    else:
        return ''


def verify_simple_number(pred_text: str, gold: str, miss: float = 1e-3) -> Optional[bool]:
    """简单数字验证"""
    try:
        # 尝试将 gold 转为 float
        try:
            gold_num = float(gold)
        except ValueError:
            return None
        
        # 提取预测答案
        pred_num = extract_answer_number(pred_text)
        
        if pred_num == float('inf'):
            return False
        
        # 比较
        return abs(gold_num - pred_num) <= miss
    except Exception as e:
        print(f"❌ Simple number verification failed: {e}")
        return None


def verify_simple_letter(pred_text: str, gold: str) -> Optional[bool]:
    """简单字母验证"""
    try:
        pred_letter = extract_answer_letter(pred_text)
        if not pred_letter:
            return False
        return pred_letter == gold.strip().upper()
    except Exception as e:
        print(f"❌ Simple letter verification failed: {e}")
        return None

# -----------------------------
# Latex-based Verification（用于复杂数学数据集）
# -----------------------------
def verify_with_latex(idx: str, pred: str, gold: str) -> Optional[bool]:
    try:
        gold = '$' + gold.strip().strip('$') + '$'
        gold_parsed = parse(
            gold,
            extraction_mode="first_match",
        )
        if len(gold_parsed) == 0:
            print(f"⚠️ Failed to parse gold solution: {gold}")
            return None

        answer_parsed = parse(
            pred,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="all",
        )
        print(f"id: {idx}, pred_parsed: {answer_parsed}, gold_parsed: {gold_parsed}, gold: {gold}")
        return bool(verify(gold_parsed, answer_parsed))
    except Exception as e:
        print(f"❌ verify failed: {e}\n id={idx}\n pred={pred}\n gold={gold}")
        return None


def get_verification_method(dataset_name: str):
    """根据数据集名称返回对应的验证方法"""
    dataset_name = dataset_name.lower()
    
    # 简单数字提取的数据集
    simple_number_datasets = [
        "multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps", 
        "math_10k_500", "math_10k_small_5", "math_10k"
    ]
    
    # 字母选择的数据集
    letter_datasets = ["aqua"]
    
    # LaTeX 验证的数据集
    latex_datasets = ["aime", "aime2025", "math500", "math"]
    
    if any(ds in dataset_name for ds in simple_number_datasets):
        return "simple_number"
    elif any(ds in dataset_name for ds in letter_datasets):
        return "simple_letter"
    elif any(ds in dataset_name for ds in latex_datasets):
        return "latex"
    else:
        # 默认使用 LaTeX 验证
        print(f"⚠️ Unknown dataset '{dataset_name}', defaulting to LaTeX verification")
        return "latex"


def should_use_chat_template(dataset_name: str) -> bool:
    """
    根据数据集名称判断是否应该使用 chat_template
    简单数据集（如 math_10k, gsm8k 等）不需要 chat_template
    复杂数据集（如 aime, math500）需要 chat_template
    """
    dataset_name = dataset_name.lower()
    
    # 不需要 chat_template 的数据集（通常是简单的数学问题）
    simple_datasets = [
        "multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps", 
        "math_10k", "aqua"
    ]
    
    # 检查是否是简单数据集
    if any(ds in dataset_name for ds in simple_datasets):
        return False
    
    # 其他数据集（aime, math500 等）使用 chat_template
    return True

# -----------------------------
# VLLM Backend
# -----------------------------
class VLLMBackend:
    def __init__(self, model, tensor_parallel_size=1, max_model_len=8192, dtype=None, gpu_memory_utilization=0.9, seed=None):
        from vllm import LLM, SamplingParams
        self.LLM = LLM(model=model, 
                        tensor_parallel_size=tensor_parallel_size, 
                        max_model_len=max_model_len, 
                        download_dir="/root/buaa/hf_cache",
                        gpu_memory_utilization=gpu_memory_utilization,
                        seed=seed)
        self.sp = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=max_model_len, stop=None, seed=seed)

    def generate(self, prompts: List[str]) -> List[str]:
        outs = self.LLM.generate(prompts, self.sp)
        return [o.outputs[0].text for o in outs]

# -----------------------------
# Evaluator
# -----------------------------
def evaluate(dataset_path: str,
             model_path: str,
             tp: int = 1,
             out_csv: str = "results/eval_detail.csv",
             out_json: str = "results/eval_summary.json",
             max_model_len: int = 1024,
             thinking: bool = True,
             repeat: int = 1,
             dataset_name: str = None,
             seed: int = None):

    samples = load_general_dataset(dataset_path)

    # ===== 确定数据集名称 =====
    if dataset_name is None:
        # 从路径中提取数据集名称
        dataset_name = dataset_path.split("/")[-2] if "/" in dataset_path else "unknown"
    
    # 统一：始终使用 chat_template + LaTeX 验证
    use_chat_template = True
    verification_method = "latex"
    print(f"📊 Dataset: {dataset_name}, Verification method: {verification_method}")
    print(f"🔧 Use chat_template: {use_chat_template}")

    # ===== 扩充重复样本 =====
    expanded_samples = []
    for ex in samples:
        for k in range(repeat):
            ex_copy = ex.copy()
            ex_copy["id"] = f"{ex['id']}_rep{k+1}"
            expanded_samples.append(ex_copy)
    print(f"Loaded {len(samples)} samples → expanded to {len(expanded_samples)} (repeat={repeat})")

    
    engine = VLLMBackend(model_path, tensor_parallel_size=tp, max_model_len=max_model_len, seed=seed)
    tokenizer = engine.LLM.get_tokenizer()

    # 始终使用 chat_template 构造对话式 prompt
    prompts = [apply_chat_template(tokenizer, question=s["question"], thinking=thinking) for s in expanded_samples]
    
    print(f"示例 Prompt: {prompts[0]}...")
    # exit(0)
    B = 500
    preds, latencies = [], []
    for i in range(0, len(prompts), B):
        batch = prompts[i:i+B]
        t0 = time.time()
        texts = engine.generate(batch)
        dt = time.time() - t0
        per_item = dt / len(batch)
        latencies.extend([per_item] * len(batch))
        preds.extend(texts)

    # ========== 评估（统一使用 LaTeX 验证） ==========
    rows, correct = [], 0
    for ex, text, lat in zip(expanded_samples, preds, latencies):
        verified = verify_with_latex(ex["id"], text, ex["answer"])
        
        is_ok = (verified is True)
        correct += int(is_ok)
        rows.append({
            "id": ex["id"],
            "orig_id": ex["id"].split("_rep")[0],   # ⭐ 原始样本 id（重要）
            "gt": ex["answer"],
            "pred_text": text.strip(),
            "verified": verified,
            "correct": int(is_ok),
            "latency_s": round(lat, 4),
            "question": ex["question"][:2000],
            "verification_method": verification_method,
        })

    # ====== Acc (sample-level, considers each repetition independently) ======
    total = len(expanded_samples)
    acc = correct / total

    # ====== Pass@N (original-sample-level) ======
    # 例如 repeat=10 时，就是 pass@10
    grouped = {}
    for r in rows:
        oid = r["orig_id"]
        grouped.setdefault(oid, []).append(r["correct"])

    pass_total = len(grouped)  # 原始样本数
    pass_success = sum(1 for oid, corr_list in grouped.items() if any(corr_list))
    pass_at_k = pass_success / pass_total if pass_total > 0 else 0.0

    print(f"Pass@{repeat}: {pass_at_k:.4f} ({pass_success}/{pass_total})")

    # ====== 保存 CSV ======
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # ====== 保存 JSON ======
    summary = {
        "acc": acc,
        "correct": correct,
        "total": total,
        "repeat": repeat,
        "pass_at_k": pass_at_k,
        "pass_success": pass_success,
        "orig_total": pass_total,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "verification_method": verification_method,
        "use_chat_template": use_chat_template,
        "model_path": model_path,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "csv_file": out_csv,
        "seed": seed,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ====== 日志 ======
    print(f"\n========== Evaluation ==========")
    print(f"Dataset: {dataset_path}")
    print(f"Dataset Name: {dataset_name}")
    print(f"Verification: {verification_method}")
    print(f"Chat Template: {use_chat_template}")
    print(f"Model:   {model_path}")
    print(f"Seed:    {seed}")
    print(f"Samples: {len(samples)} × {repeat} = {total}")
    print(f"Acc:     {acc:.4f}  ({correct}/{total})")
    print(f"Pass@{repeat}: {pass_at_k:.4f}  ({pass_success}/{pass_total})")
    print(f"Detail CSV saved to: {out_csv}")
    print(f"Summary JSON saved to: {out_json}")

    return acc

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/root/buaa/czh/BoostLLM/Decoder_Only/dataset/math500/test.jsonl", type=str)
    p.add_argument("--model", default="/root/buaa/czh/Weak-Driving Learning/weights/ensemble/Qwen3-4B-Base/stage1_m1", type=str)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--max_model_len", type=int, default=1024 * 4)
    p.add_argument("--thinking", type=bool, default=False)
    p.add_argument("--repeat", type=int, default=3, help="重复测试次数")
    p.add_argument("--epoch", type=str, default=None, help="训练轮数")
    p.add_argument("--dataset_name", type=str, default=None, help="数据集名称（用于保存路径和日志）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现性）")
    args = p.parse_args()

     # === 自动加载最新 checkpoint ===
    model_path = args.model.rstrip('/')
    # 检查路径本身是否已经是checkpoint目录
    basename = os.path.basename(model_path)
    if _re_checkpoint.match(basename):
        # 路径本身就是一个checkpoint目录，直接使用
        print(f"[Using checkpoint path] {model_path}")
    elif os.path.isdir(model_path):
        # 路径是目录但不是checkpoint，尝试查找checkpoint
        last_ckpt = get_last_checkpoint(model_path)
        if last_ckpt is not None:
            print(f"[Auto-Load] Found latest checkpoint: {last_ckpt}")
            model_path = last_ckpt
        else:
            print(f"[Warning] No checkpoint-* found under {model_path}, using folder as model.")
    else:
        print(f"[Using model path] {model_path}")
    print(f"[Final model path] {model_path}")
    
    # 从dataset里提取dataset_name
    if args.dataset_name is None:
        dataset_name = args.dataset.split("/")[-2]
    else:
        dataset_name = args.dataset_name
    
    # 从模型路径中提取相对路径（baseline/qwen2.5-7b/checkpoint-872）
    # 查找 weights/ 或 EnsembleLLM/weights/ 在路径中的位置
    if "weights/" in model_path:
        # 提取 weights/ 之后的部分
        relative_path = model_path.split("weights/", 1)[1]
    elif "EnsembleLLM/weights/" in model_path:
        # 提取 EnsembleLLM/weights/ 之后的部分
        relative_path = model_path.split("/EnsembleLLM/weights/", 1)[1]
    else:
        # 如果找不到weights目录，使用checkpoint名称作为相对路径
        relative_path = basename
    
    # 结果保存在results目录下，保持相对路径结构
    out_dir = os.path.join("results", relative_path, str(args.epoch), dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    args.out_csv = f"{out_dir}/eval_detail.csv"
    args.out_json = f"{out_dir}/eval_summary.json"
    print(f"[Final output directory] {out_dir}")
    print(f"[Final output CSV] {args.out_csv}")
    print(f"[Final output JSON] {args.out_json}")

    evaluate(
        dataset_path=args.dataset,
        model_path=model_path,
        tp=args.tp,
        out_csv=args.out_csv,
        out_json=args.out_json,
        max_model_len=args.max_model_len,
        thinking=args.thinking,
        repeat=args.repeat,
        dataset_name=dataset_name,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
