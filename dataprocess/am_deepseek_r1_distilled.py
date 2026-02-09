import os
import json
import math
import re
import zstandard as zstd
from tqdm import tqdm
from math_verify import parse, LatexExtractionConfig, verify
from latex2sympy2_extended import NormalizationConfig

## ==============================
## Configuration section
## ==============================
dataset_dir = "/root/buaa/hf_cache/datasets/AM-DeepSeek-R1-Distilled-1.4M"
## dataset_dir = "/root/buaa/cache/huggingface/datasets--a-m-team--AM-DeepSeek-R1-Distilled-1.4M/snapshots/53531c06634904118a2dcd83961918c4d69d1cdf"

output_file = "am_deepseek_r1_filtered_ad.jsonl"

## Automatically find all .jsonl.zst files
zst_files = sorted(
        [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jsonl.zst")]
    )
## Only use normal small files for testing
## zst_files = [os.path.join(dataset_dir, "am_0.9M_sample_1k.jsonl.zst")]
if not zst_files:
    raise FileNotFoundError(f"No .jsonl.zst files found in {dataset_dir}")

print(f"Found {len(zst_files)} compressed files:")
for f in zst_files:
    print("   -", os.path.basename(f))

## ==============================
## Filter functions
## ==============================
def my_correctness_reward_func(prompts, completions, answers, **kwargs) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, sol in zip(completions, answers):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            ## We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        ## Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            ## Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                reward = float('nan')
        else:
            ## If the gold solution is not parseable, we assign `None` to skip this example
            reward = float('nan')
        rewards.append(reward)
    return rewards

def weak_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    ##HACK: here the $ can match the last \n or not in the response
    pattern = r"^<think>.*?</think>\s*<answer>.*?\\boxed\{.*?\}.*?</answer>$"
    responses = completions
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.25 if match else 0.0 for match in matches]

## ==============================
## Read + merge
## ==============================
def stream_zst_lines(zst_path):
    """Read .zst file line by line with decompression"""
    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            for chunk in iter(lambda: reader.read(65536), b""):
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    yield line
            if buffer.strip():
                yield buffer

converted = []
filtered_count = 0
for zst_path in zst_files:
    print(f"\nProcessing {os.path.basename(zst_path)} ...")
    line_iter = stream_zst_lines(zst_path)
    for line in tqdm(line_iter, desc=f"Decompressing {os.path.basename(zst_path)}"):
        try:
            data = json.loads(line)
        except Exception:
            continue

        msgs = data.get("messages", [])
        if not msgs or len(msgs) < 2:
            continue

        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        assistant_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not user_msg or not assistant_msg:
            continue

        question = user_msg.get("content", "").strip()
        output_content = assistant_msg.get("content", "").strip()

        if not question or not output_content:
            continue

        ## Filter logic: exclude specific content
        if output_content in ['KodCode', 'codeio', 'OpenCoder', 'OpenCoderStage2', None]:
            filtered_count += 1
            continue

        ## Filter logic: check format
        format_score = weak_format_reward_func([output_content])[0]
        if format_score == 0.0:
            filtered_count += 1
            continue

        ## Filter logic: check correctness
        ## Get reference_answer
        reference_answer = None
        if len(msgs) > 0 and 'info' in msgs[0] and 'reference_answer' in msgs[0]['info']:
            reference_answer = msgs[0]['info']['reference_answer']
        
        ## If no reference_answer, skip (cannot verify correctness)
        if not reference_answer:
            filtered_count += 1
            continue
        
        correctness_score = my_correctness_reward_func(
            ['x'], 
            [output_content], 
            [reference_answer]
        )[0]
        if math.isnan(correctness_score) or correctness_score == 0.0:
            filtered_count += 1
            continue

        ## Keep original format, add raw data directly
        converted.append(data)

print(f"\nTotal converted {len(converted)}  training samples (filtered  {filtered_count}  samples)")

## ==============================
## Example
## ==============================
if converted:
    print("\n📋 Example（ 3  samples)：")
    print("=" * 80)
    for i, example in enumerate(converted[:3], 1):
        print(f"\nExample {i}:")
        print(json.dumps(example, ensure_ascii=False, indent=2))
        print("-" * 80)

## ==============================
## 
## ==============================
def save_to_jsonl(data_list, filename):
    """Save as JSONL format (one JSON object per line)"""
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(data_list):
            item['idx'] = idx
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

## Save all data
save_to_jsonl(converted, output_file)
print(f"🎉 Saved to {output_file}（ {len(converted)}  samples)")

## Extract 1000 samples for testing
test_output_file = "am_deepseek_r1_filtered_ad_test_1000.jsonl"
test_samples = converted[:1000]
save_to_jsonl(test_samples, test_output_file)
print(f"🧪 Save test set to {test_output_file}（ {len(test_samples)}  samples)")
