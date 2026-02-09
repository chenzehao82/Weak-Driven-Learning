import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_math_dataset_jsonl(data_path: str, tokenizer: AutoTokenizer, use_chat_template: bool = None):
    """
    Load JSONL format math dataset and process to specified format
    
    Args:
        data_path: Dataset path, supports .jsonl format
        tokenizer: AutoTokenizer instance
        use_chat_template: Whether to use chat_template for formatting (default None, auto-detect)
                          - None: auto-detect (math_10k doesn't use, others use)
                          - True: for scenarios requiring system prompt (e.g., AM datasets)
                          - False: for already formatted data (e.g., math_10k)
        
    Returns:
        Dataset:  'prompt'  'completion' Dataset
    """
    ## Auto-detect whether to use chat_template
    if use_chat_template is None:
        ## Datasets that don't use chat_template (already formatted simple datasets)
        simple_datasets = ["math_10k", "gsm8k", "svamp", "mawps"]
        use_chat_template = not any(ds in data_path.lower() for ds in simple_datasets)
    
    system_prompt = """ You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> and <answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    data_list = []
    
    print(f'Starting to load math dataset: {data_path}')
    print(f'Use chat_template: {use_chat_template}')
    
    ## Load JSONL data
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                
                ## Detect data format: AM format (messages) or simple format (prompt/completion)
                if "messages" in item:
                    ## AM dataset format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
                    messages = item.get("messages", [])
                    if len(messages) < 2:
                        continue
                    
                    ## Extract user and assistant messages
                    user_msg = next((m for m in messages if m.get("role") == "user"), None)
                    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                    
                    if not user_msg or not assistant_msg:
                        continue
                    
                    raw_prompt = user_msg.get("content", "").strip()
                    ## For code tasks: completion prioritizes info.answer_content (structured field), don't use assistant.content (usually contains <think>）
                    info = user_msg.get("info", {}) if isinstance(user_msg, dict) else {}
                    source = info.get("source", "")
                    code_sources = {"codeio", "OpenCoder", "OpenCoderStage2", "prime"}

                    if source in code_sources:
                        ## answer_content may be in user_msg.info or assistant_msg.info, fallback
                        answer_content = info.get("answer_content")
                        if not answer_content and isinstance(assistant_msg, dict):
                            answer_content = assistant_msg.get("info", {}).get("answer_content")
                        ## Further fallback: if your preprocessing script writes extracted_code, can also be used as training target
                        if not answer_content and "extracted_code" in item:
                            answer_content = item.get("extracted_code")
                        raw_completion = (answer_content or "").strip()
                    else:
                        raw_completion = assistant_msg.get("content", "").strip()
                    
                else:
                    ## Simple format: {"prompt": "...", "completion": "..."}
                    raw_prompt = item.get("prompt", "").strip()
                    raw_completion = item.get("completion", "").strip()
                
                ## Skip empty data
                if not raw_prompt or not raw_completion:
                    continue
                
                ## Decide whether to use chat_template based on parameter
                if use_chat_template:
                    ## Construct user message, use original prompt as question content
                    user_message = {"role": "user", "content": raw_prompt}
                    
                    ## Use apply_chat_template to format prompt, add system prompt
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, user_message],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    ## Use original prompt directly (already formatted data)
                    prompt = raw_prompt
                
                ## Construct completion (original answer + eos_token)
                completion = raw_completion + eos_token
                
                data_list.append({
                    'prompt': prompt,
                    'completion': completion
                })
                
            except json.JSONDecodeError as e:
                print(f"Skip line, JSON parsing error:: {e}")
                continue
            except Exception as e:
                print(f"Skip line, error:: {e}")
                continue
    
    print(f'Successfully processed {len(data_list)} ')
    
    ## 
    if data_list:
        print("\nExample data (first 1 item):")
        print("=" * 80)
        print(f"Prompt (300): {data_list[0]['prompt'][:300]}...")
        print(f"Completion (200): {data_list[0]['completion'][:200]}...")
        print("=" * 80)
    
    return Dataset.from_list(data_list)


def load_code_dataset_jsonl(data_path: str, tokenizer: AutoTokenizer, use_chat_template: bool = False):
    """
    Load JSONL format code dataset and process to prompt/completion
    
    Expected data format (consistent with code.py output):
        {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    Or simple format:
        {"prompt": "...", "completion": "..."}
    """
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    data_list = []

    print(f'Dataset: {data_path}')
    print(f'Use chat_template: {use_chat_template}')

    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())

                if "messages" in item:
                    msgs = item.get("messages", [])
                    if len(msgs) < 2:
                        continue
                    user_msg = next((m for m in msgs if m.get("role") == "user"), None)
                    assistant_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
                    if not user_msg or not assistant_msg:
                        continue
                    raw_prompt = user_msg.get("content", "").strip()
                    raw_completion = assistant_msg.get("content", "").strip()
                else:
                    raw_prompt = item.get("prompt", "").strip()
                    raw_completion = item.get("completion", "").strip()

                if not raw_prompt or not raw_completion:
                    continue

                if use_chat_template:
                    user_message = {"role": "user", "content": raw_prompt}
                    prompt = tokenizer.apply_chat_template(
                        [user_message],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt = raw_prompt

                completion = raw_completion + eos_token
                data_list.append({
                    "prompt": prompt,
                    "completion": completion
                })
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Skip line, error:: {e}")
                continue

    print(f'Successfully processed {len(data_list)} ')

    if data_list:
        print("\nExample data (first 1 item):")
        print("=" * 80)
        print(f"Prompt (300): {data_list[0]['prompt'][:300]}...")
        print(f"Completion (200): {data_list[0]['completion'][:200]}...")
        print("=" * 80)

    return Dataset.from_list(data_list)



