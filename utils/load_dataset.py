import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_math_dataset_jsonl(data_path: str, tokenizer: AutoTokenizer, use_chat_template: bool = None):
    """
    加载 JSONL 格式的数学数据集并处理成指定格式
    
    Args:
        data_path: 数据集路径，支持 .jsonl 格式
        tokenizer: AutoTokenizer 实例
        use_chat_template: 是否使用 chat_template 格式化（默认 None，自动判断）
                          - None: 自动判断（math_10k 不使用，其他使用）
                          - True: 适用于需要系统提示词的场景（如 AM 数据集）
                          - False: 适用于已经格式化好的数据（如 math_10k）
        
    Returns:
        Dataset: 包含 'prompt' 和 'completion' 字段的数据集
    """
    # 自动判断是否使用 chat_template
    if use_chat_template is None:
        # 不使用 chat_template 的数据集（已经格式化好的简单数据集）
        simple_datasets = ["math_10k", "gsm8k", "svamp", "mawps"]
        use_chat_template = not any(ds in data_path.lower() for ds in simple_datasets)
    
    system_prompt = """ You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> and <answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    data_list = []
    
    print(f'开始加载数学数据集: {data_path}')
    print(f'使用 chat_template: {use_chat_template}')
    
    # 加载 JSONL 数据
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                
                # 检测数据格式：AM格式 (messages) 或 简单格式 (prompt/completion)
                if "messages" in item:
                    # AM 数据集格式: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
                    messages = item.get("messages", [])
                    if len(messages) < 2:
                        continue
                    
                    # 提取 user 和 assistant 消息
                    user_msg = next((m for m in messages if m.get("role") == "user"), None)
                    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                    
                    if not user_msg or not assistant_msg:
                        continue
                    
                    raw_prompt = user_msg.get("content", "").strip()
                    # 对 code 任务：completion 优先使用 info.answer_content（结构化字段），不要用 assistant.content（通常含 <think>）
                    info = user_msg.get("info", {}) if isinstance(user_msg, dict) else {}
                    source = info.get("source", "")
                    code_sources = {"codeio", "OpenCoder", "OpenCoderStage2", "prime"}

                    if source in code_sources:
                        # answer_content 可能在 user_msg.info 或 assistant_msg.info，做兜底
                        answer_content = info.get("answer_content")
                        if not answer_content and isinstance(assistant_msg, dict):
                            answer_content = assistant_msg.get("info", {}).get("answer_content")
                        # 进一步兜底：如果你的预处理脚本写入了 extracted_code，也可作为训练目标
                        if not answer_content and "extracted_code" in item:
                            answer_content = item.get("extracted_code")
                        raw_completion = (answer_content or "").strip()
                    else:
                        raw_completion = assistant_msg.get("content", "").strip()
                    
                else:
                    # 简单格式: {"prompt": "...", "completion": "..."}
                    raw_prompt = item.get("prompt", "").strip()
                    raw_completion = item.get("completion", "").strip()
                
                # 跳过空数据
                if not raw_prompt or not raw_completion:
                    continue
                
                # 根据参数决定是否使用 chat_template
                if use_chat_template:
                    # 构造 user message，使用原始 prompt 作为问题内容
                    user_message = {"role": "user", "content": raw_prompt}
                    
                    # 使用 apply_chat_template 格式化 prompt，添加 system prompt
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt}, user_message],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # 直接使用原始 prompt（已经格式化好的数据）
                    prompt = raw_prompt
                
                # 构造 completion（原始答案 + eos_token）
                completion = raw_completion + eos_token
                
                data_list.append({
                    'prompt': prompt,
                    'completion': completion
                })
                
            except json.JSONDecodeError as e:
                print(f"跳过第 {idx} 行，JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"跳过第 {idx} 行，错误: {e}")
                continue
    
    print(f'成功处理 {len(data_list)} 条数据')
    
    # 打印第一条数据示例
    if data_list:
        print("\n示例数据（第 1 条）：")
        print("=" * 80)
        print(f"Prompt (前300字符): {data_list[0]['prompt'][:300]}...")
        print(f"Completion (前200字符): {data_list[0]['completion'][:200]}...")
        print("=" * 80)
    
    return Dataset.from_list(data_list)


def load_code_dataset_jsonl(data_path: str, tokenizer: AutoTokenizer, use_chat_template: bool = False):
    """
    加载 JSONL 格式的代码数据集并处理成 prompt/completion
    
    预期数据格式（与 code.py 输出一致）：
        {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    或简洁格式：
        {"prompt": "...", "completion": "..."}
    """
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    data_list = []

    print(f'开始加载代码数据集: {data_path}')
    print(f'使用 chat_template: {use_chat_template}')

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
                print(f"跳过第 {idx} 行，错误: {e}")
                continue

    print(f'成功处理 {len(data_list)} 条数据')

    if data_list:
        print("\n示例数据（第 1 条）：")
        print("=" * 80)
        print(f"Prompt (前300字符): {data_list[0]['prompt'][:300]}...")
        print(f"Completion (前200字符): {data_list[0]['completion'][:200]}...")
        print("=" * 80)

    return Dataset.from_list(data_list)



