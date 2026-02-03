from datasets import load_dataset
import json

def load_and_convert_amc23(output_path="amc23/test.json"):
    """
    从 Hugging Face 加载 AMC 23 数据集的测试集，
    只保留 instruction 和 answer 字段并保存为 JSON。
    """
    # 加载 AMC 23 测试集（一般为 'test' split）
    dataset = load_dataset("math-ai/amc23", split="test")

    print(f"✅ 成功加载 AMC-23 测试集，共 {len(dataset)} 条样本")

    # 转换为目标格式
    simplified_data = [
        {"instruction": item["question"].strip(), "answer": str(item["answer"]).strip()}
        for item in dataset
        if "question" in item and "answer" in item
    ]

    # 保存为 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 已保存到 {output_path}")
    print(f"示例：\n{json.dumps(simplified_data[:2], indent=4, ensure_ascii=False)}")

if __name__ == "__main__":
    load_and_convert_amc23()

