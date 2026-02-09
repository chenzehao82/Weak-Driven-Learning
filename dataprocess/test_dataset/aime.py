from datasets import load_dataset
import json

def load_and_convert_aime2025(output_path="aime2024/test.json"):
    """
    Load AIME 2025 test dataset from Hugging Face,
    keep only instruction and answer fields and save as JSON.
    """
    # Load AIME 2025 test set (usually 'test' split)
    dataset = load_dataset("math-ai/aime24", split="test")

    print(f"Successfully loaded AIME-2025 test set, {len(dataset)} samples")

    # Convert to target format
    simplified_data = [
        {"instruction": item["problem"].strip(), "answer": str(item["answer"]).strip()}
        for item in dataset
        if "problem" in item and "answer" in item
    ]

    # Save as JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=4)

    print(f"Saved to {output_path}")
    print(f"Example:\n{json.dumps(simplified_data[:2], indent=4, ensure_ascii=False)}")

if __name__ == "__main__":
    load_and_convert_aime2025()
