from datasets import load_dataset
import json

def load_and_convert_amc23(output_path="amc23/test.json"):
    """
    Load AMC 23 test dataset from Hugging Face,
    keep only instruction and answer fields and save as JSON.
    """
    # Load AMC 23 test set (usually 'test' split)
    dataset = load_dataset("math-ai/amc23", split="test")

    print(f"Successfully loaded AMC-23 test set, {len(dataset)} samples")

    # Convert to target format
    simplified_data = [
        {"instruction": item["question"].strip(), "answer": str(item["answer"]).strip()}
        for item in dataset
        if "question" in item and "answer" in item
    ]

    # Save as JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=4)

    print(f"Saved to {output_path}")
    print(f"Example:\n{json.dumps(simplified_data[:2], indent=4, ensure_ascii=False)}")

if __name__ == "__main__":
    load_and_convert_amc23()

