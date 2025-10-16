import json
import logging
import os, sys
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def load_hh_rlhf(split="train"):
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    
    return [{"prompt": x["chosen"], "rejected": x["rejected"]} for x in ds]

def save_dataset_to_jsonl(data, save_path: str):
    """
    Save dataset to JSONL file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    # Path to save dataset
    train_save_path = os.path.join("data", "raw", "hh_rlhf", "train.jsonl")

    # Load dataset
    logging.info("⬇Downloading Anthropic HH-RLHF dataset...")
    train_data = load_hh_rlhf(split="train")

    # Save locally
    save_dataset_to_jsonl(train_data, train_save_path)
    
    test_save_path = os.path.join("data", "raw", "hh_rlhf", "test.jsonl")
    test_data = load_hh_rlhf(split="test")
    save_dataset_to_jsonl(test_data, test_save_path)

if __name__ == "__main__":
    main()                
