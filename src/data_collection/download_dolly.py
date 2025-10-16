import os
import json
from datasets import load_dataset
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def load_dolly(split="train"):
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    return [{"instruction": x["instruction"], "response": x["response"]} for x in ds]

def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} Dolly samples to {path}")
    

if __name__ == "__main__":
    data = load_dolly()
    save_dataset(data, "data/raw/hh_rlhf/dolly/train.jsonl")
