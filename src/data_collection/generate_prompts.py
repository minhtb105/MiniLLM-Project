import os
import json
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def build_prompts_from_dolly(input_path, output_path, num_samples=2000):
    prompts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            prompts.append({"prompt": item["instruction"]})
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    logging.info(f"Saved {len(prompts)} prompts to {output_path}")

if __name__ == "__main__":
    build_prompts_from_dolly(
        "data/raw/hh_rlhf/dolly/train.jsonl",
        "data/raw/hh_rlhf/prompts/prompts.jsonl"
    )
