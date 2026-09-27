"""
Converts data/finetune_examples.jsonl into the chat-format JSONL used for
fine-tuning on Colab. See finetune/README_COLAB.md for the full workflow.
"""
import json
from pathlib import Path

RAW_PATH = Path(__file__).parent / "finetune_examples.jsonl"
OUT_PATH = Path(__file__).parent / "finetune_formatted.jsonl"
SYSTEM_PROMPT = "You are a helpful assistant with a specific tone and personality."


def convert():
    if not RAW_PATH.exists():
        print(f"Create {RAW_PATH} first with your examples.")
        return
    examples = []
    with open(RAW_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["response"]},
            ]})
    with open(OUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} formatted examples to {OUT_PATH}")


if __name__ == "__main__":
    convert()
