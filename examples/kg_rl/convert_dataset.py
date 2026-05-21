"""One-shot converter: ISP-KGR train.json → slime-friendly JSONL.

Input row schema (ISP-KGR, expected):
    {"question": "...", "answer": ["..."], "mention_entity": [...] | "..."}

Output row schema (slime, used by the run_kg_grpo.sh wiring):
    {"messages": [<sys>, <user>], "answer": [...], "metadata": {"question": "...", "mention_entity": ...}}

Run:
    python convert_dataset.py --input ISP-KGR/datasets/train.json --output slime_train.jsonl
"""

import argparse
import json
from pathlib import Path

from prompts import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to ISP-KGR train.json")
    ap.add_argument("--output", required=True, help="Path to write slime JSONL")
    args = ap.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    assert isinstance(raw, list), f"expected list of rows, got {type(raw)}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in raw:
            question = row["question"]
            mention_entity = row.get("mention_entity", [])
            answer = row.get("answer", [])
            if isinstance(answer, str):
                answer = [answer]

            out_row = {
                "messages": build_messages(question, mention_entity),
                "answer": answer,
                "metadata": {
                    "question": question,
                    "mention_entity": mention_entity,
                },
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} rows to {out_path}")


if __name__ == "__main__":
    main()
