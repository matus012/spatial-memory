"""Token budget of the gt7b prompts under the Qwen2.5-VL-7B chat template (tokenizer files only, no weights).
`python -m spatialmem.p1b.tokens [prompts.jsonl]`."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from spatialmem.hssd.vocab import ROOT

REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
DIR = ROOT / "data" / "models" / "qwen25vl_tokenizer"
FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json")


def count(p: dict, tok) -> int:
    msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user_text"]}]
    return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))


if __name__ == "__main__":
    for f in FILES:
        hf_hub_download(REPO, f, local_dir=DIR)
    tok = AutoTokenizer.from_pretrained(DIR)
    ps = [json.loads(ln) for ln in Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/p1b/prompts_gt7b.jsonl").read_text().splitlines()]
    n = [count(p, tok) for p in ps]
    j = max(range(len(n)), key=n.__getitem__)
    print(f"prompts {len(n)}  tokens min {min(n)}  median {median(n)}  max {max(n)}  (longest: {ps[j]['qid']})")
