"""P1b planner runner (task 008, docs/P1B_PROTOCOL.md §4/§6). stdlib + vllm only; runs inside vllm.sif on PERUN.

usage: python3 vllm_run.py <model_dir> <prompts.jsonl> <out.jsonl> <tp>
in : one JSON per line, keys exactly {qid, row, system, user_text, images}; images = absolute jpg paths ([] = text only)
out: one JSON per line {qid, row, model, raw_output, n_prompt_tokens, finish_reason}, input order; <out>.run_meta.json
Idempotent: exits if <out> has one line per prompt and <out>.run_meta.json exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

KEYS = {"qid", "row", "system", "user_text", "images"}
MAX_MODEL_LEN = 32768  # Qwen2.5-VL native context (config sliding_window/README: 32,768; 128k needs YaRN, not enabled)
MAX_IMAGES = 8
MAX_PIXELS = 1280 * 28 * 28  # <= 1280 visual tokens per image -> 8 images <= 10,240 tokens; 640x480 frames stay native
SEED = 114
MEDIA_ROOT = os.environ["PROJECT_DIR"]   # the cluster project dir (README, "Cluster")


def gpu_mem_poller(stop: threading.Event, peak: list[int]) -> None:
    """Peak of summed nvidia-smi memory.used (MiB) over this job's GPUs, sampled every 1 s."""
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        peak[0] = max(peak[0], sum(int(x) for x in out.split()))
        stop.wait(1.0)


def message(p: dict) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": "file://" + im}} for im in p["images"]]
    content.append({"type": "text", "text": p["user_text"]})
    return [{"role": "system", "content": p["system"]}, {"role": "user", "content": content}]


def main(model_dir: str, prompts_path: str, out_path: str, tp: int) -> None:
    raw = Path(prompts_path).read_bytes()
    prompts = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    for p in prompts:
        assert set(p) == KEYS, f"prompt keys {sorted(p)} != {sorted(KEYS)}"
        assert len(p["images"]) <= MAX_IMAGES and all(Path(im).is_file() for im in p["images"]), p["qid"]
    out, meta_path = Path(out_path), Path(out_path + ".run_meta.json")
    if out.exists() and meta_path.exists() and len(out.read_text().splitlines()) == len(prompts):
        print(f"{out} complete ({len(prompts)} lines), skipping")
        return

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    stop, peak = threading.Event(), [0]
    poller = threading.Thread(target=gpu_mem_poller, args=(stop, peak), daemon=True)
    poller.start()
    t0 = time.time()
    llm = LLM(model=model_dir, tensor_parallel_size=tp, max_model_len=MAX_MODEL_LEN, seed=SEED, dtype="bfloat16",
              limit_mm_per_prompt={"image": MAX_IMAGES}, mm_processor_kwargs={"max_pixels": MAX_PIXELS},
              allowed_local_media_path=MEDIA_ROOT)
    t_load = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=SEED)
    t1 = time.time()
    res = llm.chat([message(p) for p in prompts], sp, use_tqdm=False)
    t_gen = time.time() - t1
    stop.set()
    poller.join()

    model_name = Path(model_dir).name
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w") as f:
        for p, r in zip(prompts, res, strict=True):
            f.write(json.dumps({"qid": p["qid"], "row": p["row"], "model": model_name, "raw_output": r.outputs[0].text,
                                "n_prompt_tokens": len(r.prompt_token_ids),
                                "finish_reason": r.outputs[0].finish_reason}) + "\n")
    rev = Path(model_dir, "REVISION")
    meta = {
        "model_path": model_dir, "model_revision": rev.read_text().strip() if rev.exists() else None,
        "vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "tp": tp, "max_model_len": MAX_MODEL_LEN,
        "limit_mm_per_prompt": MAX_IMAGES, "max_pixels": MAX_PIXELS,
        "sampling": {"temperature": sp.temperature, "max_tokens": sp.max_tokens, "seed": sp.seed},
        "prompts_file": prompts_path, "prompts_sha256": hashlib.sha256(raw).hexdigest(), "n_prompts": len(prompts),
        "n_prompt_tokens_max": max(len(r.prompt_token_ids) for r in res),
        "load_s": round(t_load, 1), "generate_s": round(t_gen, 1), "wall_s": round(time.time() - t0, 1),
        "peak_gpu_mem_mib_sum": peak[0], "gpu_name": torch.cuda.get_device_name(0),
        "n_gpus_visible": torch.cuda.device_count(), "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    tmp.replace(out)
    print(json.dumps(meta))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
