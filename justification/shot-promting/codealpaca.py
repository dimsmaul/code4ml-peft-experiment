"""Justification POC 1: prompting shots on general programming. Qwen2.5-Coder-7B-Instruct without (vanilla) and with
the LoRA adapter (seed 42) answers CodeAlpaca-20k instructions with 1 and 3 examples in the prompt; each answer is scored with ROUGE-L against the
reference, as in Nguyen et al. (IJECE 2026).

    python codealpaca.py config.yml qwen-7b

One vLLM engine per base model, adapters attached per request. Every run (<model>-<k>shot) uploads outputs.jsonl and
summary.json to <output_repo>/<output_prefix>/<run>/; runs already on the Hub are skipped. A run whose replies are
broken (raw Ġ/Ċ or empty) stops the script before more GPU time is spent. Exit code != 0 on failure (traceback in
<output_repo>/errors/).
"""
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

DATASET = "sahil2801/CodeAlpaca-20k"
SPLIT_SEED = 42
SHOTS = (1, 3)   # 1 shot = the first of the 3 examples
MODELS = {   # base model, {variant: (adapter repo, folder) or None}
    "qwen-7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", {
        "vanilla": None,   # baseline: the same base without the adapter
        "lora-seed-42": ("dimsmauls/qwen2.5-coder-7b-code4ml-lora", "seed-42"),
    }),
}


def write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_split(test_size):
    """Fixed shuffle: the first test_size rows are the test set, the next 3 the prompt examples."""
    from datasets import load_dataset

    rows = list(load_dataset(DATASET, split="train"))
    order = list(range(len(rows)))
    random.Random(SPLIT_SEED).shuffle(order)
    test = [{"idx": i, **rows[i]} for i in order[:test_size]]
    shots = [{"idx": i, **rows[i]} for i in order[test_size:test_size + max(SHOTS)]]
    return test, shots


def user_text(row):
    return row["instruction"] + (f"\n\n{row['input']}" if row["input"].strip() else "")


def messages(row, shots):
    out = []
    for shot in shots:
        out += [{"role": "user", "content": user_text(shot)}, {"role": "assistant", "content": shot["output"]}]
    return out + [{"role": "user", "content": user_text(row)}]


def check_replies(completions, name):
    garbled = sum("Ġ" in c or "Ċ" in c for c in completions)
    empty = sum(not c.strip() for c in completions)
    if garbled > 0.01 * len(completions) or empty > 0.1 * len(completions):
        raise RuntimeError(f"{name}: {garbled}/{len(completions)} replies with raw Ġ/Ċ, {empty} empty; stopping")


def run_model(cfg, api, model_key):
    import torch
    import vllm
    from huggingface_hub import snapshot_download
    from rouge_score import rouge_scorer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    base, variants = MODELS[model_key]
    prefix = cfg["output_prefix"]
    on_hub = set(api.list_repo_files(cfg["output_repo"], repo_type="dataset"))
    todo = [(v, k) for v in variants for k in SHOTS if f"{prefix}/{v}-{k}shot/summary.json" not in on_hub]
    if not todo:
        print(f"{model_key}: every run already on the Hub, skipping", flush=True)
        return
    test, shot_pool = load_split(cfg["test_size"])
    has_adapter = any(variants[v] for v, _ in todo)
    llm = LLM(model=base, dtype="bfloat16", max_model_len=cfg["max_model_len"], gpu_memory_utilization=0.90,
              seed=0, enable_lora=has_adapter)
    tokenizer = llm.get_tokenizer()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    lora_ids = {v: i + 1 for i, v in enumerate(variants)}   # unique id per adapter: vLLM caches adapters by id

    for variant, k in todo:
        name = f"{variant}-{k}shot"
        started = time.time()
        request = None
        if variants[variant]:
            repo, folder = variants[variant]
            # adapter_* only, so vLLM keeps the base model's tokenizer
            local = snapshot_download(repo, allow_patterns=[f"{folder}/adapter_*"])
            request = LoRARequest(variant, lora_ids[variant], f"{local}/{folder}")
        prompts = [tokenizer.apply_chat_template(messages(row, shot_pool[:k]), tokenize=False,
                                                 add_generation_prompt=True) for row in test]
        outputs = llm.generate(prompts, sampling, lora_request=request)
        completions = [o.outputs[0].text for o in outputs]
        check_replies(completions, name)
        rows = [{"idx": row["idx"], "instruction": row["instruction"], "input": row["input"],
                 "reference": row["output"], "completion": c,
                 "rouge_l": scorer.score(row["output"], c)["rougeL"].fmeasure}
                for row, c in zip(test, completions)]
        out = Path("results") / prefix / name
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "outputs.jsonl", rows)
        summary = {
            "run": name, "base_model": base, "adapter": variants[variant], "shots": k,
            "dataset": DATASET, "split_seed": SPLIT_SEED, "test_size": len(rows),
            "shot_idx": [s["idx"] for s in shot_pool[:k]], "rouge_l": sum(r["rouge_l"] for r in rows) / len(rows),
            "rouge": "rouge_score RougeScorer(['rougeL'], use_stemmer=True), F1, whole reply vs reference",
            "generation": {"temperature": 0.0, "max_new_tokens": cfg["max_new_tokens"]},
            "gpu": torch.cuda.get_device_name(0), "versions": {"vllm": vllm.__version__, "torch": torch.__version__},
            "minutes": round((time.time() - started) / 60, 1),
        }
        with open(out / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        api.upload_folder(repo_id=cfg["output_repo"], repo_type="dataset", folder_path=str(out),
                          path_in_repo=f"{prefix}/{name}", commit_message=f"codealpaca {name}")
        print(f"{name}: ROUGE-L {summary['rouge_l']:.4f} | {summary['minutes']} min", flush=True)


def main(argv):
    config_path, model_key = argv[:2]
    cfg = yaml.safe_load(open(config_path))
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        run_model(cfg, api, model_key)
        return 0
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            api.upload_file(path_or_fileobj=error.encode(), path_in_repo=f"errors/codealpaca-{model_key}-{stamp}.txt",
                            repo_id=cfg["output_repo"], repo_type="dataset", commit_message=f"codealpaca error {stamp}")
        except Exception as upload_exc:
            print("could not upload the error log:", upload_exc, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
