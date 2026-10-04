"""Evaluate one arm, or one refinement run, and push the results to the Hub.

Called by index.ipynb, one process per arm so GPU memory is fully released between models:

    python run_arm.py config.yml generate lora
    python run_arm.py config.yml refine lora 43
    python run_arm.py config.yml retest vanilla    # DS-1000 tests only, on the generations already on the Hub

Exit code 0 = all runs of this arm are on the Hub; 1 = failed (traceback uploaded to OUTPUT_REPO/errors/).
"""
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

import evalkit

BASE_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_14B = "Qwen/Qwen2.5-Coder-14B-Instruct"
ARMS = {
    "vanilla": dict(base=BASE_7B, adapter=None, seeds=[None], quant=False, fewshot=False),
    "few-shot": dict(base=BASE_7B, adapter=None, seeds=[None], quant=False, fewshot=True),
    "lora": dict(base=BASE_7B, adapter="dimsmauls/qwen2.5-coder-7b-code4ml-lora", seeds=[42, 43, 44],
                 quant=False, fewshot=False),
    "qlora": dict(base=BASE_7B, adapter="dimsmauls/qwen2.5-coder-7b-code4ml-qlora", seeds=[42, 43, 44],
                  quant=True, fewshot=False),
    "qwen-14b": dict(base=BASE_14B, adapter=None, seeds=[None], quant=False, fewshot=False),
}


def prefix(cfg):
    """Pilot runs go to pilot/ so they never block or overwrite the full evaluation."""
    return "pilot/" if cfg.get("pilot_problems") else ""


def run_name(cfg, kind, seed):
    return prefix(cfg) + (kind if seed is None else f"{kind}/seed-{seed}")


def write_jsonl(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_data(cfg):
    from datasets import load_dataset

    code4ml = load_dataset(cfg["dataset"], "training", split="train")
    data = {
        "code4ml_test": [row for row in code4ml if row["split"] == "test"],
        "code4ml_train": [row for row in code4ml if row["split"] == "train"],
    }
    problems = {int(p["metadata"]["problem_id"]): p for p in load_dataset("xlangai/DS-1000", split="test")}
    fewshot_ids = evalkit.fewshot_ds1000_ids(len(problems))   # examples; excluded from every arm
    eval_ids = [pid for pid in sorted(problems) if pid not in fewshot_ids]
    if cfg.get("pilot_problems"):
        data["code4ml_test"] = data["code4ml_test"][:cfg["pilot_problems"]]
        eval_ids = eval_ids[:cfg["pilot_problems"]]
    write_jsonl("ds1000_problems.jsonl",
                [{"problem_id": pid, "code_context": p["code_context"]} for pid, p in problems.items()])
    data.update(problems=problems, fewshot_ids=fewshot_ids, eval_ids=eval_ids,
                library_of={pid: p["metadata"]["library"] for pid, p in problems.items()})
    return data


class Engine:
    """One vLLM engine per process; LoRA adapters are attached per request."""

    def __init__(self, arm, gen):
        from vllm import LLM

        import torch
        import vllm

        self.arm = arm
        self.llm = LLM(model=arm["base"], dtype="bfloat16", max_model_len=gen["max_model_len"],
                       gpu_memory_utilization=0.90, seed=gen["seed"], enable_lora=arm["adapter"] is not None,
                       max_lora_rank=16, **({"quantization": "bitsandbytes"} if arm["quant"] else {}))
        self.tokenizer = self.llm.get_tokenizer()
        self.gpu = torch.cuda.get_device_name(0)
        self.versions = {"vllm": vllm.__version__, "torch": torch.__version__}

    def prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def lora(self, seed):
        if self.arm["adapter"] is None:
            return None
        from huggingface_hub import snapshot_download
        from vllm.lora.request import LoRARequest

        local = snapshot_download(self.arm["adapter"], allow_patterns=[f"seed-{seed}/*"])
        return LoRARequest(f"seed-{seed}", seed, f"{local}/seed-{seed}")

    def generate(self, prompts, request, **sampling):
        from vllm import SamplingParams

        return self.llm.generate(prompts, SamplingParams(**sampling), lora_request=request)


def run_ds1000_tests(answers, tag):
    answers_path, results_path = f"answers_{tag}.jsonl", f"results_{tag}.jsonl"
    write_jsonl(answers_path, [{k: row[k] for k in ("problem_id", "sample", "code")} for row in answers])
    subprocess.run(["ds1000-env/bin/python", "ds1000_runner.py", "ds1000_problems.jsonl",
                    answers_path, results_path], check=True)
    with open(results_path) as handle:
        return [json.loads(line) for line in handle]


def evaluate(engine, data, cfg, api, kind, seed):
    arm, gen = ARMS[kind], cfg["generation"]
    name = run_name(cfg, kind, seed)
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    request = engine.lora(seed)
    started = time.time()

    # Structural track: one greedy answer per Code4ML test instruction.
    shots = evalkit.select_fewshot(data["code4ml_train"]) if arm["fewshot"] else []
    prompts = [engine.prompt(evalkit.code4ml_messages(row["instruction"], shots)) for row in data["code4ml_test"]]
    outputs = engine.generate(prompts, request, temperature=0.0, max_tokens=gen["max_new_tokens"])
    code4ml_rows = []
    for row, output in zip(data["code4ml_test"], outputs):
        completion = output.outputs[0].text
        code = evalkit.extract_code(completion)
        code4ml_rows.append({"id": row["id"], "graph_vertex_class": row["graph_vertex_class"],
                             "graph_vertex_subclass": row["graph_vertex_subclass"],
                             "instruction": row["instruction"], "completion": completion,
                             "code": code, "ast_valid": evalkit.ast_valid(code)})
    write_jsonl(out / "code4ml_outputs.jsonl", code4ml_rows)

    # Functional track: sampled answers per DS-1000 problem.
    problems, eval_ids = data["problems"], data["eval_ids"]
    ds_shots = [problems[pid] for pid in data["fewshot_ids"]] if arm["fewshot"] else []
    prompts = [engine.prompt(evalkit.ds1000_messages(problems[pid]["prompt"], ds_shots)) for pid in eval_ids]
    outputs = engine.generate(prompts, request, n=gen["ds1000_samples"], temperature=gen["ds1000_temperature"],
                              top_p=gen["ds1000_top_p"], max_tokens=gen["max_new_tokens"], seed=gen["seed"])
    generations = [{"problem_id": pid, "sample": i, "completion": o.text, "code": evalkit.postprocess_ds1000(o.text)}
                   for pid, output in zip(eval_ids, outputs) for i, o in enumerate(output.outputs)]
    write_jsonl(out / "ds1000_generations.jsonl", generations)
    generation_minutes = round((time.time() - started) / 60, 1)

    results = run_ds1000_tests(generations, name.replace("/", "_"))
    write_jsonl(out / "ds1000_results.jsonl", results)
    ds_summary = evalkit.summarize_ds1000(results, data["library_of"])
    with open(out / "ds1000_per_problem.json", "w") as handle:
        json.dump(ds_summary.pop("per_problem"), handle)

    valid = sum(r["ast_valid"] for r in code4ml_rows)
    summary = {
        "run": name, "base_model": arm["base"], "adapter_repo": arm["adapter"], "seed": seed,
        "quantized_4bit": arm["quant"], "fewshot": arm["fewshot"], "fewshot_ds1000_ids": data["fewshot_ids"],
        "gpu": engine.gpu, "versions": engine.versions, "pilot_problems": cfg.get("pilot_problems"),
        "generation": {**gen, "code4ml": "greedy"},
        "code4ml": {"answers": len(code4ml_rows), "ast_valid": valid, "ast_valid_rate": valid / len(code4ml_rows)},
        "ds1000": ds_summary,
        "generation_minutes": generation_minutes,
        "total_minutes": round((time.time() - started) / 60, 1),
    }
    with open(out / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    api.upload_folder(repo_id=cfg["output_repo"], repo_type="dataset", folder_path=str(out),
                      path_in_repo=name, commit_message=f"evaluation {name}")
    print(f"{name}: AST valid {valid}/{len(code4ml_rows)} | DS-1000 pass@1 {ds_summary['pass@1']:.4f} "
          f"pass@10 {ds_summary['pass@10']:.4f} | {summary['total_minutes']} min", flush=True)


def retest(data, cfg, api, kind, seed):
    """Rerun the DS-1000 tests of a finished run on its uploaded generations (no generation, no GPU).

    Replaces ds1000_results.jsonl, ds1000_per_problem.json and the ds1000 part of summary.json; the Code4ML outputs and
    the generations stay as they are. The previous scores are kept in summary.json under "retested"."""
    from huggingface_hub import hf_hub_download

    name = run_name(cfg, kind, seed)
    fetch = lambda file: hf_hub_download(cfg["output_repo"], f"{name}/{file}", repo_type="dataset")
    with open(fetch("summary.json")) as handle:
        summary = json.load(handle)
    if "retested" in summary:
        print(f"{name}: already retested, skipping", flush=True)
        return
    with open(fetch("ds1000_generations.jsonl")) as handle:
        generations = [json.loads(line) for line in handle]
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    results = run_ds1000_tests(generations, name.replace("/", "_"))
    write_jsonl(out / "ds1000_results.jsonl", results)
    ds_summary = evalkit.summarize_ds1000(results, data["library_of"])
    with open(out / "ds1000_per_problem.json", "w") as handle:
        json.dump(ds_summary.pop("per_problem"), handle)

    previous = summary["ds1000"]
    summary["ds1000"] = ds_summary
    summary["retested"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "minutes": round((time.time() - started) / 60, 1),
                           "previous_pass@1": previous["pass@1"], "previous_pass@10": previous["pass@10"]}
    with open(out / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    api.upload_folder(repo_id=cfg["output_repo"], repo_type="dataset", folder_path=str(out),
                      path_in_repo=name, commit_message=f"retest {name}")
    print(f"{name}: DS-1000 pass@1 {previous['pass@1']:.4f} -> {ds_summary['pass@1']:.4f} | "
          f"pass@10 {previous['pass@10']:.4f} -> {ds_summary['pass@10']:.4f}", flush=True)


def refine(engine, data, cfg, api, method, seed):
    ref, gen = cfg["refinement-loop"], cfg["generation"]
    name = run_name(cfg, f"refinement-{method}", seed)
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    request = engine.lora(seed)
    subset = sorted(random.Random(42).sample(data["eval_ids"], min(ref["problems"], len(data["eval_ids"]))))
    started = time.time()

    histories = {pid: evalkit.ds1000_messages(data["problems"][pid]["prompt"]) for pid in subset}
    open_ids, rows = list(subset), []
    for iteration in range(ref["max_iterations"] + 1):
        outputs = engine.generate([engine.prompt(histories[pid]) for pid in open_ids], request,
                                  temperature=0.0, max_tokens=gen["max_new_tokens"])
        answers = [{"problem_id": pid, "sample": 0, "completion": o.outputs[0].text,
                    "code": evalkit.postprocess_ds1000(o.outputs[0].text)} for pid, o in zip(open_ids, outputs)]
        results = {r["problem_id"]: r for r in run_ds1000_tests(answers, f"refine_{method}_it{iteration}")}
        still_open = []
        for answer in answers:
            pid, result = answer["problem_id"], results[answer["problem_id"]]
            rows.append({"problem_id": pid, "iteration": iteration, "passed": result["passed"],
                         "result": result["result"], "completion": answer["completion"], "code": answer["code"]})
            if not result["passed"]:
                histories[pid] = evalkit.refine_messages(histories[pid], answer["completion"],
                                                         evalkit.describe_error(result["result"]))
                still_open.append(pid)
        print(f"iteration {iteration}: solved {len(subset) - len(still_open)}/{len(subset)}", flush=True)
        open_ids = still_open
        if not open_ids:
            break

    write_jsonl(out / "refinement_results.jsonl", rows)
    first_pass = {}
    for row in rows:
        if row["passed"]:
            first_pass.setdefault(row["problem_id"], row["iteration"])
    cumulative = [sum(1 for it in first_pass.values() if it <= i) / len(subset)
                  for i in range(ref["max_iterations"] + 1)]
    summary = {
        "run": name, "method": method, "seed": seed, "adapter_repo": ARMS[method]["adapter"],
        "quantized_4bit": ARMS[method]["quant"], "problems": len(subset), "problem_ids": subset,
        "max_iterations": ref["max_iterations"], "pass_rate_by_iteration": cumulative, "decoding": "greedy",
        "gpu": engine.gpu, "versions": engine.versions, "minutes": round((time.time() - started) / 60, 1),
    }
    with open(out / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    api.upload_folder(repo_id=cfg["output_repo"], repo_type="dataset", folder_path=str(out),
                      path_in_repo=name, commit_message=f"refinement {name}")
    print(f"{name}: pass rate by iteration {[round(x, 4) for x in cumulative]}", flush=True)


def main(argv):
    config_path, mode, kind = argv[:3]
    cfg = yaml.safe_load(open(config_path))
    from huggingface_hub import HfApi

    api = HfApi()
    label = {"generate": kind, "refine": f"refinement-{kind}", "retest": f"retest-{kind}"}[mode]
    try:
        if mode == "retest":
            data = load_data(cfg)
            for seed in ARMS[kind]["seeds"]:
                retest(data, cfg, api, kind, seed)
            return 0
        on_hub = set(api.list_repo_files(cfg["output_repo"], repo_type="dataset"))
        if mode == "generate":
            todo = [s for s in ARMS[kind]["seeds"] if f"{run_name(cfg, kind, s)}/summary.json" not in on_hub]
            if not todo:
                print(f"{kind}: already on the Hub, skipping", flush=True)
                return 0
        else:
            seed = int(argv[3])
            if f"{run_name(cfg, f'refinement-{kind}', seed)}/summary.json" in on_hub:
                print(f"{run_name(cfg, f'refinement-{kind}', seed)}: already on the Hub, skipping", flush=True)
                return 0
        data = load_data(cfg)
        engine = Engine(ARMS[kind], cfg["generation"])
        if mode == "generate":
            for seed in todo:
                evaluate(engine, data, cfg, api, kind, seed)
        else:
            refine(engine, data, cfg, api, kind, seed)
        return 0
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            api.upload_file(path_or_fileobj=error.encode(), path_in_repo=f"errors/{label}-{stamp}.txt",
                            repo_id=cfg["output_repo"], repo_type="dataset", commit_message=f"{label} error {stamp}")
        except Exception as upload_exc:  # the exit code still reports the failure
            print("could not upload the error log:", upload_exc, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
