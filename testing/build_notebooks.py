"""Generate the testing entry points from one source, so every path evaluates the same way:

    index.py                     run every arm enabled in config.yml (terminal: python index.py)
    index.ipynb                  the same, as a notebook
    <arm>/index.ipynb            one arm only, with its own config embedded
    refinement-loop/<m>/index.ipynb

evalkit.py, ds1000_runner.py and run_arm.py are embedded verbatim, so on the GPU machine you upload one entry point
(plus config.yml for the two run-all versions). Edit the .py sources, then run:

    python build_notebooks.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS1000_COMMIT = "b39aab71da6d23ef8d3cac59a7c5f834516ab334"
HELPERS = {name: (HERE / name).read_text() for name in ("evalkit.py", "ds1000_runner.py", "run_arm.py")}
CONFIG_TEMPLATE = (HERE / "config.yml").read_text()
ARMS = ["vanilla", "few-shot", "lora", "qlora", "qwen-14b"]

INSTALL = [
    "pip install -q -U vllm datasets huggingface_hub bitsandbytes pyyaml",
    # vLLM may upgrade torch to another CUDA build; the image's torchaudio then fails to load and transformers
    # imports it. Audio is not used, so remove it.
    "pip uninstall -y -q torchaudio",
    "pip install -q uv",
    "uv venv ds1000-env --python 3.10 -q --allow-existing",   # rerunnable: keeps an existing env
    "uv pip install -q --python ds1000-env/bin/python numpy==1.26.4 pandas==1.5.3 scipy==1.12.0 scikit-learn==1.4.0 "
    "matplotlib==3.8.4 seaborn==0.13.2 statsmodels==0.14.1 xgboost==2.0.3 gensim==4.3.2 pyyaml tensorflow-cpu==2.16.1",
    "uv pip install -q --python ds1000-env/bin/python torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu",
    f"curl -sfL https://raw.githubusercontent.com/xlang-ai/DS-1000/{DS1000_COMMIT}/execution.py -o execution.py",
]

MD_INTRO = """
# Testing — all arms

Runs every arm enabled in **`config.yml`** (upload it next to this notebook), one process per arm, and pushes all
results to the private dataset repo `output_repo`. Then **Run All**.

| Track | Data | Output per run |
|---|---|---|
| Structural | 304 `test` instructions of `dimsmauls/code4ml-annotated-rated` (config `training`), 1 greedy answer each | `code4ml_outputs.jsonl` (code + AST validity) — rated later with SonarQube + MLScent |
| Functional | DS-1000, 997 problems (3 few-shot example problems excluded for every arm), 10 sampled answers each | `ds1000_generations.jsonl`, `ds1000_results.jsonl`, `ds1000_per_problem.json` |
| Refinement | 200 DS-1000 problems, up to 3 rounds of execution feedback | `refinement_results.jsonl` |

Every run writes `summary.json`. Runs already on the Hub are skipped, so rerunning resumes. Pilot runs
(`pilot_problems` set) go to `pilot/` and never block the full evaluation.

**Machine:** template PyTorch with Jupyter. GPU ≥ 40 GB (A6000 / L40S / A100) if `qwen-14b` is enabled, otherwise
24 GB is enough. Disk ≥ 90 GB with `qwen-14b`, otherwise 60 GB.
**Credentials:** `HF_TOKEN` (write) and `VAST_API_KEY` (so the machine can stop itself), either as environment
variables in the template or as `hf_token` / `vast_api_key` in `config.yml`.
"""

MD_INSTALL = """
## 1. Install
Generation: vLLM in the main environment. DS-1000 tests: the exact versions of the official `environment.yml`
(Python 3.10, pandas 1.5.3, numpy 1.26.4, torch 2.2.0 CPU, TensorFlow 2.16.1, ...) in a separate `ds1000-env/`
built with `uv`, because they conflict with vLLM. The official `execution.py` comes from DS-1000 at a pinned commit.
"""

MD_HELPERS = """
## 2. Helper files
`evalkit.py` (prompts, official DS-1000 post-processing, pass@k, AST gate), `ds1000_runner.py` (DS-1000 tests in
`ds1000-env`), `run_arm.py` (generation + tests + upload for one arm).
"""

MD_SETUP = """
## 3. Read `config.yml`, check the GPU, log in
"""

MD_RUN = """
## 4. Run the arms
Each arm runs in its own process (fresh GPU memory) and uploads every finished run before the next starts. A failed
arm uploads its traceback to `errors/` and the next arm still runs.
"""

MD_REFINE = """
## 5. Iterative refinement
`seed: auto` picks the median seed of that method by DS-1000 pass@1 (from its `summary.json` files on the Hub).
"""

MD_SHUTDOWN = """
## 6. Shut the machine down
Runs whether everything succeeded or something failed (failures are in `errors/` on the Hub). Colab: releases the
runtime. Vast.ai: `vastai stop instance` (GPU billing stops; only the small disk fee remains) — needs `VAST_API_KEY`.
If you stop a cell yourself, this cell does not run: stop the machine yourself.
"""

SETUP = """
import json, os, subprocess, sys
import yaml
from huggingface_hub import HfApi, hf_hub_download, login

cfg = yaml.safe_load(open("config.yml"))
# Credentials from config.yml fill in only what the environment does not already provide; child processes inherit them.
for key, env in (("hf_token", "HF_TOKEN"), ("vast_api_key", "VAST_API_KEY")):
    if cfg.get(key) and not os.environ.get(env):
        os.environ[env] = str(cfg[key])
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")), "| VAST_API_KEY set:", bool(os.environ.get("VAST_API_KEY")))
arms = [kind for kind, enabled in cfg["arms"].items() if enabled]
refine_methods = [m for m in ("lora", "qlora") if cfg["refinement-loop"].get(m)]
PREFIX = "pilot/" if cfg.get("pilot_problems") else ""
print("arms:", arms, "| refinement:", refine_methods, "| pilot:", cfg.get("pilot_problems"))

# nvidia-smi instead of torch, so this process does not hold GPU memory the arms need
name, total_mb = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True).stdout.strip().split(", ")
print(f"GPU: {name}, {int(total_mb) / 1024:.0f} GB")
if "qwen-14b" in arms and int(total_mb) < 40 * 1024:
    raise SystemExit("qwen-14b needs a GPU with >= 40 GB: disable it in config.yml or rent a bigger GPU")

token = os.environ.get("HF_TOKEN")
login(token=token) if token else login()
api = HfApi()
api.create_repo(cfg["output_repo"], repo_type="dataset", private=True, exist_ok=True)

def stream(cmd):
    \"\"\"Run a command and show its output live in the notebook.\"\"\"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    return process.wait()

failed = []
"""

RUN_ARMS = """
for kind in arms:
    print(f"\\n########## {kind} ##########", flush=True)
    if stream([sys.executable, "run_arm.py", "config.yml", "generate", kind]) != 0:
        failed.append(kind)
print("failed arms:", failed)
"""

REFINE = """
def median_seed(method):
    scores = []
    for seed in (42, 43, 44):
        path = hf_hub_download(cfg["output_repo"], f"{PREFIX}{method}/seed-{seed}/summary.json", repo_type="dataset")
        scores.append((json.load(open(path))["ds1000"]["pass@1"], seed))
    scores.sort()
    print(method, "pass@1 by seed:", {seed: round(score, 4) for score, seed in scores})
    return scores[1][1]

for method in refine_methods:
    print(f"\\n########## refinement-{method} ##########", flush=True)
    try:
        seed_setting = cfg["refinement-loop"]["seed"]
        seed = median_seed(method) if seed_setting == "auto" else int(seed_setting)
    except Exception as exc:
        print(f"cannot pick the seed for {method} (are its results on the Hub?):", exc)
        failed.append(f"refinement-{method}")
        continue
    print("refinement seed:", seed)
    if stream([sys.executable, "run_arm.py", "config.yml", "refine", method, str(seed)]) != 0:
        failed.append(f"refinement-{method}")
print("failed:", failed)
"""

SHUTDOWN = """
def shutdown():
    try:
        from google.colab import runtime  # only importable on Colab
        print("releasing the Colab runtime")
        runtime.unassign()
        return
    except ImportError:
        pass
    container_id, api_key = os.environ.get("CONTAINER_ID"), os.environ.get("VAST_API_KEY")
    if not (container_id and api_key):
        print("CONTAINER_ID or VAST_API_KEY not set: STOP THE INSTANCE YOURSELF on cloud.vast.ai")
        return
    subprocess.run(["pip", "install", "-q", "vastai"], check=True)
    print(f"stopping Vast.ai instance {container_id}")
    subprocess.run(["vastai", "stop", "instance", container_id, "--api-key", api_key], check=True)

print("done; failed:", failed or "none")
print(f"results: https://huggingface.co/datasets/{cfg['output_repo']}")
if cfg.get("auto_shutdown", True):
    shutdown()
else:
    print("auto_shutdown is off: stop the machine yourself")
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def config_for(arms_on, refine_on):
    """config.yml with only the given arms / refinement methods enabled (comments kept)."""
    lines, section = [], None
    for line in CONFIG_TEMPLATE.splitlines():
        if line and not line.startswith((" ", "#")):
            section = line.split(":")[0]
        match = re.match(r"^(  )([\w-]+)(:\s*)(true|false)(.*)$", line)
        if match and section in ("arms", "refinement-loop"):
            key = match.group(2)
            enabled = key in (arms_on if section == "arms" else refine_on)
            line = f"{match.group(1)}{key}{match.group(3)}{'true' if enabled else 'false'}{match.group(5)}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def shared_cells():
    install = "%" + INSTALL[0] + "\n" + "\n".join("!" + cmd for cmd in INSTALL[1:]) + ' && echo "execution.py ok"'
    return [
        md(MD_INSTALL), code(install),
        md(MD_HELPERS), *[code(f"%%writefile {name}\n" + text) for name, text in HELPERS.items()],
        md(MD_SETUP), code(SETUP),
        md(MD_RUN), code(RUN_ARMS),
        md(MD_REFINE), code(REFINE),
        md(MD_SHUTDOWN), code(SHUTDOWN),
    ]


def single_notebook(title, description, config_text):
    intro = f"""
# Testing — {title}

{description}

Only this part is enabled in the embedded `config.yml` below. Set `hf_token` / `vast_api_key` there (or as
environment variables in the template) and `pilot_problems: 20` for a pilot, then **Run All**. Results go to the
private dataset repo `output_repo`; runs already on the Hub are skipped; the machine stops itself at the end.
"""
    return notebook([md(intro), md("## 0. Configuration (edit here)"), code("%%writefile config.yml\n" + config_text),
                     *shared_cells()])


def index_script():
    parts = [
        "#!/usr/bin/env python",
        '"""Run every arm enabled in config.yml (next to this file) and push the results to the Hub.',
        "",
        "Self-contained: installs its dependencies, writes its helper files, runs each arm in its own process, then",
        "stops the machine. Generated by build_notebooks.py; do not edit by hand.",
        "",
        "    python index.py",
        '"""',
        "import os, subprocess, sys",
        "os.chdir(os.path.dirname(os.path.abspath(__file__)))",
        "",
        "# 1. Install",
        f"INSTALL = {INSTALL!r}",
        "for cmd in INSTALL:",
        '    cmd = f"{sys.executable} -m {cmd}" if cmd.startswith("pip ") else cmd',
        "    subprocess.run(cmd, shell=True, check=True)",
        "",
        "# 2. Helper files",
        f"HELPERS = {HELPERS!r}",
        "for name, text in HELPERS.items():",
        "    with open(name, \"w\") as handle:",
        "        handle.write(text)",
        "",
        "# 3. Config, GPU check, login", SETUP, "",
        "# 4. Arms", RUN_ARMS, "",
        "# 5. Iterative refinement", REFINE, "",
        "# 6. Shut down", SHUTDOWN,
    ]
    return "\n".join(parts) + "\n"


SINGLE = {
    "vanilla": ("vanilla", "Qwen2.5-Coder-7B-Instruct without fine-tuning. GPU: 24 GB, disk 60 GB."),
    "few-shot": ("few-shot", "Vanilla with 3 training examples in the prompt. GPU: 24 GB, disk 60 GB."),
    "lora": ("lora", "LoRA adapters, seeds 42, 43, 44. GPU: 24 GB, disk 60 GB."),
    "qlora": ("qlora", "QLoRA adapters on the 4-bit base, seeds 42, 43, 44. GPU: 24 GB, disk 60 GB."),
    "qwen-14b": ("qwen-14b", "Qwen2.5-Coder-14B-Instruct without fine-tuning. GPU: >= 40 GB, disk 90 GB."),
}


def main():
    (HERE / "index.ipynb").write_text(json.dumps(notebook([md(MD_INTRO), *shared_cells()]), indent=1))
    (HERE / "index.py").write_text(index_script())
    print("wrote index.ipynb, index.py")
    for kind, (title, description) in SINGLE.items():
        path = HERE / kind / "index.ipynb"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(single_notebook(title, description, config_for({kind}, set())), indent=1))
        print("wrote", path.relative_to(HERE))
    for method in ("lora", "qlora"):
        path = HERE / "refinement-loop" / method / "index.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        description = (f"Iterative refinement (arm 5, RQ4) with the {method.upper()} adapter. Needs the {method} "
                       f"results on the Hub; `seed: auto` picks its median seed by DS-1000 pass@1. GPU: 24 GB.")
        path.write_text(json.dumps(single_notebook(f"refinement-loop {method}", description,
                                                   config_for(set(), {method})), indent=1))
        print("wrote", path.relative_to(HERE))


if __name__ == "__main__":
    main()
