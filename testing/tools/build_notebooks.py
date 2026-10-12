"""Generate the testing entry points from one source, so every path evaluates the same way:

    index.py                                run every arm enabled in config.yml (terminal: python index.py)
    index.ipynb                             the same, as a notebook
    arms/<arm>/index.ipynb                  one arm only, with its own config embedded
    refinement-loop/execution/<m>/index.ipynb
    refinement-loop/sonar/index.ipynb       refinement with execution + SonarQube feedback

src/evalkit.py, src/ds1000_runner.py and src/run_arm.py are embedded verbatim, so on the GPU machine you upload one entry
point (plus config.yml for the two run-all versions). index.py and index.ipynb work inside src/, so the testing root keeps
only the entry points and config.yml. Edit the .py sources in src/, then run from testing/:

    python tools/build_notebooks.py
"""
import json
import re
from pathlib import Path

TESTING = Path(__file__).resolve().parents[1]
SRC = TESTING / "src"
DS1000_COMMIT = "b39aab71da6d23ef8d3cac59a7c5f834516ab334"
HELPERS = {name: (SRC / name).read_text() for name in ("evalkit.py", "ds1000_runner.py", "run_arm.py")}
CONFIG_TEMPLATE = (TESTING / "config.yml").read_text()
INDEX_CONFIG = "../config.yml"   # index.py / index.ipynb run inside src/; config.yml stays next to them
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

INTO_SRC = """
# Work inside src/: helper files, ds1000-env, execution.py and results/ stay out of the testing root
import os
if os.path.basename(os.getcwd()) != "src":
    os.makedirs("src", exist_ok=True)
    os.chdir("src")
print("working directory:", os.getcwd())
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

MD_RETEST = """
## 4b. Retest (optional)
Arms enabled under `retest:` take the code again from the replies already on the Hub (first python block, see
`evalkit.python_block`), recompute AST validity and rerun the DS-1000 tests; no generation, no GPU. Old scores are kept
in `summary.json` under `retested`. Runs generated with the current extraction are skipped.
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
retest_arms = [kind for kind, enabled in cfg.get("retest", {}).items() if enabled]
PREFIX = "pilot/" if cfg.get("pilot_problems") else ""
print("arms:", arms, "| refinement:", refine_methods, "| retest:", retest_arms, "| pilot:", cfg.get("pilot_problems"))

if arms or refine_methods:  # retest alone runs on CPU
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

RETEST = """
for kind in retest_arms:
    print(f"\\n########## retest {kind} ##########", flush=True)
    if stream([sys.executable, "run_arm.py", "config.yml", "retest", kind]) != 0:
        failed.append(f"retest-{kind}")
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
        if match and section in ("arms", "refinement-loop", "retest"):
            key = match.group(2)
            enabled = key in {"arms": arms_on, "refinement-loop": refine_on}.get(section, ())
            line = f"{match.group(1)}{key}{match.group(3)}{'true' if enabled else 'false'}{match.group(5)}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def install_cell():
    """pip lines as %pip so they act on the kernel's environment (a bare !pip may be another Python on the PATH)."""
    lines = [("%" if cmd.startswith("pip ") else "!") + cmd for cmd in INSTALL]
    return "\n".join(lines) + ' && echo "execution.py ok"'


def with_config(text, path):
    """The same cell code, reading config.yml from another path."""
    return text.replace('"config.yml"', f'"{path}"')


def shared_cells(config="config.yml"):
    install = install_cell()
    return [
        md(MD_INSTALL), code(install),
        md(MD_HELPERS), *[code(f"%%writefile {name}\n" + text) for name, text in HELPERS.items()],
        md(MD_SETUP), code(with_config(SETUP, config)),
        md(MD_RUN), code(with_config(RUN_ARMS, config)),
        md(MD_RETEST), code(with_config(RETEST, config)),
        md(MD_REFINE), code(with_config(REFINE, config)),
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
        "stops the machine. Works inside src/ (helpers, ds1000-env, results/). Generated by tools/build_notebooks.py;",
        "do not edit by hand.",
        "",
        "    python index.py",
        '"""',
        "import os, subprocess, sys",
        "SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), \"src\")",
        "os.makedirs(SRC, exist_ok=True)",
        "os.chdir(SRC)",
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
        "    if os.path.exists(name):   # cloned repo: keep the source file from git, the embedded copy may be older",
        "        continue",
        "    with open(name, \"w\") as handle:",
        "        handle.write(text)",
        "",
        "# 3. Config, GPU check, login", with_config(SETUP, INDEX_CONFIG), "",
        "# 4. Arms", with_config(RUN_ARMS, INDEX_CONFIG), "",
        "# 4b. Retest DS-1000", with_config(RETEST, INDEX_CONFIG), "",
        "# 5. Iterative refinement", with_config(REFINE, INDEX_CONFIG), "",
        "# 6. Shut down", SHUTDOWN,
    ]
    return "\n".join(parts) + "\n"


SINGLE = {
    "vanilla": ("vanilla", "Qwen2.5-Coder-7B-Instruct without fine-tuning. GPU: 24 GB, disk 60 GB."),
    "few-shot": ("few-shot", "Vanilla with 3 training examples in the prompt. GPU: 24 GB, disk 60 GB."),
    "lora": ("lora", "LoRA adapters, seeds 42, 43, 44. GPU: 24 GB, disk 60 GB."),
    "qlora": ("qlora", "QLoRA adapters (trained on a 4-bit base, evaluated on the bf16 base), seeds 42, 43, 44. GPU: 24 GB, disk 60 GB."),
    "qwen-14b": ("qwen-14b", "Qwen2.5-Coder-14B-Instruct without fine-tuning. GPU: >= 40 GB, disk 90 GB."),
}


# ---------------------------------------------------------------------------------------------------------------------
# refinement-loop/sonar/index.ipynb: refinement with execution + SonarQube feedback, SonarQube installed on the GPU
# machine itself (same versions as the rating service: SonarQube Community Build 26.6.0.123539, sonar-scanner 6.2.1.4610,
# Java 21).
SONARQUBE_VERSION = "26.6.0.123539"
SCANNER_VERSION = "6.2.1.4610"

MD_SONAR_INTRO = """
# Iterative refinement loop with SonarQube feedback

Arm 5, exploratory part of RQ4: the same loop as `refinement-<method>` (200 DS-1000 problems, at most 3 iterations,
greedy, the median seed of the method), but the feedback for a failing answer is the execution error **plus the
SonarQube issues of that answer**. A problem leaves the loop once its tests pass, as in the execution-only loop, so both
loops are compared on the same problems. SonarQube measures and issues of every answer are stored too.

Results: `refinement-sonar-<method>/seed-<seed>/` (`refinement_results.jsonl`, `summary.json`) in `output_repo`.

**Machine:** template PyTorch, GPU 24 GB, disk 60 GB, RAM >= 16 GB (vLLM + SonarQube). SonarQube runs on this machine;
nothing has to run on the laptop. **Credentials:** `HF_TOKEN` (write) and `VAST_API_KEY`, as environment variables or in
the config below. Then **Run All**; the machine stops itself at the end.
"""

MD_SONAR_INSTALL = """
## 1b. Install SonarQube on this machine
Java 21 (Temurin JDK; Elasticsearch needs `jdk.jlink`), SonarQube server and sonar-scanner CLI into `/opt`. SonarQube refuses to run as root, so it runs as
user `sonar`. Elasticsearch bootstrap checks are disabled because a container cannot raise `vm.max_map_count`.
"""

SONAR_INSTALL = f"""
%%bash
set -e
cd /opt
command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip >/dev/null)
if [ ! -x jdk/bin/jlink ]; then   # a full JDK: SonarQube's Elasticsearch needs the jdk.jlink module (not in a JRE)
  rm -rf jdk && mkdir -p jdk
  curl -sfL "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse" | tar -xz -C jdk --strip-components=1
fi
if [ ! -d sonarqube ]; then
  curl -sfLo sq.zip https://binaries.sonarsource.com/Distribution/sonarqube/sonarqube-{SONARQUBE_VERSION}.zip
  unzip -q sq.zip && mv sonarqube-{SONARQUBE_VERSION} sonarqube && rm sq.zip
  echo "sonar.es.bootstrap.checks.disable=true" >> sonarqube/conf/sonar.properties
fi
if [ ! -d sonar-scanner ]; then
  curl -sfLo sc.zip https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-{SCANNER_VERSION}.zip
  unzip -q sc.zip && mv sonar-scanner-{SCANNER_VERSION} sonar-scanner && rm sc.zip
fi
id sonar >/dev/null 2>&1 || useradd -r -m -s /bin/bash sonar
chown -R sonar /opt/sonarqube
jdk/bin/java -version 2>&1 | head -1
echo "SonarQube {SONARQUBE_VERSION} and sonar-scanner {SCANNER_VERSION} installed"
"""

MD_SONAR_START = """
## 3b. Start SonarQube and create a token
Starts the server, waits until it is UP (a few minutes on first start), sets a random admin password (kept in
`/opt/sonar_admin.txt` so a rerun reuses it) and exports a fresh user token as `SONAR_TOKEN` for the refinement runs.
"""

SONAR_START = """
import secrets
import time

import httpx

os.environ.update(JAVA_HOME="/opt/jdk", SONAR_HOST_URL="http://localhost:9000",
                  SONAR_SCANNER="/opt/sonar-scanner/bin/sonar-scanner")
# no check=True: `sonar.sh start` exits 1 when SonarQube already runs (a rerun); the UP check below decides
subprocess.run(["su", "sonar", "-s", "/bin/bash", "-c",
                "SONAR_JAVA_PATH=/opt/jdk/bin/java /opt/sonarqube/bin/linux-x86-64/sonar.sh start"])
deadline = time.time() + 900
while True:
    try:
        status = httpx.get("http://localhost:9000/api/system/status", timeout=10).json()["status"]
    except Exception:
        status = "starting"
    if status == "UP":
        break
    if time.time() > deadline:
        subprocess.run("tail -n 40 /opt/sonarqube/logs/*.log", shell=True)
        raise SystemExit("SonarQube did not start within 15 minutes (logs above)")
    time.sleep(10)
print("SonarQube UP")

password_file = "/opt/sonar_admin.txt"
if not os.path.exists(password_file):
    password = "Sq-" + secrets.token_hex(8) + "-A1!"
    httpx.post("http://localhost:9000/api/users/change_password", auth=("admin", "admin"),
               data={"login": "admin", "previousPassword": "admin", "password": password}).raise_for_status()
    with open(password_file, "w") as handle:
        handle.write(password)
password = open(password_file).read().strip()
response = httpx.post("http://localhost:9000/api/user_tokens/generate", auth=("admin", password),
                      data={"name": f"refine-{int(time.time())}", "type": "USER_TOKEN"})
response.raise_for_status()
os.environ["SONAR_TOKEN"] = response.json()["token"]   # inherited by run_arm.py
print("SonarQube", httpx.get("http://localhost:9000/api/server/version").text, "| token ready")
"""

MD_SONAR_RUN = """
## 4. Refinement with SonarQube feedback
`seed: auto` picks the median seed of the method by DS-1000 pass@1 (the same seed as `refinement-<method>`).
"""

SONAR_RUN = REFINE.replace('"config.yml", "refine", method', '"config.yml", "refine-sonar", method').replace(
    "refinement-{method}", "refinement-sonar-{method}")


def sonar_notebook():
    helpers = {**HELPERS, "sonar.py": (SRC / "sonar.py").read_text()}
    install = install_cell()
    return notebook([
        md(MD_SONAR_INTRO), md("## 0. Configuration (edit here)"),
        code("%%writefile config.yml\n" + config_for(set(), {"lora", "qlora"})),
        md(MD_INSTALL), code(install),
        md(MD_SONAR_INSTALL), code(SONAR_INSTALL),
        md(MD_HELPERS), *[code(f"%%writefile {name}\n" + text) for name, text in helpers.items()],
        md(MD_SETUP), code(SETUP),
        md(MD_SONAR_START), code(SONAR_START),
        md(MD_SONAR_RUN), code(SONAR_RUN),
        md(MD_SHUTDOWN), code(SHUTDOWN),
    ])


def main():
    index = notebook([md(MD_INTRO), code(INTO_SRC), *shared_cells(INDEX_CONFIG)])
    (TESTING / "index.ipynb").write_text(json.dumps(index, indent=1))
    (TESTING / "index.py").write_text(index_script())
    print("wrote index.ipynb, index.py")
    for kind, (title, description) in SINGLE.items():
        path = TESTING / "arms" / kind / "index.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(single_notebook(title, description, config_for({kind}, set())), indent=1))
        print("wrote", path.relative_to(TESTING))
    for method in ("lora", "qlora"):
        path = TESTING / "refinement-loop" / "execution" / method / "index.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        description = (f"Iterative refinement (arm 5, RQ4) with the {method.upper()} adapter. Needs the {method} "
                       f"results on the Hub; `seed: auto` picks its median seed by DS-1000 pass@1. GPU: 24 GB.")
        path.write_text(json.dumps(single_notebook(f"refinement-loop {method}", description,
                                                   config_for(set(), {method})), indent=1))
        print("wrote", path.relative_to(TESTING))
    path = TESTING / "refinement-loop" / "sonar" / "index.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sonar_notebook(), indent=1))
    print("wrote", path.relative_to(TESTING))


if __name__ == "__main__":
    main()
