"""Shared helpers for the testing notebooks (generation side, standard library only).

The notebooks write this file next to themselves and import it, so every model is
prompted, post-processed and scored the same way.
"""
from __future__ import annotations

import ast
import random
import re
from collections import defaultdict
from math import comb

FEWSHOT_K = 3
FEWSHOT_SEED = 42
REFINE_INSTRUCTION = (
    "The code above failed when it was run against the tests. Error:\n{error}\n\n"
    "Fix the code. Reply with the corrected code only."
)


def postprocess_ds1000(code: str) -> str:
    """Official DS-1000 post-processing (test_ds1000.py, commit b39aab7)."""
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.split("\nEND SOLUTION")[0]
    code = code.replace("<code>", "")
    return code


def ds1000_program(problem: dict, code: str) -> str:
    """Test program exactly as assembled by the official DS-1000 evaluator."""
    return (
        problem["code_context"] + "\n"
        + f"code = {repr(code)}\n"
        + "test_execution(code)\n"
        + ("test_string(code)\n" if "test_string(" in problem["code_context"] else "\n")
    )


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize_ds1000(results: list[dict], library_of: dict[int, str], k_values=(1, 10)) -> dict:
    """Per-problem and aggregate pass@k from per-sample results {problem_id, sample, passed}."""
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in results:
        counts[row["problem_id"]][0] += 1
        counts[row["problem_id"]][1] += int(bool(row["passed"]))
    per_problem = {
        pid: {"n": n, "c": c, **{f"pass@{k}": pass_at_k(n, c, k) for k in k_values}}
        for pid, (n, c) in sorted(counts.items())
    }
    by_library: dict[str, list[dict]] = defaultdict(list)
    for pid, scores in per_problem.items():
        by_library[library_of[pid]].append(scores)

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    rows = list(per_problem.values())
    return {
        "problems": len(rows),
        **{f"pass@{k}": mean(rows, f"pass@{k}") for k in k_values},
        "by_library": {lib: {"problems": len(r), **{f"pass@{k}": mean(r, f"pass@{k}") for k in k_values}}
                       for lib, r in sorted(by_library.items())},
        "per_problem": per_problem,
    }


def select_fewshot(rows: list[dict], k: int = FEWSHOT_K, seed: int = FEWSHOT_SEED) -> list[dict]:
    return random.Random(seed).sample(rows, k)


def fewshot_ds1000_ids(n_problems: int = 1000, k: int = FEWSHOT_K, seed: int = FEWSHOT_SEED) -> list[int]:
    """DS-1000 problems used as few-shot examples; excluded from every arm's analysis."""
    return sorted(random.Random(seed).sample(range(n_problems), k))


def code4ml_messages(instruction: str, shots: list[dict] | None = None) -> list[dict]:
    """Same chat format as training: user = instruction, assistant = code."""
    messages = []
    for shot in shots or []:
        messages += [{"role": "user", "content": shot["instruction"]},
                     {"role": "assistant", "content": shot["code"]}]
    return messages + [{"role": "user", "content": instruction}]


def ds1000_messages(prompt: str, shots: list[dict] | None = None) -> list[dict]:
    messages = []
    for shot in shots or []:
        messages += [{"role": "user", "content": shot["prompt"]},
                     {"role": "assistant", "content": shot["reference_code"]}]
    return messages + [{"role": "user", "content": prompt}]


def describe_error(result: str) -> str:
    """Readable feedback from a DS-1000 execution result ("failed: <msg>" or "timed out")."""
    if result == "timed out":
        return "The code timed out."
    message = result.removeprefix("failed:").strip()
    return message or "The code ran but produced a wrong result (a test assertion failed)."


def refine_messages(history: list[dict], previous_code: str, error: str) -> list[dict]:
    return history + [{"role": "assistant", "content": previous_code},
                      {"role": "user", "content": REFINE_INSTRUCTION.format(error=error)}]


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Code4ML track: take the first fenced block if the model used one, else the whole reply."""
    match = _FENCE.search(completion)
    return (match.group(1) if match else completion).strip()


def ast_valid(code: str) -> bool:
    """Validity gate from proposal Bab III: only parseable code is rated."""
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False
