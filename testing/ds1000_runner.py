"""Run DS-1000 tests for generated answers. Executed with the DS-1000 Python 3.10 environment.

    python ds1000_runner.py PROBLEMS.jsonl ANSWERS.jsonl RESULTS.jsonl [--workers N] [--timeout S]

PROBLEMS.jsonl: {"problem_id", "code_context"} per line
ANSWERS.jsonl : {"problem_id", "sample", "code"} per line (code already post-processed)
RESULTS.jsonl : {"problem_id", "sample", "passed", "result"} per line

Uses the unmodified official execution.py (xlang-ai/DS-1000, commit b39aab7) placed next to this file.
"""
import argparse
import concurrent.futures as cfuts
import json
import multiprocessing
import os

# Same settings as the official test_ds1000.py: quiet TensorFlow, tests run on CPU only.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Headless matplotlib. A Jupyter kernel exports MPLBACKEND=module://matplotlib_inline..., which the test processes
# would inherit; ds1000-env has no matplotlib_inline, so every matplotlib test failed with "No module named".
os.environ["MPLBACKEND"] = "Agg"
# One thread per test process. By default TensorFlow and the BLAS libraries start a thread per host core in every
# test process; with many parallel tests that exceeds the container's thread limit and TensorFlow aborts
# ("Thread tf_Compute creation via pthread_create() failed"), which would count as a failed test.
for _var in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import execution  # noqa: E402


def allocated_cpus():
    """CPUs this container may use (cgroup quota), not the host's core count."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]          # cgroup v2
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())            # cgroup v1
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def build_program(code_context, code):
    return (
        code_context + "\n"
        + f"code = {repr(code)}\n"
        + "test_execution(code)\n"
        + ("test_string(code)\n" if "test_string(" in code_context else "\n")
    )


def run(problems_path, answers_path, results_path, workers, timeout):
    with open(problems_path) as handle:
        context = {row["problem_id"]: row["code_context"] for row in map(json.loads, handle)}
    with open(answers_path) as handle:
        answers = [json.loads(line) for line in handle]

    results = []
    with cfuts.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(execution.check_correctness,
                        build_program(context[a["problem_id"]], a["code"]), timeout, i): a
            for i, a in enumerate(answers)
        }
        for done, future in enumerate(cfuts.as_completed(futures), 1):
            answer, outcome = futures[future], future.result()
            results.append({"problem_id": answer["problem_id"], "sample": answer["sample"],
                            "passed": outcome["passed"], "result": outcome["result"]})
            if done % 500 == 0 or done == len(futures):
                print(f"tested {done}/{len(futures)}", flush=True)

    results.sort(key=lambda r: (r["problem_id"], r["sample"]))
    with open(results_path, "w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    # execution.py runs each test in a nested local function, which needs the fork start method.
    multiprocessing.set_start_method("fork", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("problems")
    parser.add_argument("answers")
    parser.add_argument("results")
    parser.add_argument("--workers", type=int, default=min(16, allocated_cpus()))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    print(f"testing with {args.workers} parallel workers", flush=True)
    run(args.problems, args.answers, args.results, args.workers, args.timeout)
