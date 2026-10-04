from evalkit import (
    ast_valid,
    code4ml_messages,
    ds1000_messages,
    ds1000_program,
    extract_code,
    fewshot_ds1000_ids,
    pass_at_k,
    postprocess_ds1000,
    refine_messages,
    select_fewshot,
    summarize_ds1000,
)


def test_postprocess_matches_official_ds1000_rules():
    raw = "```python\nresult = df.mean()\n```\nSome explanation"
    assert postprocess_ds1000(raw) == "\nresult = df.mean()\n"
    assert postprocess_ds1000("<code>x = 1</code> trailing") == "x = 1"
    assert postprocess_ds1000("x = 1\nEND SOLUTION\nmore") == "x = 1"


def test_program_appends_test_string_only_when_defined():
    with_string = {"code_context": "def test_execution(s): pass\ndef test_string(s): pass"}
    without = {"code_context": "def test_execution(s): pass"}
    assert ds1000_program(with_string, "x = 1").endswith("test_execution(code)\ntest_string(code)\n")
    assert "test_string(code)" not in ds1000_program(without, "x = 1")
    assert "code = 'x = 1'" in ds1000_program(without, "x = 1")


def test_pass_at_k_unbiased_estimator():
    assert pass_at_k(10, 0, 1) == 0.0
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9
    assert pass_at_k(10, 1, 10) == 1.0
    assert abs(pass_at_k(10, 2, 5) - (1 - (8 * 7 * 6 * 5 * 4) / (10 * 9 * 8 * 7 * 6))) < 1e-9


def test_summary_per_problem_and_library():
    problems = {0: "Pandas", 1: "Numpy"}
    results = (
        [{"problem_id": 0, "sample": s, "passed": s < 4} for s in range(10)]
        + [{"problem_id": 1, "sample": s, "passed": False} for s in range(10)]
    )
    summary = summarize_ds1000(results, problems)
    assert summary["problems"] == 2
    assert abs(summary["pass@1"] - 0.2) < 1e-9
    assert summary["pass@10"] == (1.0 + 0.0) / 2
    assert abs(summary["by_library"]["Pandas"]["pass@1"] - 0.4) < 1e-9
    assert summary["per_problem"][1] == {"n": 10, "c": 0, "pass@1": 0.0, "pass@10": 0.0}


def test_fewshot_selection_is_fixed_by_seed():
    rows = [{"id": str(i)} for i in range(50)]
    assert select_fewshot(rows, 3, 42) == select_fewshot(rows, 3, 42)
    assert len(select_fewshot(rows, 3, 42)) == 3
    assert fewshot_ds1000_ids(1000) == fewshot_ds1000_ids(1000)
    assert len(set(fewshot_ds1000_ids(1000))) == 3


def test_messages_put_shots_before_the_question():
    shots = [{"instruction": "Write Python code to train the model.", "code": "model.fit(X, y)"}]
    msgs = code4ml_messages("Write Python code to shuffle the data.", shots)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "Write Python code to shuffle the data."
    ds = ds1000_messages("Problem text", [{"prompt": "P1", "reference_code": "x = 1"}])
    assert [m["role"] for m in ds] == ["user", "assistant", "user"] and ds[-1]["content"] == "Problem text"


def test_refine_messages_continue_the_conversation_with_the_error():
    history = ds1000_messages("Problem text", [])
    msgs = refine_messages(history, "x = 1/0", "failed: division by zero")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "division by zero" in msgs[-1]["content"]


def test_extract_code_and_ast_gate():
    assert extract_code("Here:\n```python\nimport pandas as pd\n```") == "import pandas as pd"
    assert extract_code("plain = 1") == "plain = 1"
    assert ast_valid("x = 1")
    assert not ast_valid("x = (")


def test_describe_error_turns_empty_assertions_into_feedback():
    from evalkit import describe_error
    assert "wrong result" in describe_error("failed: ")
    assert "timed out" in describe_error("timed out")
    assert describe_error("failed: name 'np' is not defined") == "name 'np' is not defined"


def test_pass_at_k_is_zero_when_nothing_passed_even_with_few_samples():
    assert pass_at_k(3, 0, 10) == 0.0
