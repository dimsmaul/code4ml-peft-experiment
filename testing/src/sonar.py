"""SonarQube client for the refinement loop with structural feedback.

Scans the answers of one iteration as the files of one SonarQube project, the same way the rating service does
(dataset/pipeline/sonarqube/api/app.py: sonar-scanner CLI, then the Web API), and returns per answer its measures and
its open issues. Needs a running SonarQube server and SONAR_TOKEN (a user token of an admin, see the notebook).
"""
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx

HOST = os.environ.get("SONAR_HOST_URL", "http://localhost:9000")
SCANNER = os.environ.get("SONAR_SCANNER", "/opt/sonar-scanner/bin/sonar-scanner")
METRIC_KEYS = ["sqale_rating", "code_smells", "complexity", "cognitive_complexity", "bugs", "sqale_index", "ncloc"]
RATING_MAP = {"1.0": "A", "2.0": "B", "3.0": "C", "4.0": "D", "5.0": "E"}
PAGE = 500


def _client():
    return httpx.Client(base_url=HOST, auth=(os.environ["SONAR_TOKEN"], ""), timeout=60)


def version():
    with _client() as client:
        return client.get("/api/server/version").text.strip()


def _wait(client, key, timeout_s):
    """Wait until the Compute Engine task of the project finished (issues are indexed by then)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get("/api/ce/component", params={"component": key}).json()
        status = (body.get("current") or {}).get("status")
        if not body.get("queue") and status == "SUCCESS":
            return
        if status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"SonarQube analysis {key} {status}")
        time.sleep(2)
    raise TimeoutError(f"SonarQube analysis {key} did not finish in {timeout_s} s")


def scan(codes, timeout_s=600):
    """{id: code} -> {id: {"measures": {...}, "issues": [{"line", "message", "rule"}, ...]}}"""
    key = f"refine_{uuid.uuid4().hex[:12]}"
    files = {cid: f"s_{cid}.py" for cid in codes}
    with tempfile.TemporaryDirectory() as workdir, _client() as client:
        for cid, code in codes.items():
            (Path(workdir) / files[cid]).write_text(code, encoding="utf-8")
        proc = subprocess.run([SCANNER, f"-Dsonar.host.url={HOST}", f"-Dsonar.token={os.environ['SONAR_TOKEN']}",
                               f"-Dsonar.projectKey={key}", "-Dsonar.sources=.", "-Dsonar.python.version=3",
                               "-Dsonar.sourceEncoding=UTF-8"], cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"sonar-scanner failed: {proc.stderr[-2000:] or proc.stdout[-2000:]}")
        _wait(client, key, timeout_s)

        issues, page = {}, 1
        while True:
            response = client.get("/api/issues/search", params={"components": key, "resolved": "false",
                                                                 "ps": PAGE, "p": page})
            response.raise_for_status()
            body = response.json()
            for issue in body["issues"]:
                issues.setdefault(issue["component"].split(":", 1)[1], []).append(
                    {"line": issue.get("line"), "message": issue["message"], "rule": issue["rule"]})
            if page * PAGE >= body["paging"]["total"]:
                break
            page += 1

        out = {}
        for cid, name in files.items():
            response = client.get("/api/measures/component", params={"component": f"{key}:{name}",
                                                                     "metricKeys": ",".join(METRIC_KEYS)})
            measures = ({m["metric"]: m.get("value") for m in response.json().get("component", {}).get("measures", [])}
                        if response.status_code == 200 else {})
            if "sqale_rating" in measures:
                measures["maintainability_rating"] = RATING_MAP.get(measures["sqale_rating"], "?")
            out[cid] = {"measures": measures,
                        "issues": sorted(issues.get(name, []), key=lambda issue: issue["line"] or 0)}
    return out
