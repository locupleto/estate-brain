"""Golden cases: selection (evals/golden.toml, committed: ids only) and
captured content (~/.estate-brain/cases/<id>/, never committed)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

HOME = Path(os.environ.get("ESTATE_BRAIN_HOME", "~/.estate-brain")).expanduser()
CASES = HOME / "cases"
RUNS = HOME / "runs"
GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden.toml"
_REMOTE_SCRIPT = Path(__file__).with_name("capture_remote.py")
_ROLE_DIR = re.compile(r"data/memory/([A-Z][A-Z_]+)\b")


def load_golden(path: Path = GOLDEN) -> list[dict]:
    data = tomllib.loads(path.read_text())
    d = data.get("defaults", {})
    out = []
    for c in data.get("case", []):
        c = {**d, **c}
        c["remote_repo"] = f"{c['remote_base']}/{c['repo']}"
        c["local_repo"] = f"{c['local_base']}/{c['repo']}"
        c.setdefault("expect", [])
        out.append(c)
    return out


def _ssh(host: str, script: str, stdin: bytes | None = None, timeout: int = 120) -> bytes:
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, f"ulimit -n 4096; {script}"],
                       input=stdin, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh {host}: {r.stderr.decode()[-300:]}")
    return r.stdout


def capture(case: dict, force: bool = False) -> Path:
    """Pull one transcript's turn (and, for bot turns, the bot's memory as
    it is now) into CASES/<id>/."""
    out = CASES / case["id"]
    if (out / "case.json").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    slug = case["remote_repo"].replace("/", "-")
    path = f"~/.claude/projects/{slug}/{case['transcript']}.jsonl"
    rec = json.loads(_ssh(case["host"], f"python3 - {path}",
                          stdin=_REMOTE_SCRIPT.read_bytes()))
    roles = sorted(set(_ROLE_DIR.findall(rec["prompt"])))
    rec.update({k: case[k] for k in ("id", "repo", "transcript", "contract", "expect",
                                     "remote_repo", "local_repo", "host")})
    rec["memory_roles"] = roles
    if roles and case["contract"] in ("wrapper-json", "wrapper-any", "door"):
        dirs = " ".join(f"data/memory/{r}" for r in roles)
        tgz = _ssh(case["host"], f"cd {case['remote_repo']} && tar czf - {dirs} 2>/dev/null",
                   timeout=300)
        (out / "memory.tgz").write_bytes(tgz)
    (out / "case.json").write_text(json.dumps(rec, indent=1))
    return out


def load_case(case_id: str) -> dict:
    """Captured content, with the scoring fields (contract, expect) taken
    from golden.toml as it is NOW, so editing an expectation needs no
    recapture."""
    rec = json.loads((CASES / case_id / "case.json").read_text())
    for c in load_golden():
        if c["id"] == case_id:
            rec["contract"], rec["expect"] = c["contract"], c.get("expect", [])
    return rec
