"""`brain run` / `brain policy`: one task from the shell.

    brain run --spec SPEC.json --prompt-file PROMPT.txt
    brain policy --spec SPEC.json

For callers written in bash (the estate's bot wrapper). The spec is a JSON
object; every key is optional except `workdir`:

    workdir           the session's working directory (the workspace)
    app, role         for the resolver and the ledger
    model, tier       a model alias/id, else the tier (default "deep")
    effort            low | medium | high
    budget_usd        a spend ceiling for the turn
    timeout_s         wall-clock limit (default: none; the caller keeps the clock)
    persist           keep the session transcript (default true)
    workspace_config  load the workspace's own settings, its hooks (default true)
    permissions_file  a Claude Code permission file, translated by claude_rules
    project_dir       substituted for {{PROJECT_DIR}} in it (default: workdir)
    mcp_config        where the allowed servers come from (default: workdir/.mcp.json)
    agents_file       Claude Code --agents JSON (subagents)
    debug_file        claude-code: the CLI's debug log

`run` prints the reply text on stdout. Exit codes, the caller's contract:

    0   ok
    1   any other error (the engine's text, if any, still on stdout)
    3   the task's own timeout
    4   the spend ceiling stopped the turn
    5   transient provider trouble (overloaded, 529, 503): worth a retry
    64  the spec or the fence was refused; NO model was called
    143 stopped by SIGTERM (the caller's watchdog); the engine is killed with it

`policy` prints the translated ToolPolicy and, for claude-code, the settings
it emits and any round-trip difference against the permission file; it
exits 64 when the fence cannot be translated or does not round-trip.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from pathlib import Path

from .claude_rules import PolicyError, load_rules, policy_from, roundtrip_problems
from .task import Task, ToolPolicy

OK, ERROR, TIMEOUT, BUDGET, TRANSIENT, REFUSED, STOPPED = 0, 1, 3, 4, 5, 64, 143
_TRANSIENT = re.compile(r"overloaded|API Error: 5(29|03)|\"?status\"?:? *529|Error: 503|529 |503 Service", re.I)
_NO_LIMIT = 7 * 24 * 3600.0


class _Stopped(BaseException):
    pass


def _stop(signum, frame):
    raise _Stopped()


def task_from(spec: dict, prompt: str) -> Task:
    if not spec.get("workdir"):
        raise PolicyError("spec: workdir is required")
    workdir = str(spec["workdir"])
    tools = ToolPolicy(files=False)
    if spec.get("permissions_file"):
        rules = load_rules(spec["permissions_file"], spec.get("project_dir") or workdir)
        agents = json.loads(Path(spec["agents_file"]).read_text()) if spec.get("agents_file") else None
        mcp = spec.get("mcp_config") or str(Path(workdir) / ".mcp.json")
        tools = policy_from(rules, mcp, agents)
    known = {"workdir", "app", "role", "model", "tier", "effort", "budget_usd", "timeout_s",
             "persist", "workspace_config", "permissions_file", "project_dir", "mcp_config",
             "agents_file", "debug_file"}
    extra = set(spec) - known
    if extra:
        raise PolicyError(f"spec: unknown keys {sorted(extra)}")
    return Task(
        prompt=prompt, workdir=workdir, tools=tools,
        model=spec.get("model") or None, tier=spec.get("tier") or "deep",
        effort=spec.get("effort") or None,
        budget_usd=float(spec["budget_usd"]) if spec.get("budget_usd") else None,
        timeout_s=float(spec.get("timeout_s") or _NO_LIMIT),
        persist=spec.get("persist", True), workspace_config=spec.get("workspace_config", True),
        app=spec.get("app", ""), role=spec.get("role", ""),
        engine_opts={"claude-code": {"debug_file": spec["debug_file"]}} if spec.get("debug_file") else {},
    )


def classify(res) -> int:
    if not res.error:
        return OK
    blob = f"{res.text}\n{res.raw_tail}"
    if res.error == "timeout":
        return TIMEOUT
    if "budget" in res.error or re.search(r"Exceeded USD budget", blob, re.I):
        return BUDGET
    if _TRANSIENT.search(blob):
        return TRANSIENT
    return ERROR


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="brain")
    ap.add_argument("cmd", choices=("run", "policy"))
    ap.add_argument("--spec", required=True)
    ap.add_argument("--prompt-file")
    a = ap.parse_args(argv)
    try:
        spec = json.loads(Path(a.spec).read_text())
        prompt = Path(a.prompt_file).read_text() if a.prompt_file else ""
        if a.cmd == "run" and not prompt:
            raise PolicyError("run: --prompt-file is required and must not be empty")
        task = task_from(spec, prompt)
    except (PolicyError, OSError, ValueError) as e:
        print(f"brain: refused: {e}", file=sys.stderr)
        return REFUSED

    if a.cmd == "policy":
        from .engines import claude_code
        out = {"policy": task.tools.to_dict(), "settings": claude_code.settings_for(task)}
        problems = []
        if spec.get("permissions_file"):
            rules = load_rules(spec["permissions_file"], spec.get("project_dir") or spec["workdir"])
            problems = roundtrip_problems(rules, task.tools, out["settings"])
        out["roundtrip_problems"] = problems
        print(json.dumps(out, indent=2))
        for pr in problems:
            print(f"brain: round trip: {pr}", file=sys.stderr)
        return REFUSED if problems else OK

    from . import run as brain_run
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)
    try:
        res = brain_run(task)
    except _Stopped:
        # The engine's own cleanup (a finally around the turn) has killed its
        # process group by the time this lands.
        print("brain: stopped by signal", file=sys.stderr)
        return STOPPED
    if res.text:
        sys.stdout.write(res.text)
        sys.stdout.flush()
    rc = classify(res)
    if rc:
        print(f"brain: {res.engine} {res.model}: {res.error}: {res.raw_tail}", file=sys.stderr)
    return rc
