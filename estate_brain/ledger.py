"""One JSON line per brain call: who asked, which engine/model, what it cost.

Default file ~/.estate-brain/ledger.jsonl; BRAIN_LEDGER=<path> moves it,
BRAIN_LEDGER=off disables it. Never raises: a ledger problem must not break
the caller's turn. No prompt or reply text is ever written.
"""

from __future__ import annotations

import json
import os
import socket
import time

from .task import Result, Task


def path() -> str | None:
    p = os.environ.get("BRAIN_LEDGER", "")
    if p.lower() in ("off", "0", "no", "false"):
        return None
    return os.path.expanduser(p or "~/.estate-brain/ledger.jsonl")


def record(task: Task, res: Result, resolution=None) -> None:
    p = path()
    if not p:
        return
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname().split(".")[0],
           "app": task.app, "role": task.role, "engine": res.engine, "model": res.model,
           "tier": task.tier, "ok": res.ok, "error": res.error[:160],
           "latency_s": res.latency_s, "input_tokens": res.input_tokens,
           "output_tokens": res.output_tokens, "cost_usd": res.cost_usd,
           "tools": len(res.tool_calls)}
    if resolution is not None:
        # Where the choice came from: the Armory's stale-cache sense reads these.
        row.update(source=resolution.source, profile=resolution.profile,
                   registry=resolution.version)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
