"""Turn a captured case into a Task that runs in a throwaway sandbox.

The sandbox holds a COPY of the bot's memory as captured (the only writable
place) and symlinks to the rest of the repo's Studio checkout for reading.
Prompt paths that pointed at the production checkout are rewritten into the
sandbox. Only read-only MCP servers are mounted: nothing a replay does can
mail, post, generate paid media or touch real memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from ..task import Task, ToolPolicy
from .cases import CASES

SAFE_MCP = {"finance", "marketdata-db", "exchange-calendar", "openweathermap",
            "rss-news", "polymarket", "google-trends", "telegram-chat-db",
            "youtube-history", "ai-search", "maps"}


def infer_schema(value, depth: int = 0) -> dict:
    """A loose JSON Schema from a reference value: types and required keys,
    two levels deep. Enough to hold a replay to the same reply shape."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        s: dict = {"type": "array"}
        if value and depth < 2:
            s["items"] = infer_schema(value[0], depth + 1)
        return s
    if isinstance(value, dict):
        s = {"type": "object", "required": [k for k, v in value.items() if v is not None]}
        if depth < 2:
            s["properties"] = {k: infer_schema(v, depth + 1) for k, v in value.items() if v is not None}
        return s
    return {}


def _mcp_servers(local_repo: str) -> dict:
    for repo in (local_repo, str(Path(local_repo).parent / "the-armory")):
        p = Path(repo) / ".mcp.json"
        if p.exists():
            servers = json.loads(p.read_text()).get("mcpServers", {})
            return {n: {"command": [s["command"], *s.get("args", [])], "env": s.get("env") or {}}
                    for n, s in servers.items() if n in SAFE_MCP and s.get("command")}
    return {}


def memory_digest(sandbox: str) -> str:
    h = hashlib.sha256()
    root = Path(sandbox) / "data" / "memory"
    for p in sorted(root.rglob("*")) if root.exists() else []:
        if p.is_file() and not p.is_symlink():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def build(case: dict, model_tier: str = "standard") -> tuple[Task, str]:
    """(task, sandbox dir). The caller removes the sandbox."""
    sandbox = tempfile.mkdtemp(prefix=f"eb-{case['id']}-")
    local = Path(case["local_repo"])
    if local.exists():
        for entry in local.iterdir():
            if entry.name in ("data", ".git", ".mcp.json", ".claude"):
                continue
            os.symlink(entry, Path(sandbox) / entry.name)
    (Path(sandbox) / "data" / "memory").mkdir(parents=True, exist_ok=True)
    tgz = CASES / case["id"] / "memory.tgz"
    if tgz.exists() and tgz.stat().st_size:
        with tarfile.open(tgz) as t:
            t.extractall(sandbox, filter="data")
    prompt = case["prompt"].replace(case["remote_repo"], sandbox)
    contract = case["contract"]
    if contract == "bare-json":
        tools = ToolPolicy(files=False)
        ref = None
        try:
            ref = json.loads(case["reference_text"])
        except (TypeError, ValueError):
            pass
        schema = infer_schema(ref) if isinstance(ref, (dict, list)) else None
    elif contract == "structured":
        tools = ToolPolicy(write_paths=[f"{sandbox}/data/memory"], web=True)
        schema = infer_schema(case.get("reference_structured")) if case.get("reference_structured") else None
    else:
        tools = ToolPolicy(write_paths=[f"{sandbox}/data/memory"], web=True,
                           mcp_servers=_mcp_servers(case["local_repo"]))
        schema = None
    task = Task(prompt=prompt, workdir=sandbox, schema=schema, tools=tools,
                tier=model_tier, timeout_s=900, max_turns=40,
                app=case["repo"], role=case["id"])
    return task, sandbox
