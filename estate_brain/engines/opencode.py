"""opencode engine: OpenCode's headless `run`, any provider it supports.

Every run is isolated (verified 2026-10-31, OpenCode 2.2.5):
- its own XDG config/data/state/cache dirs, so the user's global config and
  stored logins are never merged in;
- the whole config passed inline (OPENCODE_CONFIG_CONTENT);
- Claude Code / external skills / project config discovery switched off;
- `--standalone` (a private server, not the shared background service) and
  stdin closed (an open stdin makes `run` wait forever).

Permissions are enforced by OpenCode itself: a catch-all deny, then explicit
allows. MCP tools must be allowed by name (`<server>_*`) — the catch-all
deny hides them otherwise. No native JSON-schema output: the schema goes
into the prompt and the reply is extracted and validated (one retry).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from ..proc import Run, find_binary
from ..structured import extract_json, schema_instruction, validate
from ..task import Result, Task, ToolCall

NAME = "opencode"
_CANDIDATES = ("/opt/homebrew/bin/opencode", "~/.opencode/bin/opencode", "/usr/local/bin/opencode")
_NEUTRAL = {"read": "read", "glob": "read", "grep": "read", "list": "read",
            "edit": "edit", "write": "edit", "patch": "edit",
            "webfetch": "web", "websearch": "web", "bash": "shell",
            "task": "subagent", "skill": "skill"}
_ISOLATION_FLAGS = ("OPENCODE_DISABLE_CLAUDE_CODE", "OPENCODE_DISABLE_EXTERNAL_SKILLS",
                    "OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE_DISABLE_AUTOUPDATE",
                    "OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_PURE")
# A workspace's own config (Task.workspace_config) is the Claude Code
# .claude/ directory today (hooks); OpenCode has no equivalent to load, so the
# flag is accepted and nothing extra is read.


def _patterns(path: str, workdir: str) -> list[str]:
    path = path.rstrip("/")
    if os.path.isfile(path):
        out = [path]
        if workdir and path.startswith(workdir.rstrip("/") + "/"):
            out.append(path[len(workdir.rstrip("/")) + 1:])
        return out
    out = [f"{path}/*", f"{path}/**"]
    if workdir and path.startswith(workdir.rstrip("/") + "/"):
        rel = path[len(workdir.rstrip("/")) + 1:]
        out += [f"{rel}/*", f"{rel}/**"]
    return out


def config_for(task: Task, model: str) -> dict:
    t = task.tools
    perm: dict = {"*": "deny"}
    if t.files:
        perm.update({"read": "allow", "glob": "allow", "grep": "allow", "list": "allow"})
    edit = {"*": "deny"}
    for p in t.write_paths:
        for pat in _patterns(p, task.workdir):
            edit[pat] = "allow"
    perm["edit"] = edit
    ext = {"*": "deny"}
    for p in t.read_paths + t.write_paths:
        if not (task.workdir and p.startswith(task.workdir.rstrip("/") + "/")):
            ext[f"{p.rstrip('/')}/**"] = "allow"
    perm["external_directory"] = ext
    perm["bash"] = "allow" if t.shell else "deny"
    perm["webfetch"] = perm["websearch"] = "allow" if t.web else "deny"
    perm["skill"] = "allow" if t.skills else "deny"
    perm["task"] = "allow" if t.subagents else "deny"
    for s in t.mcp_servers:
        perm[f"{s}_*"] = "allow"
    for d in t.mcp_deny:
        srv, _, tool = d.partition(".")
        perm[f"{srv}_{tool}" if tool else f"{srv}_*"] = "deny"
    cfg: dict = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False, "share": "disabled", "model": model,
        "permission": perm,
        "mcp": {name: {"type": "local", "command": list(spec["command"]),
                       "environment": spec.get("env") or {}, "enabled": True,
                       "timeout": 60000}
                for name, spec in t.mcp_servers.items()},
    }
    agents = {}
    if task.system:
        agents["estate"] = {"mode": "primary", "description": "estate task",
                            "prompt": task.system}
    for n, a in t.subagents.items():
        agents[n] = {"mode": "subagent", "description": a.get("description", n),
                     "prompt": a.get("prompt", "")}
    if agents:
        cfg["agent"] = agents
    return cfg


def env_for(task: Task, model: str, home: str) -> dict:
    env = dict(os.environ)
    for sub in ("config", "data", "state", "cache"):
        os.makedirs(os.path.join(home, sub), exist_ok=True)
    env.update({"XDG_CONFIG_HOME": os.path.join(home, "config"),
                "XDG_DATA_HOME": os.path.join(home, "data"),
                "XDG_STATE_HOME": os.path.join(home, "state"),
                "XDG_CACHE_HOME": os.path.join(home, "cache"),
                "OPENCODE_CONFIG_CONTENT": json.dumps(config_for(task, model))})
    for f in _ISOLATION_FLAGS:
        env[f] = "1"
    # The models catalogue is normally fetched/cached per data dir; reuse the
    # user's cached copy read-only so an isolated run still knows its models.
    cached = os.path.expanduser("~/.cache/opencode/models.json")
    if os.path.exists(cached):
        env.setdefault("OPENCODE_MODELS_PATH", cached)
    return env


def neutral_tool_name(name: str, servers) -> str:
    for s in servers:
        if name.startswith(f"{s}_"):
            return f"mcp:{s}.{name[len(s) + 1:]}"
    return _NEUTRAL.get(name, name)


def _run_once(task: Task, model: str, exe: str, prompt: str) -> Result:
    res = Result(engine=NAME, model=model)
    home = tempfile.mkdtemp(prefix="estate-brain-oc-")
    cmd = [exe, "run", "--standalone", "--auto", "--format", "json", "-m", model]
    if task.system:
        cmd += ["--agent", "estate"]
    for i, img in enumerate(task.images):
        if isinstance(img, (bytes, bytearray)):
            path = os.path.join(home, f"image{i}." + ("png" if img[:4] == b"\x89PNG" else "jpg"))
            with open(path, "wb") as f:
                f.write(img)
            img = path
        cmd += ["--file", img]
    cmd.append(prompt)
    t0 = time.monotonic()
    run_ = Run(cmd, env=env_for(task, model, home), cwd=task.workdir or None,
               stdin_text=None, timeout=task.timeout_s)
    texts: list[str] = []
    cost = 0.0
    try:
        for line in run_.lines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = ev.get("part") or {}
            pt = part.get("type")
            if ev.get("type") == "error":
                err = ev.get("error") or {}
                res.error = f"{err.get('type', 'error')}: {err.get('message', '')}"[:300]
            elif pt == "text" and part.get("text"):
                texts.append(part["text"])
                if task.on_text:
                    task.on_text(part["text"])
            elif pt == "tool":
                st = part.get("state") or {}
                status = st.get("status")
                if status in ("completed", "error"):
                    res.tool_calls.append(ToolCall(
                        neutral_tool_name(part.get("tool", ""), task.tools.mcp_servers),
                        st.get("input") or {}, ok=(status == "completed"),
                        error=str(st.get("error") or "")[:300],
                        output=str(st.get("output") or "")[:800]))
            elif pt == "step-finish":
                tok = part.get("tokens") or {}
                cache = tok.get("cache") or {}
                res.input_tokens += (tok.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
                res.output_tokens += (tok.get("output") or 0) + (tok.get("reasoning") or 0)
                cost += part.get("cost") or 0.0
    finally:
        rc = run_.close()
        shutil.rmtree(home, ignore_errors=True)
    res.latency_s = round(time.monotonic() - t0, 2)
    res.cost_usd = round(cost, 6)
    res.text = texts[-1].strip() if texts else ""
    if run_.timed_out:
        res.error = "timeout"
    elif not res.error and not texts:
        res.error = f"no reply (rc={rc})"
    if res.error:
        res.raw_tail = run_.stderr_tail()
    return res


def run(task: Task, model: str | None = None) -> Result:
    model = model or task.model
    if not model:
        return Result(engine=NAME, error="opencode needs an explicit provider/model")
    exe = find_binary("opencode", "OPENCODE_BIN", _CANDIDATES)
    if not exe:
        return Result(engine=NAME, model=model, error="opencode binary not found")
    prompt = task.prompt + (schema_instruction(task.schema) if task.schema else "")
    res = _run_once(task, model, exe, prompt)
    if task.schema and res.ok:
        val = extract_json(res.text)
        errs = ["no JSON found"] if val is None else validate(val, task.schema)
        if errs:
            retry = _run_once(task, model, exe, prompt + "\n\nYour previous reply was "
                              "not valid (" + "; ".join(errs[:3]) + "). Reply again with ONLY the JSON.")
            retry.tool_calls = res.tool_calls + retry.tool_calls
            retry.latency_s += res.latency_s
            retry.cost_usd = (retry.cost_usd or 0) + (res.cost_usd or 0)
            res = retry
            val = extract_json(res.text)
            errs = ["no JSON found"] if val is None else validate(val, task.schema)
        if not errs:
            res.structured = val
        elif res.ok:
            res.error = "schema: " + "; ".join(errs[:3])
    return res
