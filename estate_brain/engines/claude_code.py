"""claude-code engine: the `claude` CLI on the host's subscription login.

Translates the neutral Task into the flags the estate's harnesses already use
(--settings permission rules, --strict-mcp-config, --json-schema, --agents)
and reads the stream-json event log back into a Result.

Two modes:
- bare (no tools at all): `--tools ""` + `--system-prompt` replaces Claude
  Code's own system prompt — a plain model call (fast brain, Scriptorium).
- agent: Claude Code's scaffolding stays; a persona goes in through
  `--append-system-prompt`, the fence through `--settings`.
User settings are never loaded (`--setting-sources ""`): no global CLAUDE.md,
no hooks (the TTS announcer), no plugins leak into an estate turn.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time

from ..proc import Run, find_binary
from ..structured import extract_json
from ..task import Result, Task, ToolCall

NAME = "claude-code"
TIERS = {"fast": "haiku", "standard": "sonnet", "deep": "opus"}
_CANDIDATES = ("/opt/homebrew/bin/claude", "~/.local/bin/claude", "/usr/local/bin/claude")
_NEUTRAL = {"Read": "read", "Glob": "read", "Grep": "read", "LS": "read",
            "Edit": "edit", "Write": "edit", "MultiEdit": "edit", "NotebookEdit": "edit",
            "WebSearch": "web", "WebFetch": "web", "Bash": "shell",
            "Task": "subagent", "Agent": "subagent", "Skill": "skill"}


def is_bare(task: Task) -> bool:
    t = task.tools
    return not (t.files or t.write_paths or t.web or t.shell or t.mcp_servers
                or t.skills or t.subagents)


def neutral_tool_name(name: str) -> str:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        return f"mcp:{parts[1]}.{parts[2] if len(parts) > 2 else ''}"
    return _NEUTRAL.get(name, name)


def settings_for(task: Task) -> dict:
    t = task.tools
    allow, deny = [], []
    if t.files:
        allow += ["Read", "Glob", "Grep"]
    for p in t.read_paths:
        # Scoped rules pin the per-task grant (and document it) even where the
        # bare rules already allow reading; the dirs also go in with --add-dir.
        p = p.rstrip("/")
        allow += [f"Read(/{p}/**)", f"Glob(/{p}/**)", f"Grep(/{p}/**)"]
    for p in t.write_paths:
        p = p.rstrip("/")
        allow += [f"Edit(/{p}/**)", f"Write(/{p}/**)"]   # leading "//" = absolute
    if t.web:
        allow += ["WebSearch", "WebFetch"]
    if t.skills:
        allow.append("Skill")
    if t.subagents:
        allow.append("Task")
    for s in t.mcp_servers:
        allow.append(f"mcp__{s}")
    for d in t.mcp_deny:
        srv, _, tool = d.partition(".")
        deny.append(f"mcp__{srv}__{tool}" if tool else f"mcp__{srv}")
    if not t.shell:
        deny.append("Bash")
    return {"permissions": {"allow": allow, "deny": deny}}


def command(task: Task, model: str, exe: str) -> tuple[list[str], str]:
    """(argv, stdin text)."""
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose",
           "--model", model, "--setting-sources", "", "--no-session-persistence",
           "--strict-mcp-config", "--permission-mode", "default"]
    if task.on_text:
        cmd.append("--include-partial-messages")
    if is_bare(task):
        cmd += ["--tools", ""]
        if task.system:
            cmd += ["--system-prompt", task.system]
    else:
        cmd += ["--settings", json.dumps(settings_for(task))]
        if task.system:
            cmd += ["--append-system-prompt", task.system]
        for p in task.tools.read_paths:
            cmd += ["--add-dir", p]
        mcp = {"mcpServers": {
            name: {"type": "stdio", "command": spec["command"][0],
                   "args": list(spec["command"][1:]), "env": spec.get("env") or {}}
            for name, spec in task.tools.mcp_servers.items()}}
        cmd += ["--mcp-config", json.dumps(mcp)]
        if task.tools.subagents:
            cmd += ["--agents", json.dumps({
                n: {"description": a.get("description", n), "prompt": a.get("prompt", ""),
                    "model": TIERS.get(a.get("tier", "fast"), "haiku")}
                for n, a in task.tools.subagents.items()})]
    if task.effort:
        cmd += ["--effort", task.effort]
    if task.budget_usd:
        cmd += ["--max-budget-usd", str(task.budget_usd)]
    if task.max_turns:
        cmd += ["--max-turns", str(task.max_turns)]
    if task.schema:
        cmd += ["--json-schema", json.dumps(task.schema)]
    if task.images:
        cmd += ["--input-format", "stream-json"]
        content = []
        for img in task.images:
            if isinstance(img, (bytes, bytearray)):
                data = bytes(img)
                mt = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
            else:
                mt = mimetypes.guess_type(img)[0] or "image/png"
                with open(img, "rb") as f:
                    data = f.read()
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": mt,
                "data": base64.b64encode(data).decode()}})
        content.append({"type": "text", "text": task.prompt})
        stdin = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
    else:
        stdin = task.prompt
    return cmd, stdin


def env_for() -> dict:
    env = dict(os.environ)
    # Subscription auth only: an API key in the environment would make the
    # CLI bill the metered Console account instead.
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def run(task: Task, model: str | None = None) -> Result:
    model = model or task.model or TIERS.get(task.tier, "sonnet")
    res = Result(engine=NAME, model=model)
    exe = find_binary("claude", "CLAUDE_BIN", _CANDIDATES)
    if not exe:
        res.error = "claude binary not found"
        return res
    cmd, stdin = command(task, model, exe)
    t0 = time.monotonic()
    run_ = Run(cmd, env=env_for(), cwd=task.workdir or None, stdin_text=stdin,
               timeout=task.timeout_s)
    pending: dict[str, ToolCall] = {}
    final = None
    try:
        for line in run_.lines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "stream_event" and task.on_text:
                d = (ev.get("event") or {}).get("delta") or {}
                if d.get("type") == "text_delta" and d.get("text"):
                    task.on_text(d["text"])
            elif kind == "system" and ev.get("subtype") == "init":
                res.model = ev.get("model") or res.model
            elif kind == "assistant":
                for b in (ev.get("message") or {}).get("content") or []:
                    if b.get("type") == "tool_use" and b.get("name") != "StructuredOutput":
                        tc = ToolCall(neutral_tool_name(b.get("name", "")), b.get("input") or {})
                        pending[b.get("id")] = tc
                        res.tool_calls.append(tc)
            elif kind == "user":
                for b in (ev.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tc = pending.get(b.get("tool_use_id"))
                        if tc:
                            tc.ok = not b.get("is_error")
                            c = b.get("content")
                            if isinstance(c, list):
                                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                            tc.output = str(c or "")[:800]
                            if b.get("is_error"):
                                c = b.get("content")
                                tc.error = (c if isinstance(c, str) else json.dumps(c))[:300]
            elif kind == "result":
                final = ev
                break
    finally:
        rc = run_.close()
    res.latency_s = round(time.monotonic() - t0, 2)
    if final is None:
        res.error = "timeout" if run_.timed_out else f"no result (rc={rc})"
        res.raw_tail = run_.stderr_tail()
        return res
    res.text = final.get("result") or ""
    res.structured = final.get("structured_output")
    if task.schema and res.structured is None and not final.get("is_error"):
        res.structured = extract_json(res.text)   # older CLIs answer in the text
        if res.structured is None:
            res.error = "schema: no structured output"
    res.cost_usd = final.get("total_cost_usd")
    u = final.get("usage") or {}
    res.input_tokens = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
        + (u.get("cache_creation_input_tokens") or 0)
    res.output_tokens = u.get("output_tokens") or 0
    if final.get("is_error"):
        res.error = final.get("subtype") or "error"
        res.raw_tail = (res.text or run_.stderr_tail())[-600:]
    return res
