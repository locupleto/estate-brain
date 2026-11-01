"""claude-code engine: the `claude` CLI on the host's subscription login.

Translates the neutral Task into the flags the estate's harnesses already use
(--settings permission rules, --strict-mcp-config, --json-schema, --agents)
and reads the stream-json event log back into a Result.

Two modes:
- no built-in tools (none at all, or MCP servers only): `--tools ""` +
  `--system-prompt` replaces Claude Code's own system prompt, which exists to
  drive its built-in tools — a plain model call, or one with MCP tools only
  (the speech fast brain: the replaced prompt is what keeps it fast).
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
    """No tools of any kind."""
    return not (builtins_allowed(task) or task.tools.mcp_servers)


def builtins_allowed(task: Task) -> bool:
    """Does the fence allow any of Claude Code's own (non-MCP) tools?"""
    t = task.tools
    return bool(t.files or t.read_paths or t.write_paths or t.write_files or t.web or t.shell
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
        if p == "*":
            allow += ["Edit", "Write"]                     # anywhere the session reaches
            continue
        explicit_dir = p.endswith("/**")
        p = p[:-3] if explicit_dir else p.rstrip("/")
        if not explicit_dir and os.path.isfile(p):
            allow += [f"Edit(/{p})"]                       # one file
        else:
            allow += [f"Edit(/{p}/**)", f"Write(/{p}/**)"]   # leading "//" = absolute
    for p in t.write_files:
        allow.append(f"Edit(/{p})")                        # Edit() governs Write too
    if t.web:
        allow += ["WebSearch", "WebFetch"]
    if t.skills:
        allow.append("Skill")
    if t.subagents:
        allow += ["Task", "Agent"]
    narrowed = {a.partition(".")[0] for a in t.mcp_allow}
    for s in t.mcp_servers:
        if s not in narrowed:
            allow.append(f"mcp__{s}")
    for a in t.mcp_allow:
        srv, _, tool = a.partition(".")
        allow.append(f"mcp__{srv}__{tool}")
    for d in t.mcp_deny:
        srv, _, tool = d.partition(".")
        deny.append(f"mcp__{srv}__{tool}" if tool else f"mcp__{srv}")
    # What the fence leaves out is denied outright, not merely unallowed, so a
    # project or local settings file loaded with workspace_config cannot widen it.
    if t.shell:
        allow.append("Bash")
    else:
        deny.append("Bash")
    if not t.web:
        deny += ["WebSearch", "WebFetch"]
    out = {"permissions": {"allow": allow, "deny": deny}}
    if task.persist:
        keep = _user_cleanup_days()
        if keep is not None:
            out["cleanupPeriodDays"] = keep
    return out


def _user_cleanup_days():
    """The user's transcript retention, the ONE user setting carried into a
    turn. User settings are never loaded, and without this key the CLI's
    default (30 days) would govern the cleanup of the transcripts a
    persisted turn writes, which the Armory's token ledger reads."""
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    try:
        with open(os.path.join(base, "settings.json")) as f:
            v = json.load(f).get("cleanupPeriodDays")
    except (OSError, ValueError, AttributeError):
        return None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _agent(a: dict) -> dict:
    out = {"description": a.get("description", ""), "prompt": a.get("prompt", ""),
           "model": TIERS.get(a.get("tier", "fast"), "haiku")}
    if a.get("effort"):
        out["effort"] = a["effort"]
    tools = (["WebSearch", "WebFetch"] if a.get("web") else []) + \
        [f"mcp__{s}" for s in a.get("mcp", [])]
    if tools or "web" in a or "mcp" in a:
        out["tools"] = tools
    return out


def _mcp_config(task: Task) -> str:
    return json.dumps({"mcpServers": {
        name: {"type": "stdio", "command": spec["command"][0],
               "args": list(spec["command"][1:]), "env": spec.get("env") or {}}
        for name, spec in task.tools.mcp_servers.items()}})


def command(task: Task, model: str, exe: str, partial: bool = False) -> tuple[list[str], str]:
    """(argv, stdin text). `partial` streams raw Messages events."""
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose",
           "--model", model,
           # Never the user's settings (global CLAUDE.md, hooks, plugins); the
           # workspace's own project settings only when the task asks for them.
           "--setting-sources", "project,local" if task.workspace_config else "",
           "--strict-mcp-config", "--permission-mode", "default"]
    if not task.persist:
        cmd.append("--no-session-persistence")
    if task.on_text or partial:
        cmd.append("--include-partial-messages")
    if not builtins_allowed(task):
        cmd += ["--tools", ""]
        if task.system:
            cmd += ["--system-prompt", task.system]
        if task.tools.mcp_servers:
            cmd += ["--settings", json.dumps(settings_for(task)), "--mcp-config", _mcp_config(task)]
        elif task.persist:
            cmd += ["--settings", json.dumps(settings_for(task))]
    else:
        cmd += ["--settings", json.dumps(settings_for(task))]
        if task.system:
            cmd += ["--append-system-prompt", task.system]
        for p in task.tools.read_paths:
            cmd += ["--add-dir", p]
        cmd += ["--mcp-config", _mcp_config(task)]
        if task.tools.subagents:
            cmd += ["--agents", json.dumps({
                n: _agent(a) for n, a in task.tools.subagents.items()})]
    if task.effort:
        cmd += ["--effort", task.effort]
    if task.budget_usd:
        cmd += ["--max-budget-usd", str(task.budget_usd)]
    if task.max_turns:
        cmd += ["--max-turns", str(task.max_turns)]
    if task.schema:
        cmd += ["--json-schema", json.dumps(task.schema)]
    debug_file = (task.engine_opts.get(NAME) or {}).get("debug_file")
    if debug_file:
        cmd += ["--debug-file", debug_file]
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


def env_for(api_key: str | None = None) -> dict:
    env = dict(os.environ)
    # Subscription auth only: an API key in the environment would make the
    # CLI bill the metered Console account instead. The one exception is a
    # profile that names a key (the metered fallback, `claude-api-cli`):
    # then the CLI runs on that key, on purpose.
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


class Stream:
    """One streaming turn. Iterate it for the raw Messages stream events
    (message_start, content_block_delta, message_stop ... — several messages
    when tools run); `close()` ends the turn at any point (a barge-in kills
    the CLI and its MCP children) and fills `result`."""

    def __init__(self, task: Task, model: str | None = None, partial: bool = True,
                 profile: dict | None = None):
        self.task = task
        model = model or task.model or TIERS.get(task.tier, "sonnet")
        self.result = Result(engine=NAME, model=model)
        self._final = None
        self._run = None
        self._closed = False
        self._t0 = time.monotonic()
        exe = find_binary("claude", "CLAUDE_BIN", _CANDIDATES)
        if not exe:
            self.result.error = "claude binary not found"
            self._gen = iter(())
            return
        api_key = None
        if profile and profile.get("key"):
            from .. import keys
            api_key = keys.resolve(profile["key"])
            if not api_key:
                # Never fall back to the subscription here: this profile exists
                # to run when the subscription cannot.
                self.result.error = f"no API key for this profile ({profile['key']})"
                self._gen = iter(())
                return
        cmd, stdin = command(task, model, exe, partial=partial)
        self._run = Run(cmd, env=env_for(api_key), cwd=task.workdir or None, stdin_text=stdin,
                        timeout=task.timeout_s)
        self._pending: dict[str, ToolCall] = {}
        self._gen = self._events()

    def __iter__(self):
        return self._gen

    def _events(self):
        res = self.result
        for line in self._run.lines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "stream_event":
                inner = ev.get("event")
                if isinstance(inner, dict):
                    if self.task.on_text:
                        d = inner.get("delta") or {}
                        if d.get("type") == "text_delta" and d.get("text"):
                            self.task.on_text(d["text"])
                    yield inner
            elif kind == "system" and ev.get("subtype") == "init":
                res.model = ev.get("model") or res.model
            elif kind == "assistant":
                for b in (ev.get("message") or {}).get("content") or []:
                    if b.get("type") == "tool_use" and b.get("name") != "StructuredOutput":
                        tc = ToolCall(neutral_tool_name(b.get("name", "")), b.get("input") or {})
                        self._pending[b.get("id")] = tc
                        res.tool_calls.append(tc)
            elif kind == "user":
                for b in (ev.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tc = self._pending.get(b.get("tool_use_id"))
                        if tc:
                            tc.ok = not b.get("is_error")
                            c = b.get("content")
                            if isinstance(c, list):
                                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                            tc.output = str(c or "")[:800]
                            if b.get("is_error"):
                                tc.error = tc.output[:300]
            elif kind == "result":
                self._final = ev
                return

    def close(self, grace: float = 0.0) -> Result:
        """End the turn and fill `result`. `grace` > 0: a caller that stopped
        reading after the last message gives the CLI that long to print its
        closing result line (tokens, cost) before it is killed; 0 kills at
        once (a barge-in)."""
        if self._closed:
            return self.result
        self._closed = True
        res = self.result
        if self._run is None:
            return res
        if grace > 0 and self._final is None:
            import threading
            drain = threading.Thread(target=lambda: [None for _ in self._gen], daemon=True)
            drain.start()
            drain.join(grace)
        rc = self._run.close()
        res.latency_s = round(time.monotonic() - self._t0, 2)
        final = self._final
        if final is None:
            if not res.error:
                res.error = "timeout" if self._run.timed_out else f"no result (rc={rc})"
            res.raw_tail = self._run.stderr_tail()
            return res
        res.text = final.get("result") or ""
        res.structured = final.get("structured_output")
        if self.task.schema and res.structured is None and not final.get("is_error"):
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
            res.raw_tail = (res.text or self._run.stderr_tail())[-600:]
        return res


def stream(task: Task, model: str | None = None, profile: dict | None = None) -> Stream:
    return Stream(task, model, profile=profile)


def run(task: Task, model: str | None = None, profile: dict | None = None) -> Result:
    s = Stream(task, model, partial=False, profile=profile)
    try:
        for _ in s:
            pass
    finally:
        res = s.close()
    return res
