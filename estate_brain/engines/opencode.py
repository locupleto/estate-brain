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

What a bot turn needs beyond that (2026-11-01):
- the profile's `key` is resolved and set as the provider's own key
  variable (the catalogue names it); a key that cannot be read refuses;
- the model catalogue is the engine's own copy of models.dev
  (~/.estate-brain/opencode-models.json, refreshed daily), because the
  isolated run may not fetch one and a stale copy hides new models;
- `budget_usd` is enforced from the per-step cost events: the run is
  stopped at the first step that crosses it;
- `persist` keeps the run's event stream as its transcript, under
  ~/.estate-brain/opencode-sessions/<day>/<session>.jsonl;
- a subagent's `tier` picks its model from the profile's tiers.
Not carried over: the workspace's Claude Code hooks (telemetry only today),
its Claude Code skills, and `effort`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.request

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


_CATALOGUE_URL = "https://models.dev/api.json"
# Used only when the catalogue is missing or does not name the variable.
_KEY_VARS = {"openai": ["OPENAI_API_KEY"], "anthropic": ["ANTHROPIC_API_KEY"],
             "google": ["GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"],
             "xai": ["XAI_API_KEY"], "openrouter": ["OPENROUTER_API_KEY"]}
_CATALOGUE_MAX_AGE_S = 24 * 3600


def _home() -> str:
    return os.path.expanduser(os.environ.get("ESTATE_BRAIN_HOME", "~/.estate-brain"))


def catalogue_path(now: float | None = None) -> str | None:
    """The engine's models.dev copy, refreshed when older than a day. A
    failed refresh keeps the old copy; with none at all, OpenCode's own
    cache; None when there is nothing."""
    path = os.path.join(_home(), "opencode-models.json")
    now = time.time() if now is None else now
    fresh = os.path.exists(path) and now - os.path.getmtime(path) < _CATALOGUE_MAX_AGE_S
    if not fresh and not os.environ.get("ESTATE_BRAIN_OFFLINE"):
        try:
            from .. import __version__
            # models.dev answers Python's default User-Agent with 403.
            req = urllib.request.Request(_CATALOGUE_URL,
                                         headers={"User-Agent": f"estate-brain/{__version__}"})
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
            json.loads(body)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, path)
        except Exception:      # offline, slow or a bad body: keep what there is
            pass
    if os.path.exists(path):
        return path
    cached = os.path.expanduser("~/.cache/opencode/models.json")
    return cached if os.path.exists(cached) else None


def _catalogue(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def check_model(model: str, cat: dict) -> str | None:
    """Why `model` (provider/id) cannot run, or None. An empty catalogue
    says nothing (OpenCode then decides)."""
    if not cat:
        return None
    prov, _, mid = model.partition("/")
    if prov not in cat:
        return f"provider {prov!r} is not in OpenCode's model catalogue"
    if mid not in (cat[prov].get("models") or {}):
        return f"model {model!r} is not in OpenCode's model catalogue"
    return None


def key_env(model: str, profile: dict | None, cat: dict) -> tuple[dict, str | None]:
    """The provider key variable(s) for a profile that names a key:
    ({VAR: value}, None), or ({}, why) when it cannot be resolved."""
    if not profile or not profile.get("key"):
        return {}, None
    from .. import keys
    value = keys.resolve(profile["key"])
    if not value:
        return {}, f"no API key for this profile ({profile['key']})"
    prov = model.partition("/")[0]
    names = (cat.get(prov) or {}).get("env") or _KEY_VARS.get(prov) or []
    if not names:
        return {}, f"no key variable known for provider {prov!r}"
    return {n: value for n in names}, None


def refuse(task: Task) -> str | None:
    """Why this engine cannot run `task` faithfully, or None. A spend
    ceiling is enforced (per step), hooks are telemetry, so nothing in a bot
    turn needs refusing today; caller-run functions are refused upstream."""
    return None


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


def config_for(task: Task, model: str, tiers: dict | None = None) -> dict:
    t = task.tools
    perm: dict = {"*": "deny"}
    if t.files:
        perm.update({"read": "allow", "glob": "allow", "grep": "allow", "list": "allow"})
    edit = {"*": "deny"}
    paths = [p[:-3] if p.endswith("/**") else p for p in t.write_paths if p != "*"]
    if "*" in t.write_paths:
        edit = "allow"
    else:
        for p in paths:
            for pat in _patterns(p, task.workdir):
                edit[pat] = "allow"
        for f in t.write_files:
            edit[f] = "allow"
            if task.workdir and f.startswith(task.workdir.rstrip("/") + "/"):
                edit[f[len(task.workdir.rstrip("/")) + 1:]] = "allow"
    perm["edit"] = edit
    ext = {"*": "deny"}
    for p in t.read_paths + paths + [os.path.dirname(f) for f in t.write_files]:
        if not (task.workdir and p.startswith(task.workdir.rstrip("/") + "/")):
            ext[f"{p.rstrip('/')}/**"] = "allow"
    perm["external_directory"] = ext
    perm["bash"] = "allow" if t.shell else "deny"
    perm["webfetch"] = perm["websearch"] = "allow" if t.web else "deny"
    perm["skill"] = "allow" if t.skills else "deny"
    perm["task"] = "allow" if t.subagents else "deny"
    narrowed = {a.partition(".")[0] for a in t.mcp_allow}
    for s in t.mcp_servers:
        if s not in narrowed:
            perm[f"{s}_*"] = "allow"
    for a in t.mcp_allow:
        srv, _, tool = a.partition(".")
        perm[f"{srv}_{tool}"] = "allow"
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
        sub_model = (tiers or {}).get(a.get("tier") or "")
        if sub_model:
            agents[n]["model"] = sub_model
    if agents:
        cfg["agent"] = agents
    return cfg


def env_for(task: Task, model: str, home: str, catalogue: str | None = None,
            extra: dict | None = None, tiers: dict | None = None) -> dict:
    env = dict(os.environ)
    for sub in ("config", "data", "state", "cache"):
        os.makedirs(os.path.join(home, sub), exist_ok=True)
    env.update({"XDG_CONFIG_HOME": os.path.join(home, "config"),
                "XDG_DATA_HOME": os.path.join(home, "data"),
                "XDG_STATE_HOME": os.path.join(home, "state"),
                "XDG_CACHE_HOME": os.path.join(home, "cache"),
                "OPENCODE_CONFIG_CONTENT": json.dumps(config_for(task, model, tiers))})
    env.update(extra or {})
    for f in _ISOLATION_FLAGS:
        env[f] = "1"
    # The isolated run may not fetch a catalogue: hand it the engine's copy.
    if catalogue:
        env["OPENCODE_MODELS_PATH"] = catalogue
    return env


def neutral_tool_name(name: str, servers) -> str:
    for s in servers:
        if name.startswith(f"{s}_"):
            return f"mcp:{s}.{name[len(s) + 1:]}"
    return _NEUTRAL.get(name, name)


def _transcript_path(session: str) -> str:
    day = time.strftime("%Y-%m-%d")
    d = os.path.join(_home(), "opencode-sessions", day)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{session or 'unknown-' + str(int(time.time()))}.jsonl")


def _run_once(task: Task, model: str, exe: str, prompt: str, env_kw: dict | None = None,
              spent: float = 0.0) -> Result:
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
    run_ = Run(cmd, env=env_for(task, model, home, **(env_kw or {})), cwd=task.workdir or None,
               stdin_text=None, timeout=task.timeout_s)
    texts: list[str] = []
    cost = 0.0
    events: list[str] = []
    session = ""
    try:
        for line in run_.lines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(line.rstrip("\n"))
            session = session or ev.get("sessionID") or ""
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
                if task.budget_usd is not None and spent + cost > task.budget_usd:
                    res.error = (f"budget: Exceeded USD budget ({spent + cost:.4f} > "
                                 f"{task.budget_usd:g})")
                    break
    finally:
        rc = run_.close()
        shutil.rmtree(home, ignore_errors=True)
        if task.persist and events:
            try:
                with open(_transcript_path(session), "w") as f:
                    f.write("\n".join(events) + "\n")
            except OSError:
                pass
    res.latency_s = round(time.monotonic() - t0, 2)
    res.cost_usd = round(cost, 6)
    res.text = texts[-1].strip() if texts else ""
    if run_.timed_out and not res.error:
        res.error = "timeout"
    elif not res.error and not texts:
        res.error = f"no reply (rc={rc})"
    if res.error:
        res.raw_tail = run_.stderr_tail()
    return res


def run(task: Task, model: str | None = None, profile: dict | None = None) -> Result:
    model = model or task.model
    if not model:
        return Result(engine=NAME, error="opencode needs an explicit provider/model")
    exe = find_binary("opencode", "OPENCODE_BIN", _CANDIDATES)
    if not exe:
        return Result(engine=NAME, model=model, error="opencode binary not found")
    catalogue = catalogue_path()
    cat = _catalogue(catalogue)
    why = check_model(model, cat)
    keyvars, why_key = key_env(model, profile, cat)
    if why or why_key:
        return Result(engine=NAME, model=model, error=why or why_key)
    env_kw = {"catalogue": catalogue, "extra": keyvars, "tiers": (profile or {}).get("tiers")}
    prompt = task.prompt + (schema_instruction(task.schema) if task.schema else "")
    res = _run_once(task, model, exe, prompt, env_kw)
    if task.schema and res.ok:
        val = extract_json(res.text)
        errs = ["no JSON found"] if val is None else validate(val, task.schema)
        if errs:
            retry = _run_once(task, model, exe, prompt + "\n\nYour previous reply was "
                              "not valid (" + "; ".join(errs[:3]) + "). Reply again with ONLY the JSON.",
                              env_kw, spent=res.cost_usd or 0.0)
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
