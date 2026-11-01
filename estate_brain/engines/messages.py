"""messages engine: a model's HTTP API, called directly (no agent CLI).

Two wire formats:

- `anthropic`: the Messages API (api.anthropic.com/v1/messages);
- `openai-compatible`: /chat/completions as spoken by OpenAI, Gemini's
  OpenAI endpoint, xAI, Perplexity, OpenRouter, Ollama, LM Studio and
  mlx-lm servers. Its stream is translated into Messages-API events, so a
  `stream()` caller sees one shape whatever the provider.

Where a call goes comes from the registry profile (`api`, `base_url`, `key`)
when there is one, else from the model id's prefix (claude-, gpt-/o<n>,
gemini-, grok-, sonar). Keys: a profile's `keyring:<name>` reference, else
the provider's usual keyring name, fetched from the estate keyring next to
the registry (`<registry>/api/keyring/values`, memory only, never logged or
written); else the provider's usual environment variable. A local base_url
needs none.

What it can run: a model call with a system prompt, prior turns
(`Task.messages`), images, a JSON-schema reply (schema in the prompt, the
reply extracted and validated, one retry; the same path as opencode) and
caller-run functions (`Task.functions`: the turn stops at the first tool
call and the caller sends the results back as the next turn). It has no
built-in tools, so a task that asks for files, web, shell, MCP, skills or
subagents is REFUSED (`refuse()`), never run without them; the resolver
passes such a profile over.

Stdlib only (urllib): the voice assistant's venv cannot take new packages.
`temperature` is never sent: Anthropic removed it, and an invented default
is what broke the Armory's AI view twice.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from ..structured import extract_json, schema_instruction, validate
from ..task import Result, Task, ToolCall

NAME = "messages"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 32000

# model-id prefix -> (api, base_url, keyring name, environment variable)
PROVIDERS = (
    ("claude", "anthropic", ANTHROPIC_URL, "anthropic_api_key", "ANTHROPIC_API_KEY"),
    ("gpt", "openai-compatible", "https://api.openai.com/v1", "openai_api_key", "OPENAI_API_KEY"),
    ("o1", "openai-compatible", "https://api.openai.com/v1", "openai_api_key", "OPENAI_API_KEY"),
    ("o3", "openai-compatible", "https://api.openai.com/v1", "openai_api_key", "OPENAI_API_KEY"),
    ("o4", "openai-compatible", "https://api.openai.com/v1", "openai_api_key", "OPENAI_API_KEY"),
    ("openai/", "openai-compatible", "https://api.openai.com/v1", "openai_api_key", "OPENAI_API_KEY"),
    ("gemini", "openai-compatible", "https://generativelanguage.googleapis.com/v1beta/openai",
     "google_api_key", "GEMINI_API_KEY"),
    ("grok", "openai-compatible", "https://api.x.ai/v1", "grok_api_key", "XAI_API_KEY"),
    ("sonar", "openai-compatible", "https://api.perplexity.ai", "perplexity_api_key", "PERPLEXITY_API_KEY"),
)
TIERS = {"fast": "claude-haiku-5-5", "standard": "claude-sonnet-5-5", "deep": "claude-opus-5-5"}
# OpenAI-compatible hosts known to accept stream_options.include_usage.
_USAGE_HOSTS = ("api.openai.com", "generativelanguage.googleapis.com", "api.x.ai", "openrouter.ai")

# USD per million tokens (input, output, cache read) at Anthropic list prices
# (2026-10-06); cache writes are 1.25x input (5-minute TTL). A model not
# listed reports cost None, never a guess.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}
_DATED = re.compile(r"-\d{8}$")          # claude-haiku-4-5-20251001 -> claude-haiku-4-5


def refuse(task: Task) -> str | None:
    """Why this engine cannot run `task` faithfully, or None."""
    t = task.tools
    wants = [name for name, on in (
        ("files", t.files), ("web", t.web), ("shell", t.shell),
        ("write access", bool(t.write_paths or t.write_files)),
        ("MCP servers", bool(t.mcp_servers or t.mcp_allow)), ("skills", t.skills),
        ("subagents", bool(t.subagents))) if on]
    if wants:
        return "the messages engine has no built-in tools; this task wants " + ", ".join(wants)
    return None


# --- where a call goes ---------------------------------------------------------

def route(model: str, profile: dict | None = None) -> dict:
    """{api, url, keyring, env} for one call."""
    p = profile or {}
    for prefix, api, url, kr, env in PROVIDERS:
        if model.startswith(prefix):
            break
    else:
        api, url, kr, env = "openai-compatible", "", "", ""
    if p.get("api"):
        api = p["api"]
        if api == "anthropic" and not p.get("base_url"):
            url, kr, env = ANTHROPIC_URL, "anthropic_api_key", "ANTHROPIC_API_KEY"
    if p.get("base_url"):
        url = p["base_url"].rstrip("/")
        if api == "anthropic" and not url.endswith("/messages"):
            url += "/v1/messages"
    if api == "openai-compatible" and url and not url.endswith("/chat/completions"):
        url += "/chat/completions"
    if p.get("key"):
        kr = p["key"][len("keyring:"):] if p["key"].startswith("keyring:") else ""
    if not url:
        raise ValueError(f"no endpoint for model {model!r}: give the profile an api and base_url")
    return {"api": api, "url": url, "keyring": kr, "env": env}


def _local(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local")


_keys: dict = {"at": 0.0, "values": {}}
_key_lock = threading.Lock()
KEY_TTL_S = 300.0
_resolver = None


def set_key_resolver(fn) -> None:
    """Let a host process supply keys itself (`fn(name) -> str | None`), e.g.
    the Armory from its own keyring rather than over HTTP to itself."""
    global _resolver
    _resolver = fn


def _keyring(name: str) -> str | None:
    if _resolver is not None:
        return _resolver(name)
    from .. import registry
    base = registry.url()
    if not base:
        return None
    with _key_lock:
        if time.monotonic() - _keys["at"] > KEY_TTL_S:
            try:
                req = urllib.request.Request(base + "/api/keyring/values",
                                             headers={"X-Brain-Host": socket.gethostname().split(".")[0]})
                with urllib.request.urlopen(req, timeout=3) as r:
                    vals = json.load(r)
                _keys.update(at=time.monotonic(), values=vals if isinstance(vals, dict) else {})
            except (OSError, ValueError):
                _keys["at"] = time.monotonic() - KEY_TTL_S + 30   # retry in 30 s, not per call
        return _keys["values"].get(name)


def api_key(rt: dict) -> str | None:
    key = _keyring(rt["keyring"]) if rt["keyring"] else None
    return key or (os.environ.get(rt["env"]) if rt["env"] else None)


# --- request bodies ------------------------------------------------------------

def _image(img) -> tuple[str, str]:
    data = img if isinstance(img, (bytes, bytearray)) else open(img, "rb").read()
    mt = ("image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else
          "image/gif" if data[:4] == b"GIF8" else
          "image/webp" if data[8:12] == b"WEBP" else "image/jpeg")
    return mt, base64.standard_b64encode(bytes(data)).decode()


def _turns(task: Task, prompt: str) -> list[dict]:
    """Prior turns plus this one, in the neutral (Messages-API) shape."""
    content: list | str = prompt
    if task.images:
        content = [{"type": "image", "source": {"type": "base64", "media_type": mt, "data": d}}
                   for mt, d in map(_image, task.images)]
        if prompt:
            content.append({"type": "text", "text": prompt})
    turns = [dict(m) for m in task.messages]
    if prompt or task.images:
        turns.append({"role": "user", "content": content})
    return turns


def anthropic_body(task: Task, model: str, prompt: str) -> dict:
    body = {"model": model, "max_tokens": task.max_output_tokens or MAX_TOKENS, "stream": True,
            "messages": _turns(task, prompt)}
    if task.system:
        body["system"] = task.system
    if task.effort:
        body["output_config"] = {"effort": task.effort}
    if task.functions:
        body["tools"] = task.functions
    return body


def _oa_content(c):
    if isinstance(c, str):
        return c
    out = []
    for b in c:
        if b.get("type") == "text":
            out.append({"type": "text", "text": b["text"]})
        elif b.get("type") == "image":
            s = b["source"]
            out.append({"type": "image_url", "image_url": {"url": f"data:{s['media_type']};base64,{s['data']}"}})
    return out


def openai_messages(task: Task, prompt: str) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": task.system}] if task.system else []
    for m in _turns(task, prompt):
        c = m["content"]
        blocks = c if isinstance(c, list) else []
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        results = [b for b in blocks if b.get("type") == "tool_result"]
        if m["role"] == "assistant" and uses:
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            out.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                for b in uses]})
        elif results:
            for b in results:
                rc = b.get("content")
                if isinstance(rc, list):
                    rc = "".join(x.get("text", "") for x in rc if isinstance(x, dict))
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(rc or "")})
            rest = [b for b in blocks if b.get("type") not in ("tool_result",)]
            if rest:
                out.append({"role": "user", "content": _oa_content(rest)})
        else:
            out.append({"role": m["role"], "content": _oa_content(c)})
    return out


def openai_body(task: Task, model: str, prompt: str, url: str) -> dict:
    body = {"model": model, "stream": True, "messages": openai_messages(task, prompt)}
    if task.max_output_tokens:
        body["max_tokens"] = task.max_output_tokens
    if (urlparse(url).hostname or "") in _USAGE_HOSTS:
        body["stream_options"] = {"include_usage": True}
    if task.functions:
        body["tools"] = [{"type": "function", "function": {
            "name": f["name"], "description": f.get("description", ""),
            "parameters": f.get("input_schema") or {"type": "object", "properties": {}}}}
            for f in task.functions]
    return body


# --- the stream ----------------------------------------------------------------

def _sse(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
         "function_call": "tool_use", "content_filter": "refusal"}


def translate_openai(frames):
    """OpenAI-compatible stream frames -> Messages-API stream events."""
    started, text_open, idx = False, False, -1
    tools: dict[int, int] = {}          # openai tool index -> our block index
    stop, usage = None, {}
    for f in frames:
        if not started:
            started = True
            yield {"type": "message_start", "message": {"id": f.get("id", ""), "type": "message",
                   "role": "assistant", "model": f.get("model", ""), "content": [],
                   "usage": {"input_tokens": 0, "output_tokens": 0}}}
        if f.get("usage"):
            usage = f["usage"]
        for ch in f.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                if not text_open:
                    idx += 1
                    text_open = True
                    yield {"type": "content_block_start", "index": idx,
                           "content_block": {"type": "text", "text": ""}}
                yield {"type": "content_block_delta", "index": idx,
                       "delta": {"type": "text_delta", "text": d["content"]}}
            for tc in d.get("tool_calls") or []:
                ti = tc.get("index", 0)
                fn = tc.get("function") or {}
                if ti not in tools:
                    if text_open:
                        yield {"type": "content_block_stop", "index": idx}
                        text_open = False
                    idx += 1
                    tools[ti] = idx
                    yield {"type": "content_block_start", "index": idx, "content_block": {
                        "type": "tool_use", "id": tc.get("id") or f"call_{ti}",
                        "name": fn.get("name", ""), "input": {}}}
                if fn.get("arguments"):
                    yield {"type": "content_block_delta", "index": tools[ti],
                           "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}
            if ch.get("finish_reason"):
                stop = _STOP.get(ch["finish_reason"], ch["finish_reason"])
    if not started:
        return
    for i in sorted(set(([idx] if text_open else []) + list(tools.values()))):
        yield {"type": "content_block_stop", "index": i}
    yield {"type": "message_delta", "delta": {"stop_reason": stop or "end_turn"},
           "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                     "output_tokens": usage.get("completion_tokens", 0)}}
    yield {"type": "message_stop"}


def cost(model: str, u: dict) -> float | None:
    p = PRICES.get(model) or PRICES.get(_DATED.sub("", model or ""))
    if not p:
        return None
    return round((u.get("input_tokens", 0) * p[0] + u.get("output_tokens", 0) * p[1]
                  + u.get("cache_read_input_tokens", 0) * p[2]
                  + u.get("cache_creation_input_tokens", 0) * p[0] * 1.25) / 1e6, 8)


class Stream:
    """One streaming call. Iterate for Messages-API events; `close()` ends it
    (early = abort) and fills `result`."""

    def __init__(self, task: Task, model: str | None = None, profile: dict | None = None,
                 prompt: str | None = None):
        self.task = task
        tiers = (profile or {}).get("tiers") or {}
        model = model or task.model or tiers.get(task.tier) or TIERS.get(task.tier, TIERS["standard"])
        self.result = Result(engine=NAME, model=model)
        self._t0 = time.monotonic()
        self._resp = None
        self._closed = False
        self._usage: dict = {}
        self._blocks: dict[int, dict] = {}
        self._stop = None
        self._gen = iter(())
        why = refuse(task)
        if why:
            self.result.error = why
            return
        try:
            rt = route(model, profile)
        except ValueError as e:
            self.result.error = str(e)
            return
        key = api_key(rt)
        if not key and not _local(rt["url"]):
            self.result.error = (f"no API key for {model}: keyring {rt['keyring'] or '-'} "
                                 f"and ${rt['env'] or '-'} are both empty")
            return
        prompt = task.prompt if prompt is None else prompt
        if rt["api"] == "anthropic":
            body = anthropic_body(task, model, prompt)
            headers = {"anthropic-version": ANTHROPIC_VERSION}
            if key:
                headers["x-api-key"] = key
        else:
            body = openai_body(task, model, prompt, rt["url"])
            headers = {"Authorization": f"Bearer {key}"} if key else {}
        headers["content-type"] = "application/json"
        req = urllib.request.Request(rt["url"], data=json.dumps(body).encode(), headers=headers)
        host = urlparse(rt["url"]).hostname or rt["url"]
        try:
            self._resp = urllib.request.urlopen(req, timeout=min(task.timeout_s or 300, 300))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            # "API Error: <status>" so runcli's transient sense (529/503) reads it.
            self.result.error = f"API Error: {e.code} from {host}"
            self.result.raw_tail = detail
            return
        except (OSError, ValueError) as e:
            self.result.error = f"connection to {host} failed: {e}"
            return
        frames = _sse(self._resp)
        self._gen = self._events(frames if rt["api"] == "anthropic" else translate_openai(frames))

    def __iter__(self):
        return self._gen

    def _events(self, events):
        deadline = self._t0 + (self.task.timeout_s or 1e9)
        try:
            for ev in events:
                kind = ev.get("type")
                if kind == "message_start":
                    m = ev.get("message") or {}
                    self.result.model = m.get("model") or self.result.model
                    self._usage.update(m.get("usage") or {})
                elif kind == "content_block_start":
                    cb = dict(ev.get("content_block") or {})
                    if cb.get("type") == "text":
                        cb.setdefault("text", "")
                    else:
                        cb["_json"] = ""
                    self._blocks[ev.get("index", 0)] = cb
                elif kind == "content_block_delta":
                    b = self._blocks.get(ev.get("index", 0))
                    d = ev.get("delta") or {}
                    if b is not None and d.get("type") == "text_delta":
                        b["text"] = b.get("text", "") + d.get("text", "")
                        if self.task.on_text and d.get("text"):
                            self.task.on_text(d["text"])
                    elif b is not None and d.get("type") == "input_json_delta":
                        b["_json"] = b.get("_json", "") + d.get("partial_json", "")
                elif kind == "message_delta":
                    self._stop = (ev.get("delta") or {}).get("stop_reason") or self._stop
                    for k, v in (ev.get("usage") or {}).items():
                        if v:
                            self._usage[k] = v
                elif kind == "error":
                    err = ev.get("error") or {}
                    self.result.error = f"API Error: {err.get('type', 'error')}: {err.get('message', '')}"[:300]
                yield ev
                if time.monotonic() > deadline:
                    self.result.error = "timeout"
                    return
        except Exception as e:      # noqa: BLE001 - see below
            # close() from another thread (a barge-in, a probe stopping at its
            # first delta) pulls the socket from under the read; http.client
            # then raises whatever it trips on (AttributeError on 3.14). That
            # is the requested end, not an error.
            if self._closed:
                return
            if not self.result.error:
                self.result.error = f"stream broke: {e.__class__.__name__}: {e}"

    def close(self, grace: float = 0.0) -> Result:
        if self._closed:
            return self.result
        self._closed = True
        res = self.result
        if self._resp is not None:
            try:
                self._resp.close()
            except OSError:
                pass
        res.latency_s = round(time.monotonic() - self._t0, 2)
        res.text = "".join(b.get("text", "") for _, b in sorted(self._blocks.items())
                           if b.get("type") == "text")
        for _, b in sorted(self._blocks.items()):
            if b.get("type") == "tool_use":
                try:
                    inp = json.loads(b.get("_json") or "{}")
                except json.JSONDecodeError:
                    inp = {"_unparsed": b.get("_json", "")}
                res.tool_calls.append(ToolCall("fn:" + b.get("name", ""), inp, id=b.get("id", "")))
        u = self._usage
        res.input_tokens = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
            + (u.get("cache_creation_input_tokens") or 0)
        res.output_tokens = u.get("output_tokens") or 0
        res.cost_usd = cost(res.model, u) if self._resp is not None else None
        if not res.error and self._resp is not None and self._stop is None and not self._blocks:
            res.error = "empty reply"
        if not res.error and self._stop == "refusal":
            res.error = "refusal"
        return res

    @property
    def stop_reason(self):
        return self._stop


def stream(task: Task, model: str | None = None, profile: dict | None = None) -> Stream:
    return Stream(task, model, profile)


def _once(task: Task, model, profile, prompt: str) -> Result:
    s = Stream(task, model, profile, prompt)
    try:
        for _ in s:
            pass
    finally:
        res = s.close()
    return res


def run(task: Task, model: str | None = None, profile: dict | None = None) -> Result:
    if not task.schema:
        return _once(task, model, profile, task.prompt)
    prompt = task.prompt + schema_instruction(task.schema)
    res = _once(task, model, profile, prompt)
    if res.ok:
        val = extract_json(res.text)
        errs = ["no JSON found"] if val is None else validate(val, task.schema)
        if errs:
            first = res
            res = _once(task, model, profile, prompt + "\n\nYour previous reply was not valid: "
                        + "; ".join(errs[:3]) + ". Reply again with ONLY the corrected JSON.")
            res.input_tokens += first.input_tokens
            res.output_tokens += first.output_tokens
            if first.cost_usd is not None and res.cost_usd is not None:
                res.cost_usd = round(res.cost_usd + first.cost_usd, 8)
            val = extract_json(res.text) if res.ok else None
            errs = ["no JSON found"] if val is None else validate(val, task.schema)
        if errs:
            res.error = res.error or "schema: " + "; ".join(errs[:3])
        else:
            res.structured = val
    return res
