"""The central brain registry: which profile (engine + models) each app uses.

The registry document is served by the estate's control plane (the Armory,
`GET <url>/api/brains/registry`) from one hand-edited file, and resolved
here, locally, so a call never waits on more than one small fetch a minute:

    {"version": "<sha>",
     "profiles": {"claude-sub": {"engine": "claude-code",
                                 "tiers": {"fast": "haiku", ...}}, ...},
     "defaults": {"profile": "claude-sub"},
     "assign": {"bots.JARVIS": {"profile": "claude-sub", "tier": "deep"}, ...}}

Where the registry lives is per host configuration, never a default in this
public code: `BRAIN_REGISTRY=<url>` (or `off`), else `registry` in
`~/.config/estate-brain/config.json`. No registry configured = Phase 2
behaviour exactly.

Every good fetch is cached (`~/.config/estate-brain/registry.cache.json`)
and the cache is used without a fetch for TTL seconds. When the registry
cannot be reached the cache is used however old (the control plane being
down must never stop a call; it only freezes the configuration), and a
failed fetch is not retried for TTL seconds either, so an outage costs one
short timeout a minute, not one per call.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from dataclasses import dataclass, field

TTL_S = 60.0
TIMEOUT_S = 1.5
TIERS = ("fast", "standard", "deep")


def _dir() -> str:
    return os.path.expanduser(os.environ.get("BRAIN_CONFIG_DIR") or "~/.config/estate-brain")


def url(environ: dict | None = None) -> str | None:
    v = (os.environ if environ is None else environ).get("BRAIN_REGISTRY")
    if v is None:
        try:
            with open(os.path.join(_dir(), "config.json")) as f:
                v = json.load(f).get("registry")
        except (OSError, ValueError, AttributeError):
            v = None
    if not v or str(v).lower() in ("off", "0", "no", "false"):
        return None
    return str(v).rstrip("/")


def _cache_path() -> str:
    return os.path.join(_dir(), "registry.cache.json")


def _read_cache() -> dict | None:
    try:
        with open(_cache_path()) as f:
            c = json.load(f)
        return c if isinstance(c, dict) else None
    except (OSError, ValueError):
        return None


def _write_cache(c: dict) -> None:
    try:
        os.makedirs(_dir(), exist_ok=True)
        tmp = _cache_path() + f".{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(c, f)
        os.replace(tmp, _cache_path())
    except OSError:
        pass


def fetch(base: str) -> dict:
    req = urllib.request.Request(base + "/api/brains/registry", headers={
        "X-Brain-Host": socket.gethostname().split(".")[0], "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        doc = json.load(r)
    if not isinstance(doc, dict) or "profiles" not in doc:
        raise ValueError("not a registry document")
    return doc


def current(environ: dict | None = None) -> tuple[dict | None, str]:
    """(registry document or None, source): source is "live" (fetched or
    fresh in cache), "cache" (registry unreachable, last good copy) or
    "none" (no registry configured, or never reached)."""
    base = url(environ)
    if not base:
        return None, "none"
    now = time.time()
    c = _read_cache() or {}
    same = c.get("url") == base
    doc = c.get("doc") if same else None
    if doc and now - c.get("fetched_at", 0) < TTL_S:
        return doc, "live"
    if same and now - c.get("failed_at", 0) < TTL_S:
        return (doc, "cache") if doc else (None, "none")
    try:
        doc = fetch(base)
    except Exception:
        c.update({"url": base, "failed_at": now})
        if not same:
            c.pop("doc", None)
        _write_cache(c)
        return (c.get("doc"), "cache") if c.get("doc") else (None, "none")
    _write_cache({"url": base, "doc": doc, "fetched_at": now, "failed_at": 0})
    return doc, "live"


@dataclass
class Choice:
    profile: str
    engine: str
    model: str | None   # None = the engine picks from the task's tier
    tier: str
    key: str | None     # a keyring reference ("keyring:<name>"), never a secret
    version: str
    # The profile's engine settings (api, base_url, key, tiers): what the
    # messages engine needs to know where to send the call.
    spec: dict = field(default_factory=dict)


def choose(doc: dict, app: str, role: str, tier: str, model: str | None,
           profile: str | None = None) -> Choice | None:
    """Resolve one call against a registry document. `profile` forces one
    (BRAIN_PROFILE). Returns None when nothing usable is assigned: an
    unknown or disabled profile falls through to the defaults, and a
    disabled default to None (the caller's built-in default)."""
    profiles = doc.get("profiles") or {}
    assign = doc.get("assign") or {}
    a = {}
    for key in ([f"{app}.{role}"] if app and role else []) + ([app] if app else []):
        if key in assign:
            a = assign[key] or {}
            break
    names = [n for n in (profile, a.get("profile"), (doc.get("defaults") or {}).get("profile")) if n]
    for name in names:
        p = profiles.get(name)
        if not p or p.get("enabled", True) is False or not p.get("engine"):
            continue
        t = a.get("tier") if a.get("tier") in TIERS else tier
        tiers = p.get("tiers") or {}
        if a.get("model"):
            m = a["model"]
        elif p["engine"] == "claude-code" and model:
            m = model                       # the caller's Claude alias stays with Claude
        else:
            m = tiers.get(t)
        spec = {k: p[k] for k in ("api", "base_url", "key", "provider", "tiers") if k in p}
        return Choice(name, p["engine"], m, t, p.get("key"), str(doc.get("version", "")), spec)
    return None


_ALIAS_TIER = {"haiku": "fast", "sonnet": "standard", "opus": "deep"}


def fallbacks(doc: dict, app: str, role: str, tier: str, model: str | None) -> list[Choice]:
    """The profiles to try, in order, when the chosen one fails before acting:
    the assignment's `fallback` (app.role, else app), else [defaults]
    fallback; a string or a list of profile names, [] = none. Disabled or
    unknown profiles are skipped. A caller's Claude alias picks the tier on a
    profile that is not claude-code (opus -> deep)."""
    assign = doc.get("assign") or {}
    names = None
    for key in ([f"{app}.{role}"] if app and role else []) + ([app] if app else []):
        if key in assign and "fallback" in (assign[key] or {}):
            names = assign[key]["fallback"]
            break
    if names is None:
        names = (doc.get("defaults") or {}).get("fallback") or []
    if isinstance(names, str):
        names = [names]
    t = _ALIAS_TIER.get((model or "").lower(), tier)
    out = []
    for name in names:
        c = choose({**doc, "assign": {}}, "", "", t, model, name)
        if c and c.profile == name:
            out.append(c)
    return out
