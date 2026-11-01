"""Provider keys for engines that need one (the messages engine, a metered
claude-code profile).

A profile names a key as `keyring:<name>`; the value comes from the estate
keyring next to the registry (`<registry>/api/keyring/values`, held in
memory for KEY_TTL_S, never logged or written), or from a resolver the host
process installs (`set_key_resolver`: the Armory reads its own keyring, the
voice assistant passes the key it already holds).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request

KEY_TTL_S = 300.0
_keys: dict = {"at": 0.0, "values": {}}
_lock = threading.Lock()
_resolver = None


def set_key_resolver(fn) -> None:
    """Let a host process supply keys itself (`fn(name) -> str | None`)."""
    global _resolver
    _resolver = fn


def keyring(name: str) -> str | None:
    if not name:
        return None
    if _resolver is not None:
        return _resolver(name)
    from . import registry
    base = registry.url()
    if not base:
        return None
    with _lock:
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


def ref_name(ref: str | None) -> str:
    """'keyring:openai_api_key' -> 'openai_api_key'; anything else -> ''."""
    return ref[len("keyring:"):] if ref and ref.startswith("keyring:") else ""


def resolve(ref: str | None, env: str = "") -> str | None:
    return keyring(ref_name(ref)) or (os.environ.get(env) if env else None)
