"""Structured replies for engines without native JSON-schema output:
prompt suffix, JSON extraction, and a small JSON-Schema subset validator
(type, required, properties, items, enum) — enough for the estate's schemas,
stdlib only."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def schema_instruction(schema: dict) -> str:
    return ("\n\nReply with ONLY one JSON value that validates against this "
            "JSON Schema — no prose, no code fences:\n" + json.dumps(schema))


def extract_json(text: str):
    """The last parseable JSON object/array in `text` (fenced first)."""
    s = (text or "").strip()
    for cand in [s] + [m.group(1).strip() for m in _FENCE.finditer(s)][::-1]:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
    # outermost brace span, scanning from the end
    for open_, close in (("{", "}"), ("[", "]")):
        end = s.rfind(close)
        while end != -1:
            start = s.rfind(open_, 0, end)
            while start != -1:
                try:
                    return json.loads(s[start:end + 1])
                except (json.JSONDecodeError, ValueError):
                    start = s.rfind(open_, 0, start)
            end = s.rfind(close, 0, end)
    return None


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "integer": int, "number": (int, float), "null": type(None)}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Errors (empty = valid) for the supported JSON-Schema subset."""
    errs: list[str] = []
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any(isinstance(value, _TYPES[x]) and not (x in ("integer", "number") and isinstance(value, bool))
                 for x in types if x in _TYPES)
        if not ok:
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing required '{k}'")
        for k, sub in (schema.get("properties") or {}).items():
            if k in value:
                errs.extend(validate(value[k], sub, f"{path}.{k}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            errs.extend(validate(v, schema["items"], f"{path}[{i}]"))
    return errs
