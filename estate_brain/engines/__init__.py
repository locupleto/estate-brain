"""Engine registry. Each engine module exposes NAME and run(task, model)."""

from __future__ import annotations

from . import claude_code, messages, opencode

ENGINES = {claude_code.NAME: claude_code, opencode.NAME: opencode, messages.NAME: messages}


def get(name: str):
    try:
        return ENGINES[name]
    except KeyError:
        raise ValueError(f"unknown engine {name!r} (have: {', '.join(ENGINES)})") from None
