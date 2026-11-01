"""Which engine and model a task runs on.

Order, first match wins:

1. explicit `engine=` / `model=` arguments to `run()` (evals, experiments);
2. per-app environment: BRAIN_<APP>_ENGINE / BRAIN_<APP>_MODEL
   (app "photo-ocr" -> BRAIN_PHOTO_OCR_ENGINE);
3. estate-wide environment: BRAIN_ENGINE / BRAIN_MODEL;
4. the central registry (`registry.py`): the assignment for app.role, else
   app, else the defaults; BRAIN_<APP>_PROFILE / BRAIN_PROFILE force a
   profile by name. A profile whose engine this library does not have (or
   cannot stream, for `stream()`, or refuses the task: the messages engine
   has no built-in tools) is passed over, never half-used;
5. built-in default: claude-code, model from the task (`model`, else its tier).

A model only travels with the engine it was set for: an override that
changes the engine without naming a model leaves the model to that engine,
and a caller's Claude alias is kept only when the registry's profile is a
claude-code one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .task import Task

DEFAULT_ENGINE = "claude-code"


@dataclass
class Resolution:
    engine: str
    model: str | None
    source: str = "default"     # explicit | env | registry | registry-cache | default
    profile: str = ""
    tier: str | None = None     # the registry's tier, when it overrides the task's
    version: str = ""           # registry version the choice came from
    spec: dict | None = None    # the profile's engine settings (registry only)
    # What to try when this one fails before acting (registry only; run()
    # decides eligibility). Each is a Resolution with source "fallback".
    fallbacks: list = field(default_factory=list)


def _app_key(app: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in app.upper())


def _usable(engine: str, need_stream: bool, task: Task | None = None) -> bool:
    """The engine exists, can stream when asked to, and can run this task
    faithfully (an engine's `refuse(task)` names what it cannot do)."""
    from . import engines
    impl = engines.ENGINES.get(engine)
    if impl is None or (need_stream and not hasattr(impl, "stream")):
        return False
    if task is not None and hasattr(impl, "refuse") and impl.refuse(task):
        return False
    if task is not None and task.functions and engine != "messages":
        return False
    return True


def resolve_full(task: Task, engine: str | None = None, model: str | None = None,
                 environ: dict | None = None, need_stream: bool = False) -> Resolution:
    env = os.environ if environ is None else environ
    if engine:
        return Resolution(engine, model or task.model, "explicit")
    scopes = ([f"BRAIN_{_app_key(task.app)}_"] if task.app else []) + ["BRAIN_"]
    for prefix in scopes:
        e, m = env.get(prefix + "ENGINE"), env.get(prefix + "MODEL")
        if e or m:
            if e and e != DEFAULT_ENGINE:
                return Resolution(e, m or model, "env")
            return Resolution(DEFAULT_ENGINE, m or model or task.model, "env")
    forced = next((env[p + "PROFILE"] for p in scopes if env.get(p + "PROFILE")), None)
    from . import registry
    # An explicit environ (tests) sees a registry only when it names one.
    doc, src = registry.current(environ) if environ is None or "BRAIN_REGISTRY" in env \
        else (None, "none")
    if doc:
        c = registry.choose(doc, task.app, task.role, task.tier, model or task.model, forced)
        if c and _usable(c.engine, need_stream, task):
            fbs = [Resolution(f.engine, f.model, "fallback", f.profile,
                              f.tier if f.tier != task.tier else None, f.version, f.spec or None)
                   for f in registry.fallbacks(doc, task.app, task.role, task.tier, model or task.model)
                   if f.profile != c.profile]
            return Resolution(c.engine, c.model, "registry" if src == "live" else "registry-cache",
                              c.profile, c.tier if c.tier != task.tier else None, c.version,
                              getattr(c, "spec", None) or None, fbs)
    return Resolution(DEFAULT_ENGINE, model or task.model, "default")


def resolve(task: Task, engine: str | None = None, model: str | None = None,
            environ: dict | None = None) -> tuple[str, str | None]:
    r = resolve_full(task, engine, model, environ)
    return r.engine, r.model
