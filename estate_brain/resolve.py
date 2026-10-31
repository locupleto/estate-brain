"""Which engine and model a task runs on.

Phase 2 order (the Armory registry joins in Phase 3):

1. explicit `engine=` / `model=` arguments to `run()` (evals, experiments);
2. per-app environment: BRAIN_<APP>_ENGINE / BRAIN_<APP>_MODEL
   (app "photo-ocr" -> BRAIN_PHOTO_OCR_ENGINE);
3. estate-wide environment: BRAIN_ENGINE / BRAIN_MODEL;
4. built-in default: claude-code, model from the task (`model`, else its tier).

A model only travels with the engine it was set for: an env override that
changes the engine without naming a model leaves the model to that engine.
"""

from __future__ import annotations

import os

from .task import Task

DEFAULT_ENGINE = "claude-code"


def _app_key(app: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in app.upper())


def resolve(task: Task, engine: str | None = None, model: str | None = None,
            environ: dict | None = None) -> tuple[str, str | None]:
    env = os.environ if environ is None else environ
    if engine:
        return engine, model or task.model
    scopes = ([f"BRAIN_{_app_key(task.app)}_"] if task.app else []) + ["BRAIN_"]
    for prefix in scopes:
        e, m = env.get(prefix + "ENGINE"), env.get(prefix + "MODEL")
        if e or m:
            if e and e != DEFAULT_ENGINE:
                return e, m or model
            return DEFAULT_ENGINE, m or model or task.model
    return DEFAULT_ENGINE, model or task.model
