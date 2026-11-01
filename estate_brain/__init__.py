"""estate-brain: one provider-agnostic LLM layer for the estate.

    from estate_brain import Task, ToolPolicy, run
    result = run(Task(prompt="...", tier="fast"), engine="claude-code")

Callers describe the task (prompt, persona, schema, tool policy, tier, app)
and never name a provider. `resolve` picks the engine and model (explicit
arguments, then BRAIN_<APP>_* / BRAIN_* environment, then claude-code); every
call is appended to the ledger. The Armory registry joins the resolver in
Phase 3.
"""

from __future__ import annotations

from .task import Result, Task, ToolCall, ToolPolicy

__version__ = "0.1.8"


def run(task: Task, engine: str | None = None, model: str | None = None,
        ledger: bool = True) -> Result:
    from . import engines
    from .ledger import record
    from .resolve import resolve_full
    r = resolve_full(task, engine, model)
    task = _with_tier(task, r)
    try:
        impl = engines.get(r.engine)
    except ValueError as e:
        res = Result(engine=r.engine, model=r.model or "", error=str(e))
    else:
        task, err = _adapt(task, r.engine)
        res = Result(engine=r.engine, model=r.model or "", error=err) if err \
            else impl.run(task, r.model, **_engine_kw(r))
    if ledger:
        record(task, res, r)
    return res


def _engine_kw(r) -> dict:
    return {"profile": r.spec} if r.engine == "messages" and r.spec else {}


def _adapt(task: Task, engine: str):
    """Fit a task to an agent engine: prior turns become a transcript above
    the prompt; caller-run functions cannot be expressed (refused)."""
    if engine == "messages":
        return task, ""
    if task.functions:
        return task, f"engine {engine!r} cannot run caller functions (messages engine only)"
    if task.messages:
        from dataclasses import replace
        from .task import transcript
        return replace(task, prompt=transcript(task.messages) + task.prompt, messages=[]), ""
    return task, ""


def _with_tier(task: Task, r) -> Task:
    if r.tier and r.tier != task.tier:
        from dataclasses import replace
        return replace(task, tier=r.tier)
    return task


class _Unstreamable:
    """What `stream()` returns for an engine that cannot stream: no events,
    an error result."""

    def __init__(self, result: Result):
        self.result = result

    def __iter__(self):
        return iter(())

    def close(self, grace: float = 0.0) -> Result:
        return self.result


class _Recorded:
    """An engine stream that writes its ledger line when closed."""

    def __init__(self, inner, task: Task, ledger: bool, resolution=None):
        self._inner, self._task, self._ledger, self._done = inner, task, ledger, False
        self._resolution = resolution

    def __iter__(self):
        return iter(self._inner)

    @property
    def result(self) -> Result:
        return self._inner.result

    def close(self, grace: float = 0.0) -> Result:
        res = self._inner.close(grace)
        if self._ledger and not self._done:
            from .ledger import record
            record(self._task, res, self._resolution)
        self._done = True
        return res

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def stream(task: Task, engine: str | None = None, model: str | None = None,
           ledger: bool = True):
    """One streaming turn: iterate for raw Messages-API stream events (the
    de-facto shape; an engine that speaks another protocol translates into
    it), then `close()` for the Result — close early to abort (barge-in).
    Engines that cannot stream give no events and an error Result."""
    from . import engines
    from .resolve import resolve_full
    r = resolve_full(task, engine, model, need_stream=True)
    task = _with_tier(task, r)
    engine, model = r.engine, r.model
    try:
        impl = engines.get(engine)
    except ValueError as e:
        return _Unstreamable(Result(engine=engine, model=model or "", error=str(e)))
    if not hasattr(impl, "stream"):
        return _Unstreamable(Result(engine=engine, model=model or "",
                                    error=f"engine {engine!r} cannot stream"))
    task, err = _adapt(task, engine)
    if err:
        return _Unstreamable(Result(engine=engine, model=model or "", error=err))
    return _Recorded(impl.stream(task, model, **_engine_kw(r)), task, ledger, r)


__all__ = ["Task", "ToolPolicy", "ToolCall", "Result", "run", "stream", "__version__"]
