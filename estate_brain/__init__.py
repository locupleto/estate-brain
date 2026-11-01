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

__version__ = "0.1.5"


def run(task: Task, engine: str | None = None, model: str | None = None,
        ledger: bool = True) -> Result:
    from . import engines
    from .ledger import record
    from .resolve import resolve
    engine, model = resolve(task, engine, model)
    try:
        impl = engines.get(engine)
    except ValueError as e:
        res = Result(engine=engine, model=model or "", error=str(e))
    else:
        res = impl.run(task, model)
    if ledger:
        record(task, res)
    return res


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

    def __init__(self, inner, task: Task, ledger: bool):
        self._inner, self._task, self._ledger, self._done = inner, task, ledger, False

    def __iter__(self):
        return iter(self._inner)

    @property
    def result(self) -> Result:
        return self._inner.result

    def close(self, grace: float = 0.0) -> Result:
        res = self._inner.close(grace)
        if self._ledger and not self._done:
            from .ledger import record
            record(self._task, res)
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
    from .resolve import resolve
    engine, model = resolve(task, engine, model)
    try:
        impl = engines.get(engine)
    except ValueError as e:
        return _Unstreamable(Result(engine=engine, model=model or "", error=str(e)))
    if not hasattr(impl, "stream"):
        return _Unstreamable(Result(engine=engine, model=model or "",
                                    error=f"engine {engine!r} cannot stream"))
    return _Recorded(impl.stream(task, model), task, ledger)


__all__ = ["Task", "ToolPolicy", "ToolCall", "Result", "run", "stream", "__version__"]
