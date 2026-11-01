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

import re

from .task import Result, Task, ToolCall, ToolPolicy

__version__ = "0.1.14"


# Failures worth one more try on a fallback profile: the provider or the
# subscription could not serve the call (overloaded, outage, usage or rate
# limit, login). Never a timeout, a spend ceiling, a refusal or a bad request.
_FALLBACK_WORTHY = re.compile(
    r"overloaded|API Error: 5\d\d|\b(500|502|503|529)\b|Internal server error|"
    r"usage limit|limit reached|hit your limit|rate.?limit|\b429\b|"
    r"not logged in|OAuth token|authentication_error|invalid api key|\b401\b|"
    r"claude binary not found|connection to .* failed|stream broke", re.I)


def fallback_worthy(res: Result) -> bool:
    """A failed call that may be retried elsewhere: nothing acted (no tool
    call at all, so nothing can run twice) and the failure is the provider's."""
    if res.ok or res.tool_calls or res.error in ("timeout", "refusal"):
        return False
    if "budget" in res.error:
        return False
    return bool(_FALLBACK_WORTHY.search(f"{res.error}\n{res.text}\n{res.raw_tail}"))


def _run_one(task: Task, r) -> Result:
    from . import engines
    try:
        impl = engines.get(r.engine)
    except ValueError as e:
        return Result(engine=r.engine, model=r.model or "", error=str(e))
    t, err = _adapt(_with_tier(task, r), r.engine)
    if err:
        return Result(engine=r.engine, model=r.model or "", error=err)
    return impl.run(t, r.model, **_engine_kw(r))


def run(task: Task, engine: str | None = None, model: str | None = None,
        ledger: bool = True) -> Result:
    """One call. When the registry's profile fails before acting (see
    `fallback_worthy`), the registry's fallback profiles are tried in order;
    each attempt is its own ledger line (the fallback's with source
    "fallback"). Only `run()` falls back: a stream has already been read."""
    from .ledger import record
    from .resolve import _usable, resolve_full
    r = resolve_full(task, engine, model)
    res = _run_one(task, r)
    if ledger:
        record(_with_tier(task, r), res, r)
    for fb in r.fallbacks:
        if not fallback_worthy(res):
            break
        if not _usable(fb.engine, False, task):
            continue                      # e.g. a tool task cannot run on the messages engine
        first_error = res.error
        res = _run_one(task, fb)
        if ledger:
            record(_with_tier(task, fb), res, fb)
        if not res.ok:
            res.error = f"{res.error} (fallback {fb.profile}, after: {first_error})"[:400]
    return res


def _engine_kw(r) -> dict:
    # The messages and opencode engines need their profile (where to send
    # the call, which key, the tier models); a claude-code profile only when
    # it carries a key (the metered CLI).
    if r.spec and (r.engine in ("messages", "opencode")
                   or (r.engine == "claude-code" and r.spec.get("key"))):
        return {"profile": r.spec}
    return {}


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


__all__ = ["Task", "ToolPolicy", "ToolCall", "Result", "run", "stream", "fallback_worthy",
           "__version__"]
