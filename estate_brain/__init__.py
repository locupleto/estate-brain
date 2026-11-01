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

__version__ = "0.1.3"


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


__all__ = ["Task", "ToolPolicy", "ToolCall", "Result", "run", "__version__"]
