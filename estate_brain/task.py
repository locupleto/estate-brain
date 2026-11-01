"""The task contract: what a caller asks of a brain, and what comes back.

Callers describe WHAT they need (prompt, persona, schema, tool policy, tier);
engines decide HOW a given agent CLI or API does it. Nothing in here names a
provider, a model alias or a CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

TIERS = ("fast", "standard", "deep")
EFFORTS = ("low", "medium", "high")


@dataclass
class ToolPolicy:
    """Neutral tool fence. Engines translate it to their own permission
    model and must REFUSE a task they cannot express (never widen it).

    Paths are absolute. `read_paths` empty = the engine's default read scope
    (the workspace); `write_paths` empty = no writes at all."""

    files: bool = True          # file read/search tools at all (False + nothing else = a bare model call)
    read_paths: list[str] = field(default_factory=list)
    write_paths: list[str] = field(default_factory=list)
    web: bool = False
    shell: bool = False
    mcp_servers: dict[str, dict] = field(default_factory=dict)  # name -> {"command": [...], "env": {...}}
    mcp_deny: list[str] = field(default_factory=list)  # "server" or "server.tool"
    skills: bool = False
    subagents: dict[str, dict] = field(default_factory=dict)  # name -> {"description", "prompt", "tier"}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ToolPolicy":
        return cls(**(d or {}))


@dataclass
class Task:
    prompt: str
    system: str = ""                       # persona / system prompt ("" = none)
    workdir: str = ""                      # cwd for agent engines
    schema: dict | None = None             # JSON Schema for a structured reply
    images: list = field(default_factory=list)  # file paths or raw bytes (JPEG/PNG)
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    tier: str = "standard"
    effort: str | None = None
    model: str | None = None               # explicit engine model id; overrides tier
    budget_usd: float | None = None
    timeout_s: float = 300
    max_turns: int | None = None
    persist: bool = False                  # keep the engine's session transcript (the Armory
                                           # token ledger and the golden-case capture read them)
    app: str = ""                          # for the resolver and the ledger
    role: str = ""
    on_text: Callable[[str], Any] | None = None  # streaming text deltas (optional)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("on_text", None)
        d["images"] = [i for i in d["images"] if isinstance(i, str)]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d.pop("on_text", None)
        d["tools"] = ToolPolicy.from_dict(d.get("tools"))
        return cls(**d)


@dataclass
class ToolCall:
    name: str            # neutral: "read", "edit", "mcp:<server>.<tool>", "web", "shell", "subagent"
    input: dict = field(default_factory=dict)
    ok: bool | None = None
    error: str = ""
    output: str = ""     # first ~800 chars of the tool result


@dataclass
class Result:
    text: str = ""
    structured: Any = None
    engine: str = ""
    model: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    error: str = ""      # "" = success
    raw_tail: str = ""   # last bytes of engine stderr/output, for diagnosis

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return asdict(self)
