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
    # Directories, or single existing files. A path ending in "/**" is always a
    # directory; "*" alone = write anywhere the session can reach.
    write_paths: list[str] = field(default_factory=list)
    write_files: list[str] = field(default_factory=list)   # single files, existing or not
    web: bool = False
    shell: bool = False
    mcp_servers: dict[str, dict] = field(default_factory=dict)  # name -> {"command": [...], "env": {...}}
    mcp_deny: list[str] = field(default_factory=list)  # "server" or "server.tool"
    # Tool-level grants on a mounted server ("server.tool") for a server that
    # is not allowed whole; its spec still goes in mcp_servers.
    mcp_allow: list[str] = field(default_factory=list)
    skills: bool = False
    # name -> {"description", "prompt", "tier", "effort"?, "web": bool, "mcp": [server, ...]}
    subagents: dict[str, dict] = field(default_factory=dict)

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
    # Prior turns, oldest first, in the Messages-API shape: {"role": "user" |
    # "assistant", "content": str | [blocks]} (text, image, tool_use,
    # tool_result blocks). `prompt` is then the new user turn ("" when the
    # last entry already is one). The messages engine sends them as turns;
    # agent engines get a plain transcript above the prompt.
    messages: list = field(default_factory=list)
    # Caller-run tools (Messages-API tool schemas: name, description,
    # input_schema). The turn stops at the first tool call; the caller runs
    # it and sends the result back as the next turn. messages engine only.
    functions: list = field(default_factory=list)
    tools: ToolPolicy = field(default_factory=ToolPolicy)
    tier: str = "standard"
    effort: str | None = None
    model: str | None = None               # explicit engine model id; overrides tier
    budget_usd: float | None = None
    timeout_s: float = 300
    max_turns: int | None = None
    workspace_config: bool = False         # load the workspace's own agent config (its hooks)
    persist: bool = False                  # keep the engine's session transcript (the Armory
                                           # token ledger and the golden-case capture read them)
    app: str = ""                          # for the resolver and the ledger
    role: str = ""
    on_text: Callable[[str], Any] | None = None  # streaming text deltas (optional)
    # Engine-specific knobs that change no behaviour, only diagnostics, keyed
    # by engine name; an engine ignores the others' (claude-code: debug_file).
    engine_opts: dict = field(default_factory=dict)

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
    id: str = ""         # the call's id (messages engine: answer it with a tool_result)


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


def transcript(messages: list) -> str:
    """Prior turns as plain text, for engines that take one prompt. Non-text
    blocks are named, not rendered."""
    lines = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") if b.get("type") == "text" else f"[{b.get('type')}]"
                         for b in c if isinstance(b, dict))
        lines.append(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {c}")
    return "Conversation so far:\n" + "\n\n".join(lines) + "\n\n" if lines else ""
