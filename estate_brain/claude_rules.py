"""Claude Code permission files -> a neutral ToolPolicy.

The estate wrote its tool fences as Claude Code `--settings` permission files
(`bots/bot_permissions*.json`, `config/permissions-*.json`) and its subagents
as Claude Code `--agents` JSON. This module reads both into the neutral
contract, so any engine can be handed the same fence.

Faithful or refused: a rule this module cannot express exactly raises
`PolicyError`; nothing is ever guessed at or widened. `roundtrip_problems()`
proves the claude-code engine writes the same fence back out.

Servers are taken from an MCP config file (the workspace's `.mcp.json`, or a
caller's narrower one) and only the ones the allowlist or a subagent names
are mounted: the rest could never be called anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

from .task import ToolPolicy

_READ = ("Read", "Glob", "Grep")
_WEB = ("WebSearch", "WebFetch")
_TIER_OF = {"haiku": "fast", "sonnet": "standard", "opus": "deep", "fable": "deep"}


class PolicyError(ValueError):
    """A rule that cannot be translated faithfully."""


def load_rules(path: str | Path, project_dir: str | Path | None = None) -> dict:
    text = Path(path).read_text()
    if project_dir is not None:
        text = text.replace("{{PROJECT_DIR}}", str(project_dir))
    return json.loads(text)


def _scoped(rule: str) -> tuple[str, str] | None:
    """'Edit(//abs/path/**)' -> ('Edit', '/abs/path/**'); a relative path is
    refused (it resolves against the settings file's own directory)."""
    if "(" not in rule or not rule.endswith(")"):
        return None
    head, _, inner = rule[:-1].partition("(")
    if not inner.startswith("//"):
        raise PolicyError(f"rule {rule!r}: only absolute (//) paths are translated")
    return head, inner[1:]


def mcp_specs(config: str | Path | None, names: list[str]) -> dict[str, dict]:
    """The named stdio servers from an MCP config file, as ToolPolicy specs.
    A name the file does not configure is skipped, as the CLI skips it."""
    try:
        servers = json.loads(Path(config).read_text()).get("mcpServers", {}) if config else {}
    except (OSError, ValueError):
        servers = {}
    out = {}
    for name in names:
        spec = servers.get(name)
        if not spec or not spec.get("command"):
            continue
        if spec.get("type", "stdio") != "stdio":
            raise PolicyError(f"MCP server {name!r} is not stdio")
        out[name] = {"command": [spec["command"], *spec.get("args", [])],
                     "env": dict(spec.get("env") or {})}
    return out


def agents_from(agents: dict) -> dict[str, dict]:
    out = {}
    for name, a in agents.items():
        tools = a.get("tools", [])
        unknown = [t for t in tools if t not in _WEB and not (t.startswith("mcp__") and "__" not in t[5:])]
        if unknown:
            raise PolicyError(f"subagent {name!r}: cannot translate tools {unknown}")
        web = [t for t in tools if t in _WEB]
        if web and len(web) != 2:
            raise PolicyError(f"subagent {name!r}: web tools must come as a pair, got {web}")
        model = a.get("model", "haiku")
        if model not in _TIER_OF:
            raise PolicyError(f"subagent {name!r}: unknown model {model!r}")
        sub = {"description": a.get("description", name), "prompt": a.get("prompt", ""),
               "tier": _TIER_OF[model], "web": bool(web),
               "mcp": [t[5:] for t in tools if t.startswith("mcp__")]}
        if a.get("effort"):
            sub["effort"] = a["effort"]
        extra = set(a) - {"description", "prompt", "tools", "model", "effort"}
        if extra:
            raise PolicyError(f"subagent {name!r}: cannot translate keys {sorted(extra)}")
        out[name] = sub
    return out


def policy_from(rules: dict, mcp_config: str | Path | None = None,
                agents: dict | None = None) -> ToolPolicy:
    """Translate one permission document (placeholders already substituted)."""
    perms = rules.get("permissions", {})
    allow, deny = list(perms.get("allow", [])), list(perms.get("deny", []))
    extra = set(perms) - {"allow", "deny"}
    if extra:
        raise PolicyError(f"cannot translate permission keys {sorted(extra)}")
    pol = ToolPolicy(files=False)
    seen = set(allow)
    reads: dict[str, set] = {}
    servers: list[str] = []
    bare_write = set()
    for rule in allow:
        if rule in _READ:
            continue
        if rule in _WEB:
            continue
        if rule in ("Edit", "Write"):
            bare_write.add(rule)
        elif rule == "Bash":
            pol.shell = True
        elif rule == "Skill":
            pol.skills = True
        elif rule in ("Agent", "Task"):
            if not agents:
                raise PolicyError(f"{rule!r} is allowed but no subagents are defined")
        elif rule.startswith("mcp__"):
            srv, _, tool = rule[5:].partition("__")
            if tool:
                pol.mcp_allow.append(f"{srv}.{tool}")
            if srv not in servers:
                servers.append(srv)
        else:
            sc = _scoped(rule)
            if sc is None:
                raise PolicyError(f"cannot translate allow rule {rule!r}")
            head, path = sc
            if head in _READ:
                if not path.endswith("/**"):
                    raise PolicyError(f"rule {rule!r}: a read grant must be a directory (/**)")
                reads.setdefault(path[:-3], set()).add(head)
            elif head in ("Edit", "Write"):
                if path.endswith("/**"):
                    if path not in pol.write_paths:
                        pol.write_paths.append(path)
                elif path not in pol.write_files:
                    pol.write_files.append(path)
            else:
                raise PolicyError(f"cannot translate allow rule {rule!r}")
    present = [r for r in _READ if r in seen]
    if present and len(present) != 3:
        raise PolicyError(f"Read, Glob and Grep must be allowed together, got {present}")
    pol.files = bool(present)
    web = [r for r in _WEB if r in seen]
    if web and len(web) != 2:
        raise PolicyError(f"WebSearch and WebFetch must be allowed together, got {web}")
    pol.web = bool(web)
    if bare_write:
        if bare_write != {"Edit", "Write"}:
            raise PolicyError(f"bare Edit and Write must be allowed together, got {sorted(bare_write)}")
        pol.write_paths.insert(0, "*")
    for path, heads in reads.items():
        if heads != set(_READ):
            raise PolicyError(f"scoped read of {path}: Read, Glob and Grep must come together")
        pol.read_paths.append(path)
    for rule in deny:
        if rule == "Bash":
            if pol.shell:
                raise PolicyError("Bash is both allowed and denied")
        elif rule in _WEB:
            if pol.web:
                raise PolicyError(f"{rule} is both allowed and denied")
        elif rule.startswith("mcp__"):
            srv, _, tool = rule[5:].partition("__")
            pol.mcp_deny.append(f"{srv}.{tool}" if tool else srv)
        else:
            raise PolicyError(f"cannot translate deny rule {rule!r}")
    if agents:
        if not ({"Agent", "Task"} & seen):
            raise PolicyError("subagents are defined but neither Agent nor Task is allowed")
        pol.subagents = agents_from(agents)
        for a in pol.subagents.values():
            for s in a["mcp"]:
                if s not in servers:
                    servers.append(s)
    pol.mcp_servers = mcp_specs(mcp_config, servers)
    return pol


def _norm(rules: list[str]) -> set[str]:
    # Edit() governs Write and Edit alike; the engine writes both for a directory.
    return {("Edit" + r[5:]) if r.startswith("Write(") else r for r in rules}


def roundtrip_problems(rules: dict, policy: ToolPolicy, settings: dict) -> list[str]:
    """Differences between an original permission document and the settings
    the claude-code engine emits for its translation. Not counted: Write(p)
    beside Edit(p) (same grant), web or Bash denied where the original only
    left it out (a -p turn cannot use an unallowed tool), an allow rule for a
    server that is not mounted (it has no tools to call)."""
    orig, got = rules["permissions"], settings["permissions"]
    mounted = set(policy.mcp_servers)
    def keep(r):
        if r.startswith("mcp__"):
            return r[5:].partition("__")[0] in mounted
        return True
    oa, ga = {r for r in _norm(orig.get("allow", [])) if keep(r)}, _norm(got.get("allow", []))
    od, gd = set(orig.get("deny", [])), set(got.get("deny", []))
    out = [f"allow lost: {r}" for r in sorted(oa - ga)]
    out += [f"allow added: {r}" for r in sorted(ga - oa) if r not in ("Task", "Agent") or
            not ({"Task", "Agent"} & set(orig.get("allow", [])))]
    out += [f"deny lost: {r}" for r in sorted(od - gd)]
    out += [f"deny added: {r}" for r in sorted(gd - od) if r not in ("Bash",) + _WEB]
    return out
