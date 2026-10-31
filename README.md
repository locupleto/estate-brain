# estate-brain

One provider-agnostic LLM layer for the estate's bots and agentic apps.
Callers describe *what* they need, a neutral `Task`; engines decide *how* a
given agent CLI or API does it. The plan and its decisions live in the vault:
`Projects/The-Armory/5-Future-Ideas/Provider-Agnostic-Brain-Plan.md`.

Stdlib only, Python ≥ 3.11, so every app can depend on it, including the
TCC-pinned voice assistant.

## Status

Phase 1 (golden-turn evals) — the task contract, two engines and the eval
suite. No production caller uses the library yet (Phase 2).

| Engine | What it drives | Auth |
|---|---|---|
| `claude-code` | the `claude` CLI, stripped of user settings | subscription login (API keys are removed from its env) |
| `opencode` | `opencode run`, fully isolated per run | provider keys from the environment |

## Use

```python
from estate_brain import Task, ToolPolicy, run

r = run(Task(prompt="Is the NYSE open on 2026-11-26?",
             workdir="/path/to/workspace",
             tools=ToolPolicy(write_paths=["/path/to/workspace/data/memory"],
                              mcp_servers={"exchange-calendar": {"command": ["python3", "server.py"]}})),
        engine="opencode", model="openai/gpt-6.1-sol")
print(r.ok, r.text, r.cost_usd, [t.name for t in r.tool_calls])
```

`ToolPolicy` is a fence, not a wish list: engines translate it to their own
permission model (Claude Code `--settings` rules, OpenCode `permission`),
deny everything else, and never run a shell unless `shell=True`.

## Golden-turn evals

Real production turns, replayed in a sandbox and scored against what
production answered.

```bash
python3 -m estate_brain.cli eval capture                 # pull cases from the mini (once)
python3 -m estate_brain.cli eval run --engine claude-code   # control: recorded Claude models
python3 -m estate_brain.cli eval run --engine opencode --model openai/gpt-6.1-sol
python3 -m estate_brain.cli eval report                  # latest scorecard
python3 -m estate_brain.cli eval compare CONTROL_DIR CAND_DIR   # per-case judge delta
python3 -m estate_brain.cli eval rescore RUN_DIR         # redo deterministic checks, no model calls
```

Useful flags on `eval run`: `--repeat 2` (one run per case is too noisy to
judge on), `--jobs 1` for providers with tight rate limits, and
`--into RUN_DIR --retry-failed` to re-run only the cases a provider rate
limit lost.

- `evals/golden.toml` lists the cases (transcript ids only — committed).
- Captured content (prompts, replies, memory snapshots) and run outputs live
  in `~/.estate-brain/` (`ESTATE_BRAIN_HOME`), never in git: they hold
  private chat and memory.
- Replays run in a throwaway sandbox: a copy of the bot's memory (the only
  writable place), read-only links to the repo checkout, read-only MCP
  servers only, no shell, no mail. Turns whose reference *acted* through the
  shell (calendar bookings, mail) are not golden cases.
- Scoring: run ok → reply contract (JSON action, `DOOR:` verdict, schema) →
  expected tool families (`edit:memory`, `mcp:<server>`) → an LLM judge
  (Claude Sonnet, fixed) grading the candidate against the reference.
- The Claude control replay, under the same sandbox conditions, is the bar
  every other engine/model is compared with — never the production reference
  itself. The control scores "worse" on some cases too (no live state in the
  sandbox, data drift since the recording); that is the noise floor.
- The JSON-action contract accepts plain text, because `wrapper_core.sh`
  wraps it into a `text_response` in production (about a third of Jarvis
  voice replies arrive that way).

## Tests

```bash
python3 -m unittest discover -s tests
```
