# estate-brain

One provider-agnostic LLM layer for the estate's bots and agentic apps.
Callers describe *what* they need, a neutral `Task`; engines decide *how* a
given agent CLI or API does it. The plan and its decisions live in the vault:
`Projects/The-Armory/5-Future-Ideas/Provider-Agnostic-Brain-Plan.md`.

Stdlib only, Python ≥ 3.11, so every app can depend on it, including the
TCC-pinned voice assistant.

## Status

Phase 2 — production callers move onto the library one at a time, still on
Claude: Photo-OCR and the Scriptorium (2026-11-01), then the Bastion,
Observatory, Academy, the speech fast brain and (v0.1.6, `brain run`) the bot wrapper.
Phase 3 (v0.1.7): the central registry. Phase 4 (v0.1.8): the `messages`
engine, a model's HTTP API called directly; (v0.1.12) the fallback chain.

Apps pin it by tag, like excalicore:

```text
estate-brain @ git+https://github.com/locupleto/estate-brain@v0.1.0
```

## Choosing the engine

Callers never name a provider. `run()` resolves, in order: explicit
`engine=`/`model=` arguments, then `BRAIN_<APP>_ENGINE` / `BRAIN_<APP>_MODEL`
(app `photo-ocr` → `BRAIN_PHOTO_OCR_*`), then `BRAIN_ENGINE` / `BRAIN_MODEL`,
then the central registry (below), then `claude-code` with the task's model
or tier. A Claude alias never travels to another engine. Every call appends one metadata-only line (app,
engine, model, tokens, cost, latency — never prompt or reply text) to
`~/.estate-brain/ledger.jsonl`; `BRAIN_LEDGER=<path>|off` moves or disables it.
The line also says where the choice came from (`source`: explicit, env,
registry, registry-cache, default) and the registry `profile` and version.

### The registry

A host can follow one registry document served by a control plane
(`GET <url>/api/brains/registry`): named **profiles** (an engine plus a model
per tier, a keyring *reference* for its key, `enabled`) and **assignments**
(`app.role`, else `app`, else `defaults`). Point a host at it with
`BRAIN_REGISTRY=<url>` or `{"registry": "<url>"}` in
`~/.config/estate-brain/config.json`; `BRAIN_PROFILE` / `BRAIN_<APP>_PROFILE`
force a profile. The document is resolved locally (`estate_brain.registry`),
fetched at most once a minute and cached; when the control plane cannot be
reached the cached copy is used however old, and a failed fetch is not
retried for a minute. A profile whose engine this library does not have,
cannot stream for `stream()`, or refuses the task (the `messages` engine has
no built-in tools) is passed over. No registry configured is the
same as before.

| Engine | What it drives | Auth |
|---|---|---|
| `claude-code` | the `claude` CLI, stripped of user settings | subscription login (API keys are removed from its env) |
| `opencode` | `opencode run`, fully isolated per run | provider keys from the environment |
| `claude-code` + `key` | the same CLI on a metered key (the fallback `claude-api-cli`): the profile's `keyring:` key is put in its env, on purpose | the Console key |
| `messages` | a model's HTTP API: Anthropic Messages, or OpenAI-compatible `/chat/completions` (OpenAI, Gemini, xAI, Perplexity, OpenRouter, Ollama, LM Studio, mlx-lm) | the profile's `keyring:<name>` from the estate keyring next to the registry, else the provider's usual environment variable; none for a local `base_url` |

The `messages` engine has no tools of its own: it runs a model call with a
system prompt, prior turns (`Task.messages`), images, a JSON-schema reply
(validated, one retry) and caller-run functions (`Task.functions`: the turn
stops at a tool call, the caller answers with a `tool_result` turn). A task
that wants files, web, shell, MCP, skills or subagents is refused, never
run without them. Where a call goes comes from the profile (`api`,
`base_url`, `key`), else from the model id (claude-, gpt-, gemini-, grok-,
sonar). Its `stream()` yields Messages-API events for every provider (the
OpenAI stream is translated). Cost is computed from Anthropic list prices;
other providers report none. `temperature` is never sent.

### Fallback

`[defaults] fallback = [...]` (or `fallback =` on an assignment; `[]` = none)
names profiles to try, in order, when the chosen one fails **before
acting**: no tool call at all, and the failure is the provider's (overloaded,
5xx, usage or rate limit, login, unreachable). Never a timeout, a spend
ceiling, a refusal or a bad request. A fallback that cannot run the task
(the `messages` engine and a tool task) is skipped. Each attempt is its own
ledger line; the fallback's has `source: "fallback"`. Only `run()` falls
back: a stream has already been read by its caller.

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

## From the shell: `brain run`

For callers written in bash (the estate's bot wrapper):

```sh
python -m estate_brain run --spec spec.json --prompt-file prompt.txt   # reply on stdout
python -m estate_brain policy --spec spec.json                         # fence only, no model call
```

The spec names a workdir and, optionally, a Claude Code permission file
(`permissions_file`, with `{{PROJECT_DIR}}` substituted), an MCP config and an
`--agents` file. `estate_brain.claude_rules` translates them into a
`ToolPolicy`; a rule it cannot express exactly is refused (exit 64, no model
called), and `policy` checks that the claude-code engine writes the same
fence back out. Exit codes: 0 ok, 1 error, 3 timeout, 4 spend ceiling, 5
transient provider trouble, 64 refused, 143 stopped by SIGTERM (the engine's
process group goes with it). See `estate_brain/runcli.py`.

A persisted turn carries ONE user setting, `cleanupPeriodDays`, into the
engine: user settings are otherwise never loaded, and without it the CLI's
30-day default would govern the cleanup of the transcripts it keeps.

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
