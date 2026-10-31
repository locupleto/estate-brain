"""Scoring a replay: deterministic checks, then an LLM judge against the
recorded production answer."""

from __future__ import annotations

import json
import re

from ..structured import extract_json, validate
from ..task import Result, Task, ToolPolicy
from .replay import infer_schema

_DOOR = re.compile(r"^\s*DOOR:\s*([a-z_]+)", re.I | re.M)
_USER_MSG = re.compile(r'Current User Message:\s*"?(.*?)(?:"\s*\n|\n===)', re.S)
_ROLE = re.compile(r"=== YOUR ROLE ===\s*#\s*([^\n]+)")


def _door(text: str, structured) -> str | None:
    m = _DOOR.search(text or "")
    if m:
        return m.group(1).lower()
    j = extract_json(text or "")
    if isinstance(j, dict) and isinstance(j.get("door"), str):
        return j["door"].lower()
    return None


def contract_check(case: dict, res: Result) -> tuple[bool, str]:
    c, text = case["contract"], res.text or ""
    if not res.ok:
        return False, f"error: {res.error}"
    if c == "wrapper-json":
        j = extract_json(text)
        if not isinstance(j, dict) or not isinstance(j.get("action"), str):
            # Production parity: the wrapper's fallback wraps plain text into a
            # text_response (a third of Jarvis's voice replies, 2026-10).
            if text.strip():
                return True, "plain (wrapper wraps it)"
            return False, "empty reply"
        if j["action"] == "text_response" and not str(j.get("message") or "").strip():
            return False, "empty message"
        return True, j["action"]
    if c == "wrapper-any":
        return (bool(text.strip()), "text" if text.strip() else "empty")
    if c == "door":
        d = _door(text, res.structured)
        return (d is not None, f"door={d}")
    if c == "structured":
        ref = case.get("reference_structured")
        val = res.structured if res.structured is not None else extract_json(text)
        if val is None:
            return False, "no structured output"
        errs = validate(val, infer_schema(ref)) if ref is not None else []
        return (not errs, "; ".join(errs[:2]) or "schema ok")
    if c == "bare-json":
        j = extract_json(text)
        try:
            ref = json.loads(case["reference_text"])
        except (TypeError, ValueError):
            ref = None
        if j is None:
            return False, "no JSON"
        errs = validate(j, infer_schema(ref)) if ref is not None else []
        return (not errs, "; ".join(errs[:2]) or "schema ok")
    return True, ""


def expectations(case: dict, res: Result, memory_changed: bool) -> dict:
    out = {}
    for e in case.get("expect", []):
        if e == "edit:memory":
            out[e] = memory_changed
        elif e.startswith("mcp:"):   # "mcp:a|mcp:b" = either server will do
            alts = [x if x.startswith("mcp:") else "mcp:" + x for x in e.split("|")]
            out[e] = any(t.name.startswith(a + ".") and t.ok is not False
                         for a in alts for t in res.tool_calls)
        elif e == "web":
            out[e] = any(t.name == "web" for t in res.tool_calls)
    return out


def door_match(case: dict, res: Result) -> bool | None:
    if case["contract"] != "door":
        return None
    return _door(case.get("reference_text") or "", None) == _door(res.text, res.structured)


JUDGE_SCHEMA = {
    "type": "object",
    "required": ["correctness", "persona", "format", "overall", "verdict", "notes"],
    "properties": {
        "correctness": {"type": "integer"}, "persona": {"type": "integer"},
        "format": {"type": "integer"}, "overall": {"type": "integer"},
        "verdict": {"type": "string", "enum": ["better", "same", "worse", "fail"]},
        "notes": {"type": "string"}}}

_JUDGE_SYSTEM = (
    "You grade an AI agent's answer for a personal estate of bots and apps. You "
    "get the task, a REFERENCE answer that production accepted, and a CANDIDATE "
    "answer from another model or engine for the same task. Score 1-5 each: "
    "correctness (facts, reasoning, did what was asked; live data may have moved "
    "since the reference, so judge plausibility and method, not exact numbers), "
    "persona (voice and character the task asks for), format (follows the reply "
    "contract: JSON action, DOOR verdict, schema; for wrapper-json a plain-text reply is acceptable, the wrapper wraps it), overall. verdict: better / same "
    "/ worse than the reference, or fail if unusable. The reference is one good "
    "answer, not the only one. The candidate ran in a SANDBOX: no shell, no mail, "
    "no live estate state, only read-only market/weather/news tools and its own "
    "memory files. Where the reference relied on what the sandbox lacks, grade "
    "how well the candidate coped (honest about limits, no invented facts, did "
    "the rest) rather than penalising the missing action itself. "
    "Be terse in notes (max 2 sentences).")


def _task_excerpt(case: dict, limit: int = 9000) -> str:
    p = case["prompt"]
    role = _ROLE.search(p)
    msg = _USER_MSG.search(p)
    parts = []
    if role:
        parts.append(f"Persona: {role.group(1).strip()}")
    if msg:
        parts.append("Request: " + msg.group(1).strip()[:4000])
    if case["contract"] == "structured" or not msg:
        parts.append("Task prompt (tail): " + p[-limit:])
    return "\n".join(parts)[:limit + 4500]


def judge(case: dict, res: Result, engine: str = "claude-code", model: str = "sonnet") -> dict:
    from .. import run
    ref = case.get("reference_structured")
    ref_s = json.dumps(ref, ensure_ascii=False) if ref is not None else (case.get("reference_text") or "")
    cand = json.dumps(res.structured, ensure_ascii=False) if res.structured is not None else (res.text or "")
    import time
    evidence = "\n".join(
        f"- {t.name} {json.dumps(t.input, ensure_ascii=False)[:160]} -> "
        f"{'ERROR ' + t.error[:150] if t.ok is False else t.output[:500]}"
        for t in res.tool_calls if t.name.startswith(("mcp:", "web", "read")))[:6000]
    prompt = (f"CONTRACT: {case['contract']}\n"
              f"REFERENCE recorded at: {case.get('recorded_at') or 'unknown'}\n"
              f"CANDIDATE replayed at: {time.strftime('%Y-%m-%dT%H:%M')} (live tools return TODAY's data)\n\n"
              f"TASK:\n{_task_excerpt(case)}\n\n"
              f"REFERENCE ANSWER:\n{ref_s[:7000]}\n\nCANDIDATE ANSWER:\n{cand[:7000]}\n\n"
              f"CANDIDATE TOOL RESULTS (evidence; figures traceable here are NOT invented):\n"
              f"{evidence or '(none)'}")
    r = run(Task(prompt=prompt, system=_JUDGE_SYSTEM, schema=JUDGE_SCHEMA,
                 tools=ToolPolicy(files=False), timeout_s=180), engine=engine, model=model,
            ledger=False)
    if not r.ok or not isinstance(r.structured, dict):
        return {"error": r.error or "no verdict", "cost_usd": r.cost_usd}
    return {**r.structured, "cost_usd": r.cost_usd}
