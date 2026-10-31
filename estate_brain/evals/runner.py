"""Run the golden cases under one engine/model and write a scorecard."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import run as brain_run
from .cases import RUNS, load_case, load_golden
from .replay import build, memory_digest
from .score import contract_check, door_match, expectations, judge

CLAUDE_ALIASES = {"claude-opus": "opus", "claude-sonnet": "sonnet", "claude-haiku": "haiku"}


def _recorded_alias(model: str | None) -> str | None:
    for prefix, alias in CLAUDE_ALIASES.items():
        if model and model.startswith(prefix):
            return alias
    return None


def run_case(case_id: str, engine: str, model: str | None, judge_model: str,
             out_dir: Path, rep: int = 1) -> dict:
    case = load_case(case_id)
    use_model = model or (_recorded_alias(case.get("model")) if engine == "claude-code" else None)
    task, sandbox = build(case)
    before = memory_digest(sandbox)
    try:
        res = brain_run(task, engine=engine, model=use_model, ledger=False)
        changed = memory_digest(sandbox) != before
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
    ok, why = contract_check(case, res)
    row = {
        "case": case_id, "rep": rep, "contract": case["contract"], "engine": engine, "model": res.model,
        "ok": res.ok, "error": res.error, "contract_ok": ok, "contract_note": why,
        "expect": expectations(case, res, changed), "door_match": door_match(case, res),
        "memory_changed": changed,
        "tools": [t.name for t in res.tool_calls],
        "denied": sum(1 for t in res.tool_calls if t.ok is False),
        "latency_s": res.latency_s, "cost_usd": res.cost_usd,
        "tokens": {"input": res.input_tokens, "output": res.output_tokens},
        "reference_tokens": case.get("tokens"), "reference_model": case.get("model"),
        "text": res.text, "structured": res.structured, "raw_tail": res.raw_tail,
    }
    row["judge"] = judge(case, res, model=judge_model) if res.ok else {"error": "skipped"}
    (out_dir / f"{case_id}@{rep}.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    return row


def run_suite(engine: str, model: str | None, only: list[str] | None = None,
              jobs: int = 3, judge_model: str = "sonnet", label: str = "",
              repeat: int = 1, into: str | None = None, retry_failed: bool = False) -> Path:
    ids = [(c["id"], r) for r in range(1, repeat + 1)
           for c in load_golden() if not only or c["id"] in only]
    tag = label or f"{engine}-{(model or 'recorded').replace('/', '_')}"
    if into:
        out_dir = Path(into)
        if retry_failed:
            failed = {(r["case"], r.get("rep", 1)) for r in _rows(out_dir) if is_infra_error(r)}
            ids = [i for i in ids if i in failed]
    else:
        out_dir = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(lambda i: _safe(i[0], engine, model, judge_model, out_dir, i[1]), ids))
    write_summary(out_dir)
    return out_dir


def _rows(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*@*.json"))]


def write_summary(out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("text", "structured", "raw_tail")}
         for r in _rows(out_dir)], indent=1, ensure_ascii=False))


_INFRA = ("rate-limit", "rate_limit", "429", "overloaded", "529")


def is_infra_error(row: dict) -> bool:
    """Provider capacity failures say nothing about the model's quality."""
    return any(k in (row.get("error") or "") for k in _INFRA)


def _safe(case_id, engine, model, judge_model, out_dir, rep=1):
    for attempt in range(4):
        try:
            row = run_case(case_id, engine, model, judge_model, out_dir, rep)
        except Exception as e:  # one broken case must not sink the suite
            row = {"case": case_id, "rep": rep, "engine": engine, "ok": False,
                   "error": f"harness: {e}", "contract_ok": False, "judge": {}}
        if not is_infra_error(row) or attempt == 3:
            break
    if not (out_dir / f"{case_id}@{rep}.json").exists() or row.get("error", "").startswith("harness"):
        (out_dir / f"{case_id}@{rep}.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
        time.sleep(60 * (attempt + 1))   # back off: 1, 2, 3 minutes
    j = row.get("judge") or {}
    print(f"  {case_id:28} contract={'ok ' if row.get('contract_ok') else 'BAD'} "
          f"overall={j.get('overall', '-')} verdict={j.get('verdict', '-'):6} "
          f"{row.get('latency_s', '-')}s ${row.get('cost_usd')}  {row.get('error', '')[:60]}",
          flush=True)
    return row


def report(run_dir: Path) -> str:
    rows = json.loads((run_dir / "summary.json").read_text())
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    reps = max(len(v) for v in by_case.values())
    lines = [f"Run {run_dir.name}  ({reps} run(s) per case)", "",
             f"{'case':28} {'contract':8} {'expect':16} {'judge':11} {'verdicts':16} {'secs':>6} {'cost':>7}"]
    all_overall, verdicts, cost, n_contract, n_runs = [], {}, 0.0, 0, 0
    n_infra = sum(is_infra_error(r) for r in rows)
    for case, rs in by_case.items():
        scores = [(r.get("judge") or {}).get("overall") for r in rs]
        scores = [x for x in scores if isinstance(x, int)]
        all_overall += scores
        c_ok = sum(bool(r.get("contract_ok")) for r in rs)
        n_contract += c_ok
        n_runs += len(rs)
        exp: dict[str, int] = {}
        for r in rs:
            for k, v in (r.get("expect") or {}).items():
                exp[k] = exp.get(k, 0) + bool(v)
            if r.get("door_match") is not None:
                exp["door"] = exp.get("door", 0) + bool(r["door_match"])
        exp_s = " ".join(f"{k.split('|')[0].split(':')[-1][:8]}{v}/{len(rs)}" for k, v in exp.items()) or "-"
        vs: dict[str, int] = {}
        for r in rs:
            v = (r.get("judge") or {}).get("verdict")
            if v:
                vs[v] = vs.get(v, 0) + 1
                verdicts[v] = verdicts.get(v, 0) + 1
        judge_s = (f"{sum(scores) / len(scores):.1f}" + (f" ({min(scores)}-{max(scores)})" if len(scores) > 1 else "")) if scores else "-"
        secs = [r.get("latency_s") for r in rs if isinstance(r.get("latency_s"), (int, float))]
        c = sum(r.get("cost_usd") or 0 for r in rs)
        cost += c
        lines.append(f"{case:28} {c_ok}/{len(rs):<6} {exp_s:16} {judge_s:11} "
                     f"{','.join(f'{k[:4]}{v}' for k, v in vs.items()):16} "
                     f"{(sum(secs) / len(secs)) if secs else 0:>6.1f} {c:>7.3f}")
    avg = sum(all_overall) / len(all_overall) if all_overall else 0
    lines += ["", f"runs {n_runs} · contract ok {n_contract}/{n_runs} · judge mean {avg:.2f} · "
                  f"verdicts {verdicts} · engine cost ${cost:.2f}"
                  + (f" · {n_infra} run(s) lost to provider rate limits" if n_infra else "")]
    return "\n".join(lines)


def rescore(out_dir: Path) -> None:
    """Recompute the deterministic checks on stored results (no model calls).
    The judge's scores are kept as they are."""
    from ..task import Result, ToolCall
    from .cases import load_case
    for p in sorted(out_dir.glob("*@*.json")):
        row = json.loads(p.read_text())
        if is_infra_error(row) or row.get("error", "").startswith("harness"):
            continue
        res = Result(text=row.get("text") or "", structured=row.get("structured"),
                     error=row.get("error") or "",
                     tool_calls=[ToolCall(n, ok=True) for n in row.get("tools", [])])
        case = load_case(row["case"])
        row["contract_ok"], row["contract_note"] = contract_check(case, res)
        row["expect"] = expectations(case, res, row.get("memory_changed", False))
        p.write_text(json.dumps(row, indent=1, ensure_ascii=False))
    write_summary(out_dir)


def compare(base_dir: Path, cand_dir: Path) -> str:
    """Per-case judge means, candidate minus base (the control)."""
    def means(d):
        out: dict[str, list] = {}
        for row in json.loads((d / "summary.json").read_text()):
            o = (row.get("judge") or {}).get("overall")
            out.setdefault(row["case"], [])
            if isinstance(o, int):
                out[row["case"]].append(o)
        return {k: (sum(v) / len(v) if v else None) for k, v in out.items()}
    b, c = means(base_dir), means(cand_dir)
    lines = [f"base {base_dir.name}", f"cand {cand_dir.name}", "",
             f"{'case':28} {'base':>5} {'cand':>5} {'delta':>6}"]
    deltas = []
    for case in sorted(b):
        bv, cv = b[case], c.get(case)
        d = (cv - bv) if bv is not None and cv is not None else None
        if d is not None:
            deltas.append(d)
        mark = "" if d is None else ("  ▼" if d <= -1 else ("  ▲" if d >= 1 else ""))
        lines.append(f"{case:28} {bv if bv is not None else '-':>5} {cv if cv is not None else '-':>5} "
                     f"{('%+.1f' % d) if d is not None else '-':>6}{mark}")
    if deltas:
        lines += ["", f"mean delta {sum(deltas) / len(deltas):+.2f} over {len(deltas)} cases · "
                      f"{sum(d <= -1 for d in deltas)} clearly worse · {sum(d >= 1 for d in deltas)} clearly better"]
    return "\n".join(lines)
