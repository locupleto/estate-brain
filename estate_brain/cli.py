"""`brain` command line.

    brain eval capture [--only a,b] [--force]
    brain eval run --engine claude-code|opencode [--model M] [--only a,b] [--jobs 3]
    brain eval report [RUN_DIR]          (default: the latest run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval").add_subparsers(dest="sub", required=True)
    cap = ev.add_parser("capture")
    cap.add_argument("--only")
    cap.add_argument("--force", action="store_true")
    rn = ev.add_parser("run")
    rn.add_argument("--engine", required=True)
    rn.add_argument("--model")
    rn.add_argument("--only")
    rn.add_argument("--jobs", type=int, default=3)
    rn.add_argument("--judge-model", default="sonnet")
    rn.add_argument("--label", default="")
    rn.add_argument("--repeat", type=int, default=1)
    rn.add_argument("--into", help="write into an existing run directory")
    rn.add_argument("--retry-failed", action="store_true",
                    help="with --into: only re-run cases lost to provider rate limits")
    rs = ev.add_parser("rescore")
    rs.add_argument("run_dir")
    cp = ev.add_parser("compare")
    cp.add_argument("base_dir")
    cp.add_argument("cand_dir")
    rp = ev.add_parser("report")
    rp.add_argument("run_dir", nargs="?")
    a = ap.parse_args(argv)

    from .evals import cases, runner
    only = a.only.split(",") if getattr(a, "only", None) else None
    if a.sub == "capture":
        for c in cases.load_golden():
            if only and c["id"] not in only:
                continue
            try:
                p = cases.capture(c, force=a.force)
                rec = cases.load_case(c["id"])
                mem = (p / "memory.tgz").stat().st_size if (p / "memory.tgz").exists() else 0
                print(f"  {c['id']:28} prompt {len(rec['prompt']):>7} chars  ref {len(rec['reference_text'] or ''):>6}"
                      f"  structured={'yes' if rec.get('reference_structured') else 'no '}  "
                      f"memory {mem // 1024} KB  {rec.get('model')}")
            except Exception as e:
                print(f"  {c['id']:28} FAILED: {e}")
        return 0
    if a.sub == "run":
        out = runner.run_suite(a.engine, a.model, only, a.jobs, a.judge_model, a.label,
                               a.repeat, a.into, a.retry_failed)
        print(runner.report(out))
        return 0
    if a.sub == "rescore":
        runner.rescore(Path(a.run_dir))
        print(runner.report(Path(a.run_dir)))
        return 0
    if a.sub == "compare":
        print(runner.compare(Path(a.base_dir), Path(a.cand_dir)))
        return 0
    if a.sub == "report":
        d = Path(a.run_dir) if a.run_dir else max(cases.RUNS.iterdir(), key=lambda p: p.name)
        print(runner.report(d))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
