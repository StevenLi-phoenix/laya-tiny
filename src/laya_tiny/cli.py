"""laya-tiny command line: `laya-tiny run <stage>` runs that stage and anything stale upstream."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .stages import ORDER, STAGES, Ctx, fingerprint, is_fresh, plan, run
from .util import seed_everything, setup_logging


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        cur = out
        *parents, leaf = key.split(".")
        for p in parents:
            cur = cur.setdefault(p, {})
        cur[leaf] = yaml.safe_load(raw)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="laya-tiny", description=__doc__)
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--profile", default=None, help="config profile, e.g. smoke")
    ap.add_argument("--project", default=None, help="goal-driven project under projects/<name> (see `new`)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set data.max_rows=4000 (value parsed as YAML)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a stage (and stale upstream stages)")
    r.add_argument("stage", choices=ORDER)
    r.add_argument("--force", default="", help="comma-separated stages to re-run even if fresh, or 'all'")
    r.add_argument("--only", action="store_true", help="run just this stage; upstream outputs must exist")
    sub.add_parser("status", help="show which stages are fresh")
    n = sub.add_parser("new", help="one goal in, a small expert model out: LLM designs the task and writes "
                                   "the data, Laya labels it, the student is trained, exported and packaged")
    n.add_argument("goal", help='what the model should decide, e.g. "triage food-delivery complaints"')
    n.add_argument("--name", default=None, help="project name (default: chosen by the LLM)")
    n.add_argument("--rows", type=int, default=3000, help="synthetic training texts")
    n.add_argument("--holdout-rows", type=int, default=300, help="synthetic evaluation texts")
    n.add_argument("--force-spec", action="store_true", help="re-design the task even if the project exists")
    args = ap.parse_args(argv)

    overrides = parse_overrides(args.set)
    root = Path.cwd()
    if args.cmd == "new":
        from .spec import design

        setup_logging(args.verbose)
        base = load_config(args.config_dir, args.profile, overrides)
        name = design(args.goal, args.name, base, root, args.rows, args.holdout_rows, force=args.force_spec)
        args.project = name
    cfg = load_config(args.config_dir, args.profile, overrides, project=args.project)
    ctx = Ctx(cfg, root)
    setup_logging(args.verbose, ctx.work / "pipeline.log")
    seed_everything(int(cfg["seed"]))
    if args.cmd == "new":
        run(ctx, "package")
        pkg = ctx.path("package")
        print(json.dumps({"project": args.project, "package": str(pkg), "report": str(ctx.path("report", "report.md"))}))
        print(f"\nexpert model ready: {pkg}\n  try: python {pkg / 'predict.py'} \"some text\"", file=sys.stderr)
        return 0
    if args.cmd == "run":
        force = {s for s in args.force.split(",") if s}
        summaries = run(ctx, args.stage, force=force, only=args.only)
        print(json.dumps({"profile": cfg["profile"], "ran": list(summaries)}, indent=None))
        return 0
    if args.cmd == "status":
        stale: set[str] = set()
        for name in plan("package"):
            st = STAGES[name]
            if any(d in stale for d in st.deps):
                stale.add(name)
                print(f"{name:10s} stale (upstream)")
            elif not is_fresh(ctx, st, fingerprint(ctx, st)):
                stale.add(name)
                print(f"{name:10s} stale")
            else:
                print(f"{name:10s} fresh")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
