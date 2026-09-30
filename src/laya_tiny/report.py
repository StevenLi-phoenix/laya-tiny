"""Stage `report`: render eval results to reports/<date>.md + .json and an overview chart."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .stages import Ctx
from .util import log, write_json

PALETTE = {"student": "#e4572e", "teacher": "#29335c", "baseline": "#a0a4b8", "external": "#669bbc"}


def fmt_bytes(n: int) -> str:
    if not n:
        return "–"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.3g} {unit}"
        n /= 1000
    return f"{n:.3g} TB"


def pct(x: float | None) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def render_markdown(res: dict[str, Any]) -> str:
    decs = list(res["decisions"])
    L: list[str] = []
    acc = res["acceptance"]
    L.append(f"# laya-tiny evaluation — {res['generated'][:10]}")
    L.append("")
    L.append(f"Profile `{res['profile']}` · holdout `{res['holdout']['path']}` "
             f"({res['holdout']['en']} English rows, {res['holdout']['other_lang']} other-language rows) · "
             f"teacher **{res['teacher']}** · student {res['student']['params'] / 1e6:.2f}M params, "
             f"max_len {res['student']['max_len']}, run `{res['student']['run_id']}`.")
    L.append("")
    L.append(f"**Acceptance: {'PASS' if acc['pass'] else 'FAIL'}**")
    L.append("")
    L.append("| check | value | target | |")
    L.append("|---|---|---|---|")
    for k, c in acc["checks"].items():
        if k.startswith("accuracy/"):
            L.append(f"| {k} | {pct(c['student'])}% (teacher {pct(c['teacher'])}%, gap {100 * c['gap']:+.1f} pp) "
                     f"| ≥ teacher − {100 * acc['margin']:.0f} pp | {'✅' if c['pass'] else '❌'} |")
        else:
            L.append(f"| {k} | {c['value']:.2f} | ≤ {c['limit']} | {'✅' if c['pass'] else '❌'} |")
    L.append("")
    L.append("## English holdout")
    L.append("")
    head = "| backend | size | CPU p50 / p95 (ms) | " + " | ".join(f"{d} acc · F1 · ECE" for d in decs) + " | teacher agreement |"
    L.append(head)
    L.append("|" + "---|" * (4 + len(decs)))
    for b in res["backends"]:
        en = b["slices"]["en"]
        lat = b["latency_ms"]
        lat_s = f"{lat['p50']:.2f} / {lat['p95']:.2f}" if lat else "–"
        cells = [f"{pct(en[d]['accuracy'])} · {pct(en[d]['macro_f1'])} · {en[d]['ece']:.3f}" for d in decs]
        agree = " / ".join(pct(en[d].get("teacher_agreement")) for d in decs)
        L.append(f"| {b['name']} | {fmt_bytes(b['size_bytes'])} | {lat_s} | " + " | ".join(cells) + f" | {agree} |")
    L.append("")
    L.append("Accuracy, macro-F1 and agreement in %. Teacher agreement = argmax match with the teacher, per decision in "
             "the order " + ", ".join(decs) + ". Latency is single-row (batch = 1); see `latency_ms.what` in the JSON "
             "for how each backend was timed.")
    others = [b for b in res["backends"] if "other_lang" in b["slices"]]
    if others:
        L.append("")
        L.append("## Other-language slice (out of scope: the teacher and tokenizer are English-only)")
        L.append("")
        L.append("| backend | " + " | ".join(f"{d} acc" for d in decs) + " |")
        L.append("|" + "---|" * (1 + len(decs)))
        for b in others:
            o = b["slices"]["other_lang"]
            L.append(f"| {b['name']} | " + " | ".join(pct(o[d]["accuracy"]) for d in decs) + " |")
    L.append("")
    return "\n".join(L)


def render_chart(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decs = list(res["decisions"])
    bs = [b for b in res["backends"] if b["latency_ms"] or b["size_bytes"]]
    names = [b["name"] for b in bs]
    colors = [PALETTE.get(b["kind"], "#888") for b in bs]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    fig.patch.set_facecolor("#fbfaf7")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    def hbar(ax: Any, vals: list[float], title: str, unit: str) -> None:
        y = range(len(names))
        ax.barh(y, vals, color=colors, height=0.62)
        ax.set_xscale("log")
        ax.set_yticks(list(y), names)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(v * 1.15, i, f"{v:.3g} {unit}", va="center", fontsize=9)
        ax.set_xlim(right=max(v for v in vals if v > 0) * 30)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)

    hbar(axes[0], [b["size_bytes"] / 1e6 for b in bs], "Size on disk", "MB")
    hbar(axes[1], [(b["latency_ms"] or {}).get("p50", 0) for b in bs], "CPU latency p50, batch = 1", "ms")
    axes[1].set_yticks([])
    ax = axes[2]
    width = 0.8 / len(bs)
    for i, b in enumerate(bs):
        vals = [100 * b["slices"]["en"][d]["accuracy"] for d in decs]
        xs = [j + (i - (len(bs) - 1) / 2) * width for j in range(len(decs))]
        ax.bar(xs, vals, width=width * 0.92, color=colors[i], label=b["name"])
    ax.set_xticks(range(len(decs)), decs)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy on English holdout (%)")
    ax.set_title("Accuracy per decision", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(axis="y", color="#e5e2da", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("laya-tiny: Laya distilled into a task-specific Transformer", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(ctx: Ctx) -> dict[str, Any]:
    res = json.loads(ctx.path("eval", "results.json").read_text())
    md = render_markdown(res)
    out = ctx.path("report", "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    render_chart(res, ctx.path("report", "overview.png"))
    rdir = ctx.root / ctx.cfg["reports_dir"]
    stamp = time.strftime("%Y-%m-%d")
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{stamp}.md").write_text(md, encoding="utf-8")
    write_json(rdir / f"{stamp}.json", res)
    shutil.copy2(ctx.path("report", "overview.png"), rdir / "overview.png")
    log.info("report written to %s", rdir / f"{stamp}.md")
    return {"report": str(rdir / f"{stamp}.md"), "pass": res["acceptance"]["pass"]}
