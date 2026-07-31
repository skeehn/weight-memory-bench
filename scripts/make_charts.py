"""Generate the figures, as hand-written SVG.

No matplotlib. Three reasons: the repo stays installable without a plotting stack, SVG
renders inline on GitHub without committing binaries, and a chart whose every coordinate is
computed from the ledger cannot drift from the data the way a re-exported PNG can.

Colours are mid-tone deliberately, so the figures are legible on both light and dark
backgrounds without relying on GitHub honouring `prefers-color-scheme`.

    uv run python -m scripts.make_charts
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "charts"
W, H = 720, 420
PAD_L, PAD_R, PAD_T, PAD_B = 78, 130, 46, 58

INK = "#8b949e"      # axes and labels: readable on white and on dark
GRID = "#8b949e33"
DANGER = "#e5534b"   # damage
GOOD = "#3fb950"     # retention
ACCENT = "#58a6ff"   # neutral series


def px(v, lo, hi, a, b):
    """Map a data value onto a pixel range."""
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def frame(title, subtitle=""):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="{PAD_L}" y="40" font-size="11" fill="{INK}">{subtitle}</text>',
    ]


def axes(x0, x1, y0, y1):
    return [
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" stroke-width="1"/>',
    ]


def forgetting_chart(data: dict) -> str:
    rows = data["rows"]
    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = PAD_T + 18, H - PAD_B
    xs = [r["updates"] for r in rows]

    # Log scale on perplexity: it spans 1.00x to 36.83x, and a linear axis would flatten
    # everything before the final point into the baseline.
    ratios = [r["ppl_ratio"] for r in rows]
    lo, hi = 0.0, math.log10(max(ratios) * 1.3)

    out = frame(
        "Online weight memory: damage compounds, memory never arrives",
        "Llama-3.2-1B, rank 16, lr 2e-3, one continuous update stream (n=1 seed)",
    )
    out += axes(x0, x1, y0, y1)

    for tick in (1, 2, 5, 10, 40):
        y = px(math.log10(tick), lo, hi, y1, y0)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="10" fill="{INK}" '
            f'text-anchor="end">{tick}x</text>'
        )
    for v in (0.0, 0.5, 1.0):
        y = px(v, 0, 1, y1, y0)
        out.append(
            f'<text x="{x1 + 8}" y="{y + 4:.1f}" font-size="10" fill="{GOOD}">{v:.1f}</text>'
        )

    for r in rows:
        x = px(r["updates"], min(xs), max(xs), x0, x1)
        out.append(
            f'<text x="{x:.1f}" y="{y1 + 18}" font-size="10" fill="{INK}" '
            f'text-anchor="middle">{r["updates"]}</text>'
        )

    ppl_pts = " ".join(
        f'{px(r["updates"], min(xs), max(xs), x0, x1):.1f},'
        f'{px(math.log10(r["ppl_ratio"]), lo, hi, y1, y0):.1f}'
        for r in rows
    )
    ret_pts = " ".join(
        f'{px(r["updates"], min(xs), max(xs), x0, x1):.1f},'
        f'{px(r["retention"], 0, 1, y1, y0):.1f}'
        for r in rows
    )
    out.append(
        f'<polyline points="{ppl_pts}" fill="none" stroke="{DANGER}" stroke-width="2.5"/>'
    )
    out.append(
        f'<polyline points="{ret_pts}" fill="none" stroke="{GOOD}" stroke-width="2.5" '
        f'stroke-dasharray="5 3"/>'
    )
    for r in rows:
        x = px(r["updates"], min(xs), max(xs), x0, x1)
        out.append(
            f'<circle cx="{x:.1f}" cy="{px(math.log10(r["ppl_ratio"]), lo, hi, y1, y0):.1f}" '
            f'r="3.5" fill="{DANGER}"/>'
        )
        out.append(
            f'<circle cx="{x:.1f}" cy="{px(r["retention"], 0, 1, y1, y0):.1f}" '
            f'r="3.5" fill="{GOOD}"/>'
        )

    final = rows[-1]
    fx = px(final["updates"], min(xs), max(xs), x0, x1)
    fy = px(math.log10(final["ppl_ratio"]), lo, hi, y1, y0)
    out.append(
        f'<text x="{fx - 6:.1f}" y="{fy - 10:.1f}" font-size="11" font-weight="600" '
        f'fill="{DANGER}" text-anchor="end">{final["ppl_ratio"]:.1f}x worse</text>'
    )
    out.append(
        f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 16}" font-size="11" fill="{INK}" '
        f'text-anchor="middle">online updates applied</text>'
    )
    out.append(
        f'<text x="{x1 + 8}" y="{y0 - 8}" font-size="10" fill="{GOOD}">retention</text>'
    )
    out.append(
        f'<text x="{x0}" y="{y0 - 8}" font-size="10" fill="{DANGER}">held-out perplexity '
        f'(log, x baseline)</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def cost_chart() -> str:
    """Token cost against accuracy. Both axes measured, no arm omitted."""
    arms = [
        ("full context", 105_708, 0.160, 1.000),
        ("RAG", 4_061, 0.040, 0.968),
        ("grep", 4_075, 0.040, 0.809),
        ("weight memory", 1, 0.000, 0.000),  # 1 token so it plots on a log axis
    ]
    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = PAD_T + 18, H - PAD_B
    lo, hi = 0.0, math.log10(200_000)

    out = frame(
        "Cost against accuracy: the expensive arm wins, and still gets 16%",
        "LongMemEval-S dev n=100, Llama-3.2-1B. Weight memory plotted at 0 tokens, 0 accuracy.",
    )
    out += axes(x0, x1, y0, y1)

    for tick in (1, 10, 100, 1_000, 10_000, 100_000):
        x = px(math.log10(tick), lo, hi, x0, x1)
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{GRID}"/>')
        label = f"{tick:,}" if tick < 1000 else f"{tick // 1000}k"
        out.append(
            f'<text x="{x:.1f}" y="{y1 + 18}" font-size="10" fill="{INK}" '
            f'text-anchor="middle">{label}</text>'
        )
    for v in (0.0, 0.05, 0.10, 0.15, 0.20):
        y = px(v, 0, 0.22, y1, y0)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="10" fill="{INK}" '
            f'text-anchor="end">{v:.2f}</text>'
        )

    for name, tokens, acc, _recall in arms:
        x = px(math.log10(max(tokens, 1)), lo, hi, x0, x1)
        y = px(acc, 0, 0.22, y1, y0)
        colour = DANGER if name == "full context" else (GOOD if name == "weight memory" else ACCENT)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}"/>')
        anchor = "end" if name == "full context" else "start"
        dx = -10 if anchor == "end" else 10
        out.append(
            f'<text x="{x + dx:.1f}" y="{y + 4:.1f}" font-size="11" fill="{INK}" '
            f'text-anchor="{anchor}">{name}</text>'
        )

    out.append(
        f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 16}" font-size="11" fill="{INK}" '
        f'text-anchor="middle">context tokens per query (log)</text>'
    )
    out.append(
        f'<text x="{x0}" y="{y0 - 8}" font-size="10" fill="{INK}">accuracy over all '
        f'(abstentions counted wrong)</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent

    src = root / "runs" / "forgetting_curve_remote.json"
    if src.exists():
        (OUT / "forgetting_curve.svg").write_text(forgetting_chart(json.loads(src.read_text())))
        print(f"wrote {OUT / 'forgetting_curve.svg'}")
    else:
        print(f"skipped forgetting chart: {src} not found")

    (OUT / "cost_vs_accuracy.svg").write_text(cost_chart())
    print(f"wrote {OUT / 'cost_vs_accuracy.svg'}")


if __name__ == "__main__":
    main()
