"""The v4 results report (replication + logit lens + steering), written for readers new to the field.

Runs locally on results copied into local_data/v4/results/<model>/ (analysis.json, lens.json, steer.json,
steer.npz, test/rows.jsonl) and the example images in local_data/v4/test/. Needs opencv + numpy
(reuses the data report's image helpers):

    python repro/report/build_v4_results_report.py      # -> local_data/v4/v4_results_report.html
"""
import html
import json
import math
from pathlib import Path

import numpy as np

from build_v4_data_report import COL, LOCAL, PROMPT, jpeg, outlined, read_rgb, sketch  # noqa: F401
import cv2

ROOT = Path(__file__).resolve().parents[2]
RES = LOCAL / "results"
MODELS = ["qwen3vl", "gemma4", "bagel"]
NAME = {"qwen3vl": "Qwen3-VL-8B", "gemma4": "Gemma 4 12B", "bagel": "BAGEL-7B-MoT"}
SHORT = {"qwen3vl": "Qwen", "gemma4": "Gemma", "bagel": "BAGEL"}
SER = {"qwen3vl": "s1", "gemma4": "s2", "bagel": "s3"}
PHD = json.loads((Path(__file__).parent / "phd_v4_numbers.json").read_text())
PHD_URL = "https://claude.ai/artifact/Ljb9PBXewzPvhY4DihJB8a?sk=Knm45viCXxHIyYavJrQWfw"
DATA_URL = "https://claude.ai/artifact/BSDujB7F5ff2gqDJ3ddAXe"
EXAMPLES = ["apt_1_L667_P340_B2", "apt_1_L1181_P590_B0"]   # one flip, one no-flip (images on disk)
N_BOOT = 1000


def esc(s):
    return html.escape(str(s))


def pc(x, d=0):
    return f"{x * 100:.{d}f}%"


ADD = '<span class="mk add" title="Something we added to the PhD\'s method">added</span>'
CHG = '<span class="mk chg" title="Something we did differently from the PhD">changed</span>'


# ---------------------------------------------------------------- data

RES2 = LOCAL / "results_steer2"   # the redesigned steering (PLAN_V4_STEER.md)


def load():
    D = {}
    for m in MODELS:
        d = RES / m
        D[m] = {"an": json.loads((d / "analysis.json").read_text()),
                "lens": json.loads((d / "lens.json").read_text()),
                "rows": [json.loads(l) for l in open(d / "test" / "rows.jsonl")],
                "s2": {k: json.loads((RES2 / m / f"{k}.json").read_text())
                       for k in ("reliability", "gate2", "natural", "steer")}}
    return D


def steer2_table(dm):
    """For the steering chart: per layer, shift in the lean toward B's answer [mean, lo, hi, broken] for B_perp,
    D_perp, A_perp and B; random seeds as [min, max, mean]; g as [mean, broken]."""
    st = dm["s2"]["steer"]
    rel = {r["layer"]: r["B_raw"]["train_test"] for r in dm["s2"]["reliability"]["layers"]}
    out = {"layers": st["layers"], "strengths": st["strengths"], "best": st["headline_layer"],
           "reliable": [rel[L] >= 0.5 for L in st["layers"]], "dirs": {k: [] for k in ("B_perp", "D_perp", "A_perp", "B", "rand", "g")}}
    for e in st["per_layer"]:
        for k in ("B_perp", "D_perp", "A_perp", "B"):
            out["dirs"][k].append([[round(c["shift"], 3), round(c["ci"][0], 3), round(c["ci"][1], 3), round(c["broken"], 3)]
                                   for c in e["dirs"][k]])
        rnd = [[e["dirs"][f"rand_{i}"][ki]["shift"] for i in range(10)] for ki in range(len(st["strengths"]))]
        out["dirs"]["rand"].append([[round(min(r), 3), round(max(r), 3), round(float(np.mean(r)), 3)] for r in rnd])
        out["dirs"]["g"].append([[round(c["shift"], 3), round(c["broken"], 3)] for c in e["dirs"]["g"]])
    return out


def at_layer(rows, L):
    return next(r for r in rows if r["layer"] == L)


def full_rel(rho):
    return float(np.sqrt(2 * rho / (1 + rho))) if rho > 0 else float("nan")


def s2_validation_rows(D):
    out = ""
    for m in MODELS:
        rl = D[m]["s2"]["reliability"]
        h = rl["headline_layer"]
        r = at_layer(rl["layers"], h)["B_raw"]
        out += (f'<tr><th>{NAME[m]} <small>L{h}</small></th><td class="num">{r["train_test"]:.2f}</td>'
                f'<td class="num">{r["split_half"]:.2f}</td><td class="num"><b>{full_rel(r["split_half"]):.2f}</b></td>'
                f'<td class="num">{r["null"]:+.2f}</td></tr>')
    return out


def s2_gate_rows(D):
    out = ""
    for m in MODELS:
        g = D[m]["s2"]["gate2"]
        r = at_layer(g["layers"], g["headline_layer"])
        out += (f'<tr><th>{NAME[m]} <small>L{g["headline_layer"]}</small></th><td class="num">{pc(r["grad_realistic_share"])}</td>'
                f'<td class="num">{r["k"]}</td><td class="num">{pc(r["ratio_B"], 1)}</td><td class="num"><b>{pc(r["ratio_Bperp"], 1)}</b></td>'
                f'<td class="num">{pc(r["ratio_Dperp"], 1)}</td><td class="num">{pc(r["ratio_rand"], 1)}</td><td class="num">{pc(r["B_kept"])}</td></tr>')
    return out


def s2_part1_rows(D):
    out = ""
    for m in MODELS:
        n = D[m]["s2"]["natural"]
        r = at_layer([x for x in n["rows"] if x["position"] == "answer"], n["headline_layer"])
        out += (f'<tr><th>{NAME[m]} <small>L{n["headline_layer"]}</small></th><td class="num">{r["B_overlap"]:+.2f}</td>'
                f'<td class="num"><b>{pc(r["B_overlap_pct"])}</b></td><td class="num">{r["A_overlap"]:+.2f}</td>'
                f'<td class="num"><b>{pc(r["A_overlap_pct"])}</b></td></tr>')
    return out


def s2_part1_charts(D):
    out = []
    for m in MODELS:
        n = D[m]["s2"]["natural"]
        rows = [x for x in n["rows"] if x["position"] == "answer"]
        xs = [r["layer"] for r in rows]
        nl = max(xs) + 1
        series = [{"xs": xs, "ys": [r["A_overlap"] for r in rows], "cls": "s3", "label": "camera (A)"},
                  {"xs": xs, "ys": [r["B_overlap"] for r in rows], "cls": "s1", "label": "B"}]
        band = {"xs": xs, "lo": [-r["null_overlap_95"] for r in rows], "hi": [r["null_overlap_95"] for r in rows],
                "cls": "b-perm", "title": "95% of random realistic directions"}
        out.append(f'<figure class="panel"><figcaption class="ph">{NAME[m]}</figcaption>' + line_chart(
            series, 0, nl, list(range(0, nl + 1, 8 if nl > 40 else 4)), "layer", "overlap with the answer channel",
            y0=-1, y1=1, yticks=(-1, -.5, 0, .5, 1), bands=[band], hlines=[{"y": 0}], w=420, h=260,
            title=f"Overlap with the answer channel by layer, {NAME[m]}", fmt_y=lambda v: f"{v:+.1f}", labels=False) + "</figure>")
    return "".join(out)


def s2_part2_rows(D):
    out = ""
    for m in MODELS:
        st = D[m]["s2"]["steer"]
        h = st["headline_layer"]
        e = at_layer(st["per_layer"], h)
        cells = ""
        for k in (16.0, 32.0):
            ki = st["strengths"].index(k)
            bp = e["dirs"]["B_perp"][ki]
            r = [e["dirs"][f"rand_{i}"][ki]["shift"] for i in range(10)]
            brk = f'<small>{pc(bp["broken"])} broken</small>' if bp["broken"] >= .005 else ""
            cells += (f'<td class="num"><b>{bp["shift"]:+.2f}</b><small class="ci">{bp["ci"][0]:+.2f} to {bp["ci"][1]:+.2f}</small>{brk}</td>'
                      f'<td class="num">{min(r):+.2f} to {max(r):+.2f}</td><td class="num">{e["dirs"]["D_perp"][ki]["shift"]:+.2f}</td>')
        g32 = e["dirs"]["g"][st["strengths"].index(32.0)]["shift"]
        ok = st["decision"]["content_acted_on"]
        out += (f'<tr><th>{NAME[m]} <small>L{h}</small></th>{cells}<td class="num">{g32:+.1f}</td>'
                f'<td><span class="verdict {"yes" if ok else "no"}">{"acted on" if ok else "no"}</span></td></tr>')
    return out


def lse2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


# ---------------------------------------------------------------- charts (static SVG)

def line_chart(series, x0, x1, xticks, xlabel, ylabel="", y0=0, y1=1, w=640, h=300, bands=(), hlines=(),
               yticks=(0, .25, .5, .75, 1), title="", fmt_x=lambda v: str(v), labels=True,
               fmt_y=lambda v: f"{v * 100:.0f}%"):
    """series: list of dicts {xs, ys, cls, label, dash, lo, hi}. bands: list of {xs, lo, hi, cls}."""
    ml, mr, mt, mb = 46, 118 if labels else 14, 12, 42
    pw, ph = w - ml - mr, h - mt - mb
    X = lambda v: ml + (v - x0) / (x1 - x0) * pw
    Y = lambda v: mt + (1 - (v - y0) / (y1 - y0)) * ph
    el = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    for t in yticks:
        el.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                  f'<text x="{ml - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{fmt_y(t)}</text>')
    for t in xticks:
        el.append(f'<text x="{X(t):.1f}" y="{mt + ph + 16}" class="tick" text-anchor="middle">{fmt_x(t)}</text>')
    el.append(f'<text x="{ml + pw / 2:.1f}" y="{h - 6}" class="axl" text-anchor="middle">{esc(xlabel)}</text>')
    if ylabel:
        el.append(f'<text x="12" y="{mt + ph / 2:.1f}" class="axl" text-anchor="middle" '
                  f'transform="rotate(-90 12 {mt + ph / 2:.1f})">{esc(ylabel)}</text>')
    for hl in hlines:
        el.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(hl["y"]):.1f}" y2="{Y(hl["y"]):.1f}" class="hl {hl.get("cls", "")}"/>')
        if hl.get("label"):
            el.append(f'<text x="{ml + pw - 4}" y="{Y(hl["y"]) - 5:.1f}" class="tick" text-anchor="end">{esc(hl["label"])}</text>')
    for b in bands:
        top = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(b["xs"], b["hi"]))
        bot = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(b["xs"][::-1], b["lo"][::-1]))
        el.append(f'<polygon points="{top} {bot}" class="band {b["cls"]}"><title>{esc(b.get("title", ""))}</title></polygon>')
    placed = []
    for s in series:
        pts = [(x, y) for x, y in zip(s["xs"], s["ys"]) if y is not None]
        if "lo" in s:
            top = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(s["xs"], s["hi"]))
            bot = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(s["xs"][::-1], s["lo"][::-1]))
            el.append(f'<polygon points="{top} {bot}" class="ci {s["cls"]}"/>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        el.append(f'<path d="{d}" class="ln {s["cls"]} {"dash" if s.get("dash") else ""}"/>')
        for x, y in pts:
            el.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="7" class="hit"><title>{esc(s["label"])}, '
                      f'{esc(fmt_x(x))}: {fmt_y(y)}</title></circle>')
        if labels and pts:
            ly = Y(pts[-1][1])
            for p in placed:
                if abs(ly - p) < 13:
                    ly = p + 13
            placed.append(ly)
            el.append(f'<text x="{X(pts[-1][0]) + 6:.1f}" y="{ly + 4:.1f}" class="dl {s["cls"]}">{esc(s["label"])}</text>')
    el.append("</svg>")
    return "".join(el)


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i class="sw-{c}{" dash" if d else ""}"></i>{esc(t)}</span>' for t, c, d in items) + "</div>"


# ---------------------------------------------------------------- explanatory diagrams

def flip_diagram():
    """Two panels: B on A's side of the X-Y line (same answer), B across it (answer flips)."""
    def panel(ox, title, bx, by, flipped):
        X, Y = (ox + 95, 70), (ox + 175, 70)
        A = (ox + 135, 175)
        s = [f'<text x="{ox + 135}" y="18" class="dg-t" text-anchor="middle">{esc(title)}</text>',
             f'<line x1="{ox + 15}" y1="70" x2="{ox + 255}" y2="70" class="dg-line"/>',
             f'<circle cx="{X[0]}" cy="{X[1]}" r="11" class="dg-X"/><text x="{X[0]}" y="{X[1] + 4}" class="dg-l">X</text>',
             f'<circle cx="{Y[0]}" cy="{Y[1]}" r="11" class="dg-Y"/><text x="{Y[0]}" y="{Y[1] + 4}" class="dg-l">Y</text>',
             f'<path d="M{A[0]},{A[1]} L{A[0] - 30},{A[1] - 40} L{A[0] + 30},{A[1] - 40} Z" class="dg-cone"/>',
             f'<circle cx="{A[0]}" cy="{A[1]}" r="10" class="dg-A"/><text x="{A[0]}" y="{A[1] + 4}" class="dg-l dg-lA">A</text>',
             f'<path d="M{bx},{by} L{bx - 26 if not flipped else bx - 26},{by - 34 if not flipped else by + 34} '
             f'L{bx + 26},{by - 34 if not flipped else by + 34} Z" class="dg-cone dg-coneB"/>',
             f'<circle cx="{bx}" cy="{by}" r="10" class="dg-B"/><text x="{bx}" y="{by + 4}" class="dg-l">B</text>',
             f'<text x="{ox + 135}" y="214" class="dg-c" text-anchor="middle">From A: X is left of Y.</text>',
             f'<text x="{ox + 135}" y="232" class="dg-c" text-anchor="middle">'
             f'{"From B: X is left of Y too." if not flipped else "From B: X is RIGHT of Y."}</text>']
        return "".join(s)
    return ('<svg viewBox="0 0 560 244" class="diagram" role="img" aria-label="Why the answer flips">'
            + panel(0, "B on the camera's side: same answer", 225, 150, False)
            + panel(290, "B across the line: answer flips", 425 + 10, 22 + 8, True)
            + '<text x="280" y="236" class="dg-c" text-anchor="middle"></text></svg>')


def layers_diagram():
    cols = ["photo", "…", "Imagine", "…", "B", "…", "X", "…", "Y", "…", "end"]
    w, h = 600, 230
    el = [f'<svg viewBox="0 0 {w} {h}" class="diagram" role="img" aria-label="Hidden states across layers">']
    x0, dx = 60, 48
    for li, lab in enumerate(["layer 0", "layer 1", "…", "last layer"]):
        y = 175 - li * 42
        el.append(f'<text x="{x0 - 8}" y="{y + 5}" class="dg-c" text-anchor="end">{lab}</text>')
        for ci in range(len(cols)):
            cls = "dg-cell hot" if ci == len(cols) - 1 else "dg-cell"
            el.append(f'<rect x="{x0 + ci * dx}" y="{y - 12}" width="{dx - 10}" height="22" rx="3" class="{cls}"/>')
    for ci, lab in enumerate(cols):
        el.append(f'<text x="{x0 + ci * dx + (dx - 10) / 2}" y="210" class="dg-c" text-anchor="middle">{esc(lab)}</text>')
    el.append(f'<text x="{x0}" y="16" class="dg-c">Each box is a list of a few thousand numbers (a "hidden state"). Each layer updates every box,</text>')
    el.append(f'<text x="{x0}" y="32" class="dg-c">looking at the boxes to its left. We read the highlighted column: the spot just before the answer.</text>')
    el.append("</svg>")
    return "".join(el)


def means_diagram():
    """Cartoon of the four groups and the two differences that make B's direction (illustration only)."""
    rng = np.random.default_rng(3)
    cen = {"LL": (150, 80), "LR": (150, 190), "RL": (390, 80), "RR": (390, 190)}
    cls = {"LL": "dg-p1", "LR": "dg-p2", "RL": "dg-p1", "RR": "dg-p2"}
    el = ['<svg viewBox="0 0 560 290" class="diagram" role="img" aria-label="Difference of means">']
    for k, (cx, cy) in cen.items():
        for _ in range(26):
            x, y = cx + rng.normal(0, 22), cy + rng.normal(0, 13)
            el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="{cls[k]}"/>')
        el.append(f'<circle cx="{cx}" cy="{cy}" r="6" class="dg-mean"/>')
    for (a, b) in (("LR", "LL"), ("RR", "RL")):
        (x1, y1), (x2, y2) = cen[a], cen[b]
        el.append(f'<line x1="{x1 + 14}" y1="{y1 - 10}" x2="{x2 + 14}" y2="{y2 + 14}" class="dg-arrow" marker-end="url(#ah)"/>')
    el.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
              '<path d="M0,0 L10,5 L0,10 z" class="dg-ahead"/></marker></defs>')
    el.append('<text x="150" y="30" class="dg-t" text-anchor="middle">A\'s answer: left</text>'
              '<text x="390" y="30" class="dg-t" text-anchor="middle">A\'s answer: right</text>'
              '<text x="70" y="84" class="dg-c" text-anchor="end">B left</text><text x="70" y="194" class="dg-c" text-anchor="end">B right</text>'
              '<text x="150" y="250" class="dg-c" text-anchor="middle">arrow 1 = mean(B left) − mean(B right)</text>'
              '<text x="390" y="250" class="dg-c" text-anchor="middle">arrow 2 = the same, among A-right questions</text>'
              '<text x="280" y="278" class="dg-t" text-anchor="middle">B\'s direction = the average of arrow 1 and arrow 2</text>')
    el.append("</svg>")
    return "".join(el)


def push_diagram():
    return ('<svg viewBox="0 0 560 150" class="diagram" role="img" aria-label="A push added to a hidden state">'
            '<rect x="20" y="50" width="120" height="44" rx="6" class="dg-cell"/><text x="80" y="77" class="dg-c" text-anchor="middle">hidden state</text>'
            '<text x="165" y="78" class="dg-big" text-anchor="middle">+</text>'
            '<rect x="190" y="50" width="150" height="44" rx="6" class="dg-cell hot"/><text x="265" y="70" class="dg-c" text-anchor="middle">k × (B\'s gap) ×</text>'
            '<text x="265" y="86" class="dg-c" text-anchor="middle">direction</text>'
            '<text x="365" y="78" class="dg-big" text-anchor="middle">→</text>'
            '<rect x="390" y="50" width="150" height="44" rx="6" class="dg-cell"/><text x="465" y="70" class="dg-c" text-anchor="middle">rest of the model</text>'
            '<text x="465" y="86" class="dg-c" text-anchor="middle">→ "left" or "right"?</text>'
            '<text x="280" y="128" class="dg-c" text-anchor="middle">One layer, one position (just before the answer). Everything else runs as normal.</text></svg>')


def perp_diagram():
    """2D cartoon: pushes along, perpendicular to, and nearly perpendicular to the read-out direction."""
    hx, hy, ay = 200, 190, 250
    el = ['<svg viewBox="0 0 600 300" class="diagram" role="img" aria-label="Pushes along and across the read-out direction">',
          '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
          '<path d="M0,0 L10,5 L0,10 z" class="dg-ahead"/></marker></defs>',
          f'<line x1="40" y1="{ay}" x2="560" y2="{ay}" class="dg-arrow" marker-end="url(#ah2)"/>',
          f'<text x="560" y="{ay + 22}" class="dg-c" text-anchor="end">read-out direction: further right = the output leans more to "left"</text>']
    ends = [((360, hy), "dg-ink", "along the read-out (cosine 1)", 368, hy + 4, "start"),
            ((hx, 60), "dg-s2", "perpendicular (cosine 0)", hx - 8, 52, "end"),
            ((213, 61), "dg-s1", "B's direction (cosine about 0.1)", 221, 52, "start")]
    for (x, y), cls, lab, tx, ty, anc in ends:
        el.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{ay}" class="dg-drop"/>')
    el.append(f'<line x1="{hx}" y1="{hy}" x2="{hx}" y2="{ay}" class="dg-drop"/>')
    for (x, y), cls, lab, tx, ty, anc in ends:
        el.append(f'<line x1="{hx}" y1="{hy}" x2="{x}" y2="{y}" class="dg-vec {cls}" marker-end="url(#ah2)"/>')
        el.append(f'<text x="{tx}" y="{ty}" class="dg-c" text-anchor="{anc}">{lab}</text>')
    el.append(f'<circle cx="{hx}" cy="{hy}" r="6" class="dg-mean"/><text x="{hx - 10}" y="{hy + 4}" class="dg-c" text-anchor="end">hidden state</text>')
    el.append(f'<text x="{hx}" y="{ay + 40}" class="dg-c" text-anchor="middle">score before</text>'
              f'<text x="360" y="{ay - 8}" class="dg-c" text-anchor="middle">big change</text>'
              f'<text x="222" y="{ay - 8}" class="dg-c" text-anchor="start">tiny change</text>')
    el.append("</svg>")
    return "".join(el)


def lean_diagram():
    """2D cartoon: the lean is where the hidden state lands along the read-out line; a perpendicular change keeps it."""
    ay, ox, hx, hy, h2y = 228, 380, 230, 168, 84
    el = ['<svg viewBox="0 0 600 296" class="diagram" role="img" aria-label="The lean is the position of the hidden state along the read-out direction">',
          '<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
          '<path d="M0,0 L10,5 L0,10 z" class="dg-ahead"/></marker></defs>',
          f'<line x1="{ox}" y1="34" x2="{ox}" y2="{ay + 10}" class="dg-drop"/>',
          f'<text x="{ox + 8}" y="50" class="dg-c">tie: both words</text>',
          f'<text x="{ox + 8}" y="66" class="dg-c">score the same</text>',
          f'<text x="{ox - 20}" y="22" class="dg-t" text-anchor="end">model says "left"</text>',
          f'<text x="{ox + 20}" y="22" class="dg-t">model says "right"</text>',
          f'<line x1="570" y1="{ay}" x2="30" y2="{ay}" class="dg-arrow" marker-end="url(#ah3)"/>',
          f'<text x="570" y="{ay + 20}" class="dg-c" text-anchor="end">read-out direction</text>',
          f'<line x1="{hx}" y1="{hy}" x2="{hx}" y2="{ay}" class="dg-drop"/>',
          f'<line x1="{hx}" y1="{ay + 34}" x2="{ox}" y2="{ay + 34}" class="dg-vec dg-ink"/>',
          f'<line x1="{hx}" y1="{ay + 28}" x2="{hx}" y2="{ay + 40}" class="dg-vec dg-ink"/>',
          f'<line x1="{ox}" y1="{ay + 28}" x2="{ox}" y2="{ay + 40}" class="dg-vec dg-ink"/>',
          f'<text x="{(hx + ox) / 2}" y="{ay + 58}" class="dg-c" text-anchor="middle">the lean: how far past the tie, toward "left"</text>',
          f'<line x1="{hx}" y1="{hy}" x2="{hx}" y2="{h2y + 8}" class="dg-vec dg-s1" marker-end="url(#ah3)"/>',
          f'<circle cx="{hx}" cy="{h2y}" r="6" class="dg-mean"/>',
          f'<text x="{hx - 12}" y="{h2y - 4}" class="dg-c" text-anchor="end">after a change perpendicular</text>',
          f'<text x="{hx - 12}" y="{h2y + 12}" class="dg-c" text-anchor="end">to the read-out: it lands on</text>',
          f'<text x="{hx - 12}" y="{h2y + 28}" class="dg-c" text-anchor="end">the same spot, same lean</text>',
          f'<circle cx="{hx}" cy="{hy}" r="6" class="dg-mean"/>',
          f'<text x="{hx - 12}" y="{hy + 4}" class="dg-c" text-anchor="end">hidden state h</text>',
          "</svg>"]
    return "".join(el)


def lens_scatter(model, a=1, w=420, h=360):
    """One dot per test question with A's answer = a: output lean (x) vs B reader score (y), coloured by B's answer."""
    d = json.loads((RES / "lens_scatter.json").read_text())[model]
    x, y, A, B = (np.array(d[k]) for k in ("x", "y", "A", "B"))
    mk = A == a
    x, y, B = x[mk], y[mk], B[mk]
    xl = float(np.percentile(np.abs(np.concatenate([x, d["x"]])), 99)) * 1.05
    yl = float(np.percentile(np.abs(y), 99)) * 1.05
    ml, mr, mt, mb = 28, 12, 26, 30
    pw, ph = w - ml - mr, h - mt - mb
    X = lambda v: ml + (np.clip(v, -xl, xl) + xl) / (2 * xl) * pw
    Y = lambda v: mt + (1 - (np.clip(v, -yl, yl) + yl) / (2 * yl)) * ph
    el = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Output lean against B reader score, {NAME[model]}">',
          f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" class="sc-bg"/>',
          f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{mt}" y2="{mt + ph}" class="sc-axis"/>',
          f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="sc-axis"/>']
    order = np.random.default_rng(0).permutation(len(x))
    for i in order:
        el.append(f'<circle cx="{X(x[i]):.1f}" cy="{Y(y[i]):.1f}" r="2.6" class="sc-{"L" if B[i] else "R"}"/>')
    el.append(f'<text x="{ml + pw - 4}" y="{mt + ph + 18}" class="tick" text-anchor="end">output leans "left" →</text>'
              f'<text x="{ml + 4}" y="{mt + ph + 18}" class="tick">← leans "right"</text>'
              f'<text x="{ml + 6}" y="{mt + 14}" class="tick">↑ reader: B says left</text>'
              f'<text x="{ml + 6}" y="{mt + ph - 6}" class="tick">↓ reader: B says right</text>'
              f'<text x="{ml}" y="16" class="dg-t">{NAME[model]} · layer {d["layer"]}</text>')
    el.append("</svg>")
    return "".join(el)


def lens_split_rows():
    """Within each A group: how well the output lean and the B reader tell B's answer apart (balanced accuracy)."""
    d = json.loads((RES / "lens_scatter.json").read_text())
    rows = ""
    for m in MODELS:
        x, y, A, B = (np.array(d[m][k]) for k in ("x", "y", "A", "B"))
        cells = []
        for a in (1, 0):
            mk = A == a
            bal = lambda pred: np.mean([np.mean(pred[mk & (B == c)] == c) for c in (0, 1)])
            cells.append((bal((x > 0).astype(int)), bal((y > 0).astype(int))))
        rows += (f'<tr><th>{NAME[m]} <small>L{d[m]["layer"]}</small></th>'
                 + "".join(f'<td class="num">{pc(o)}</td><td class="num"><b>{pc(r)}</b></td>' for o, r in cells) + "</tr>")
    return rows


def departure_rows(D):
    """How often each model departs from A's answer, by A's answer x B's side of the photo; and where departures land."""
    rows = ""
    for m in MODELS:
        r = D[m]["rows"]
        dep = lambda rs: sum(x["answer"] != x["relA"] for x in rs) / max(1, len(rs))
        cells = [dep([x for x in r if x["relA"] == a and x["image_side"] == sd]) for a in ("left", "right") for sd in ("left", "right")]
        d = [x for x in r if x["answer"] != x["relA"] and x["answer"] in ("left", "right")]
        hitB = sum(x["flip"] for x in d) / max(1, len(d))
        fav = "left" if sum(x["answer"] == "left" for x in r) / len(r) > .5 else "right"
        rows += (f'<tr><th>{NAME[m]}<small class="what">favourite word: "{fav}"</small></th>'
                 + "".join(f'<td class="num">{"<b>" + pc(c) + "</b>" if c > .2 else pc(c)}</td>' for c in cells)
                 + f'<td class="num">{pc(hitB)}</td></tr>')
    return rows


def prompt_diagram():
    parts = [("[picture: image tokens of B, X, Y]", "pt-img"), (" This image shows what you currently see. Imagine you are standing where the ", ""),
             ("B", "pt-mark pt-1"), (" is, looking toward the X and the Y. From there, is the ", ""), ("X", "pt-mark pt-2"),
             (" to the left or to the right of the ", ""), ("Y", "pt-mark pt-3"), ("? Answer with exactly one word: left or right.", ""),
             (" ▸ answer", "pt-mark pt-4")]
    return '<p class="promptbox">' + "".join(f'<span class="{c}">{esc(t)}</span>' if c else esc(t) for t, c in parts) + "</p>"


# ---------------------------------------------------------------- example cards

def example(q, rows):
    d = LOCAL / "test"
    A, semA = read_rgb(d / q["img_A"]), cv2.imread(str(d / q["sem_A"]), cv2.IMREAD_UNCHANGED)
    Bv, semB = read_rgb(d / q["img_B"]), cv2.imread(str(d / q["sem_B"]), cv2.IMREAD_UNCHANGED)
    a_out, a_plain = jpeg(outlined(A, semA, q["sid"]), 300), jpeg(A, 300)
    b_out, b_plain = jpeg(outlined(Bv, semB, {"X": q["sid"]["X"], "Y": q["sid"]["Y"]}), 300), jpeg(Bv, 300)
    said = " · ".join(f'{SHORT[m]}: <b>{esc(rows[m][q["qid"]]["answer"])}</b>' for m in MODELS)
    tag = ('<span class="tag tag-flip">B is across the line: the answer flips</span>' if q["flip"]
           else '<span class="tag tag-same">B is on the camera\'s side: same answer</span>')
    return f"""
<article class="card">
  <header class="card-h">{tag}</header>
  <div class="trio">
    <figure><img class="sw" data-plain="{a_plain}" data-outline="{a_out}" src="{a_out}" alt="Photo from spot A">
      <figcaption><b>What the model sees</b> (photo from spot A). From here, X is <b>{esc(q['relA'])}</b> of Y.</figcaption></figure>
    <figure><img class="sw" data-plain="{b_plain}" data-outline="{b_out}" src="{b_out}" alt="View from B's spot">
      <figcaption><b>The view from B's spot</b> (never shown to the model). From here, X is <b>{esc(q['relB'])}</b> of Y.</figcaption></figure>
    <figure>{sketch(q)}<figcaption><b>Map from above.</b> Shaded wedges show where A and B look; the dashed line runs through X and Y.</figcaption></figure>
  </div>
  <div class="roles"><span class="role role-B">B = {esc(q['B'])}</span><span class="role role-X">X = {esc(q['X'])}</span><span class="role role-Y">Y = {esc(q['Y'])}</span></div>
  <p class="prompt">“{esc(PROMPT.format(B=q['B'], X=q['X'], Y=q['Y']))}”</p>
  <p class="said">Correct answer (from B's spot): <b>{esc(q['relB'])}</b>. What the models said: {said}.</p>
</article>"""


def shortcut_section():
    summ = json.loads((LOCAL / "v4_data_summary.json").read_text())
    sc, single = summ["shortcut"], summ["shortcut_single"]
    lab = [("all", "All bookkeeping facts together"), ("distances+angles", "Distances and angles"), ("rank", "B's depth rank"),
           ("side", "B's side of the picture"), ("rank+side", "Depth rank + side"), ("categories", "The three object categories")]
    rows = "".join(f'<tr><th>{l}</th><td class="num">{pc(sc[k]["relB_within_A"])}</td><td class="num">{pc(sc[k]["flip"])}</td></tr>'
                   for k, l in lab)
    top = max(single.items(), key=lambda kv: kv[1])
    return rows, top, sc["all"]["relB_within_A"], sc["all"]["flip"]


# ---------------------------------------------------------------- page sections

def peaks(an, t):
    best = max(an["layers"], key=lambda r: r["targets"][t]["score"])
    return best["targets"][t]["score"], best["layer"], best["targets"][t]["ci"]


def reader_charts(D):
    out = []
    for m in MODELS:
        an = D[m]["an"]
        Ls = [r["layer"] for r in an["layers"]]
        B = [r["targets"]["B"] for r in an["layers"]]
        series = [{"xs": Ls, "ys": [b["score"] for b in B], "lo": [b["ci"][0] for b in B], "hi": [b["ci"][1] for b in B],
                   "cls": "s1", "label": "B's answer"},
                  {"xs": Ls, "ys": [r["targets"]["A"]["score"] for r in an["layers"]], "cls": "s-ink", "label": "A's answer", "dash": True},
                  {"xs": Ls, "ys": [r["text_only"]["score"] for r in an["layers"]], "cls": "s-mut", "label": "no picture"}]
        if m == "qwen3vl":
            series.insert(1, {"xs": list(range(37)), "ys": [v[0] / 100 for v in PHD["layers"]["B"]], "cls": "s2",
                              "label": "PhD: B's answer", "dash": True})
        band = {"xs": Ls, "lo": [r["perm"][0] for r in an["layers"]], "hi": [r["perm"][1] for r in an["layers"]],
                "cls": "b-perm", "title": "Luck band: readers trained on shuffled answers"}
        n = Ls[-1]
        xt = list(range(0, n + 1, 8 if n > 40 else 4))
        out.append(f'<figure class="panel"><figcaption class="ph">{NAME[m]} <span>({n} layers)</span></figcaption>'
                   + line_chart(series, 0, n, xt, "layer", "reader score on new rooms", y0=.3, y1=1,
                                yticks=(.3, .5, .7, .9, 1), w=900, h=280, bands=[band],
                                hlines=[{"y": .5, "cls": "chance", "label": "chance (50%)"}],
                                title=f"Reader scores by layer, {NAME[m]}", fmt_x=str) + "</figure>")
    return "".join(out)


def lens_charts(D):
    out = []
    for m in MODELS:
        L = D[m]["lens"]["layers"]
        xs = [r["layer"] for r in L]
        series = [{"xs": xs, "ys": [r["says_A"] for r in L], "cls": "s-ink", "label": "A's answer", "dash": True},
                  {"xs": xs, "ys": [r["says_B_on_flip"] for r in L], "cls": "s1", "label": "B's answer"}]
        n = xs[-1]
        out.append(f'<figure class="panel"><figcaption class="ph">{NAME[m]}</figcaption>'
                   + line_chart(series, 0, n, list(range(0, n + 1, 8 if n > 40 else 4)), "layer",
                                "share of flip questions", w=900, h=240,
                                hlines=[{"y": .5, "cls": "chance", "label": "undecided (50%)"}],
                                title=f"Logit lens, {NAME[m]}") + "</figure>")
    return "".join(out)


def sweep_rows(D):
    pos = [("name_B", "B's name (early in the sentence)"), ("name_X", "X's last mention (in the question)"),
           ("name_Y", "Y's last mention (end of the question)"), ("patch_B", "B's picture patches"),
           ("patch_X", "X's picture patches"), ("patch_Y", "Y's picture patches")]
    rows = []
    for key, lab in pos:
        cells = []
        for m in MODELS:
            best = max(D[m]["an"]["layers"], key=lambda r: r["sweep"][key]["score"])
            cells.append(f'<td class="num">{pc(best["sweep"][key]["score"])} <small>L{best["layer"]}</small></td>')
        rows.append(f"<tr><th>{esc(lab)}</th>{''.join(cells)}</tr>")
    cells = []
    for m in MODELS:
        s, L, _ = peaks(D[m]["an"], "B")
        cells.append(f'<td class="num"><b>{pc(s)}</b> <small>L{L}</small></td>')
    rows.append(f"<tr class='hi'><th>Just before the answer (the main reader)</th>{''.join(cells)}</tr>")
    cells = []
    for m in MODELS:
        best = max(D[m]["an"]["layers"], key=lambda r: r["three_objects"]["score"])
        cells.append(f'<td class="num">{pc(best["three_objects"]["score"])} <small>L{best["layer"]}</small></td>')
    rows.append(f"<tr><th>B, X and Y picture patches side by side {ADD}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def controls_rows(D):
    def mx(m, f):
        return max(f(r) for r in D[m]["an"]["layers"])
    spec = [
        ("Luck band (highest)", "Readers trained on shuffled answers. Shows what a reader scores by pure luck.", "≤ 60%",
         lambda m: mx(m, lambda r: r["perm"][1]), PHD["controls"]["perm_max"]),
        ("Shuffled within each room (highest) " + CHG, "Answers shuffled among questions from the same room. A reader that only memorised rooms would still score well here.",
         "≤ 62%", lambda m: mx(m, lambda r: r["scrambled"]["max"]), PHD["controls"]["scrambled_max"]),
        ("No picture (highest)", "The same question with the picture removed. Rules out the wording giving the answer away.", "≤ 55%",
         lambda m: mx(m, lambda r: r["text_only"]["score"]), PHD["controls"]["text_only_max"]),
        ("Layer 0", "The model's starting numbers, before any processing.", "≤ 55%",
         lambda m: D[m]["an"]["layers"][0]["targets"]["B"]["score"], PHD["controls"]["layer0"]),
        ("Picture features, whole image", "What the image encoder hands to the language model, averaged over the whole picture.", "≤ 55%",
         lambda m: D[m]["an"]["vision"]["vision_whole"]["score"], PHD["controls"]["vision_whole"]),
        ("Picture features, per object " + ADD, "The same features, but for B, X and Y separately and side by side. A fairer test of whether the picture already contains the answer.",
         "for information", lambda m: D[m]["an"]["vision"]["vision_objects"]["score"], None),
    ]
    rows = []
    for lab, what, mark, f, phd in spec:
        cells = "".join(f'<td class="num">{pc(f(m))}</td>' for m in MODELS)
        rows.append(f'<tr><th>{lab}<small class="what">{esc(what)}</small></th><td class="num mark">{mark}</td>{cells}'
                    f'<td class="num phd">{"–" if phd is None else str(phd) + "%"}</td></tr>')
    rows.append('<tr class="hi"><th>The real reader: B\'s answer (best layer)</th><td></td>'
                + "".join(f'<td class="num"><b>{pc(peaks(D[m]["an"], "B")[0])}</b></td>' for m in MODELS)
                + '<td class="num phd"><b>86%</b></td></tr>')
    return "".join(rows)


def six_rows(D):
    spec = [("A", "A's answer", "the correct answer from the camera's spot", "50%"),
            ("model", "The model's own answer", "what the model is about to say", "50%"),
            ("B", "B's answer", "the correct answer from B's spot (scored within A's answer)", "50%"),
            ("flip", "Flips?", "whether B's answer differs from A's", "50%"),
            ("rank", "B's depth rank", "is B the nearest, middle or farthest of the three objects from the camera", "33%"),
            ("side", "B's side of the picture", "does B appear on the left or the right of the photo", "50%")]
    rows = []
    for t, lab, what, ch in spec:
        cells = "".join(f'<td class="num">{pc(peaks(D[m]["an"], t)[0])} <small>L{peaks(D[m]["an"], t)[1]}</small></td>' for m in MODELS)
        phd = max(v[0] for v in PHD["layers"][t])
        rows.append(f'<tr{" class=hi" if t == "B" else ""}><th>{lab}<small class="what">{esc(what)}</small></th>'
                    f'<td class="num mark">{ch}</td>{cells}<td class="num phd">{phd}%</td></tr>')
    return "".join(rows)


def behaviour_rows(D):
    spec = [("gives_B", "Gives B's answer (the task score)", "gives_B"),
            ("gives_A", "Gives A's answer (the camera's view)", "gives_A"),
            ("flip_q", "Flip questions answered correctly", None),
            ("noflip_q", "No-flip questions answered correctly", None),
            ("direct_A", "Plain question on the photo (\"is X left or right of Y in this image?\")", "direct_A"),
            ("direct_B", "Plain question on the photo taken from B's spot", "direct_B")]
    rows = []
    for key, lab, bkey in spec:
        cells = []
        for m in MODELS:
            if bkey:
                v = D[m]["an"]["behaviour"]["test"][bkey]
            else:
                rs = [r for r in D[m]["rows"] if r["flip"] == (key == "flip_q")]
                v = sum(r["answer"] == r["relB"] for r in rs) / len(rs)
            cells.append(f'<td class="num">{pc(v)}</td>')
        rows.append(f'<tr><th>{esc(lab)}</th>{"".join(cells)}<td class="num phd">{pc(PHD["behaviour_test"][key])}</td></tr>')
    return "".join(rows)


CSS = """
:root{
  /* layout: one reading column (~70ch) with figures allowed to run wider; quiet panels only for figures and tables */
  --bg:#f6f7f9; --panel:#ffffff; --ink:#15181d; --ink2:#4a5260; --ink3:#6f7785; --rule:#dde1e7;
  --accent:#2a6fdb; --add:#6250d6; --addbg:#efedfc; --chg:#9a6700; --chgbg:#fdf3dc;
  --ok:#1d7a46; --okbg:#e3f3e9;
  --B:#2ea043; --X:#eb7a1e; --Y:#286ee6;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --sink:#3b414b; --smut:#9aa1ab;
  --sk-bg:#fbfbfa; --sk-grid:#e7e9ec; --hot:#fff1c2;
  --f-display:"Fraunces",Georgia,serif; --f-body:"Public Sans",system-ui,sans-serif; --f-mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#121417; --panel:#1b1e23; --ink:#eceef1; --ink2:#b9c0ca; --ink3:#8d95a1; --rule:#2e333b;
  --accent:#6ea0f0; --add:#a59bf0; --addbg:#26224a; --chg:#e0b048; --chgbg:#3a2f12; --ok:#5cc98a; --okbg:#173325;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --sink:#c9ced6; --smut:#6f7785;
  --sk-bg:#22262c; --sk-grid:#30353d; --hot:#4a3f12; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121417; --panel:#1b1e23; --ink:#eceef1; --ink2:#b9c0ca; --ink3:#8d95a1; --rule:#2e333b;
  --accent:#6ea0f0; --add:#a59bf0; --addbg:#26224a; --chg:#e0b048; --chgbg:#3a2f12; --ok:#5cc98a; --okbg:#173325;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --sink:#c9ced6; --smut:#6f7785;
  --sk-bg:#22262c; --sk-grid:#30353d; --hot:#4a3f12; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.6;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 64px}
.wrap{max-width:1000px;margin:0 auto;display:grid;gap:56px}
.col{max-width:70ch}
h1{font-family:var(--f-display);font-weight:600;font-size:clamp(32px,5vw,48px);line-height:1.08;margin:0;text-wrap:balance}
h2{font-family:var(--f-display);font-weight:600;font-size:clamp(24px,3vw,30px);line-height:1.15;margin:0 0 10px;text-wrap:balance}
h3{font-size:18px;margin:26px 0 6px;text-wrap:balance}
p{margin:0 0 12px} section p, section li{max-width:70ch}
a{color:var(--accent)}
.kicker{font:500 12px/1 var(--f-mono);letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);margin-bottom:10px}
.lede{font-size:19px;color:var(--ink2);max-width:62ch;margin-top:14px}
.byline{color:var(--ink3);font-size:14px;margin-top:10px}
.summary{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 18px 16px 36px;margin:22px 0 0;max-width:80ch}
.summary li{margin:0 0 10px;color:var(--ink2)} .summary li:last-child{margin-bottom:0}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin-top:22px}
.finding{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;min-width:0}
.finding b.n{display:block;font:600 30px/1.1 var(--f-mono);font-variant-numeric:tabular-nums;margin-bottom:6px}
.finding p{margin:0;color:var(--ink2);font-size:15px}
.howto{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-top:18px;font-size:14px;color:var(--ink2)}
.mk{font:600 11px/1 var(--f-mono);letter-spacing:.05em;text-transform:uppercase;padding:3px 6px;border-radius:4px;vertical-align:2px;white-space:nowrap}
.mk.add{background:var(--addbg);color:var(--add)} .mk.chg{background:var(--chgbg);color:var(--chg)}
.setup-t{margin:4px 0 0;font-size:.92em}.setup-t th{text-align:left;vertical-align:top;white-space:nowrap;padding:4px 14px 4px 0;font-weight:600}.setup-t td{padding:4px 0;vertical-align:top}
@media (max-width:600px){.setup-t th,.setup-t td{display:block;padding:2px 0}.setup-t td{padding-bottom:8px}}
.box{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;max-width:70ch}
.box.add{border-left:4px solid var(--add)} .box.chg{border-left:4px solid var(--chg)}
.box h4{margin:0 0 6px;font-size:15px}
details h4{margin:16px 0 4px;font-size:15px}
.box p:last-child,.box ul:last-child{margin-bottom:0}
.analogy{border-left:3px solid var(--rule);padding:2px 0 2px 14px;color:var(--ink2);font-style:italic;max-width:66ch}
details{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:10px 16px;max-width:70ch}
details summary{cursor:pointer;font-weight:600;font-size:15px}
details[open] summary{margin-bottom:8px}
details p,details li{font-size:15px;color:var(--ink2)}
.figs{display:grid;grid-template-columns:1fr;gap:16px}
.figs3{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px;margin-top:10px}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:12px 12px 6px;margin:0;min-width:0}
.ph{font-weight:600;font-size:15px;margin:0 0 4px 4px} .ph span{color:var(--ink3);font-weight:400}
.chart,.diagram{width:100%;height:auto;display:block}
.diagram{max-width:640px;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:8px}
.grid{stroke:var(--rule);stroke-width:1} .tick{fill:var(--ink3);font:11px var(--f-mono)} .axl{fill:var(--ink3);font:12px var(--f-body)}
.hl{stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 4}
.ln{fill:none;stroke-width:2;stroke-linejoin:round} .ln.dash{stroke-dasharray:6 4}
.ln.s1{stroke:var(--s1)} .ln.s2{stroke:var(--s2)} .ln.s3{stroke:var(--s3)} .ln.s-ink{stroke:var(--sink)} .ln.s-mut{stroke:var(--smut)}
.ci{stroke:none;opacity:.18} .ci.s1{fill:var(--s1)} .ci.s2{fill:var(--s2)} .ci.s3{fill:var(--s3)} .ci.s-mut{fill:var(--smut)} .ci.s-ink{fill:var(--sink)}
.band{stroke:none} .b-perm{fill:var(--smut);opacity:.28}
.hit{fill:transparent;stroke:none} .hit:hover{fill:var(--ink);opacity:.15}
.dl{font:600 12px var(--f-body)} .dl.s1{fill:var(--s1)} .dl.s2{fill:var(--s2)} .dl.s3{fill:var(--s3)} .dl.s-ink{fill:var(--sink)} .dl.s-mut{fill:var(--ink3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:8px 0 0}
.legend i{display:inline-block;width:18px;height:0;border-top:3px solid;vertical-align:middle;margin-right:6px}
.legend i.dash{border-top-style:dashed}
.sw-s1{border-color:var(--s1)!important} .sw-s2{border-color:var(--s2)!important} .sw-s3{border-color:var(--s3)!important}
.sw-s-ink{border-color:var(--sink)!important} .sw-s-mut{border-color:var(--smut)!important}
.sw-band{border-top:8px solid var(--smut)!important;opacity:.5}
.verdict{display:inline-block;font:600 11px var(--f-mono);letter-spacing:.04em;padding:2px 8px;border-radius:999px;text-transform:uppercase}
.verdict.no{background:var(--chgbg);color:var(--chg)} .verdict.yes{background:var(--okbg);color:var(--ok)}
.tw{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:14px;min-width:560px;width:100%;background:var(--panel);border:1px solid var(--rule);border-radius:10px}
th,td{padding:9px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
thead th{font:600 12px var(--f-mono);letter-spacing:.04em;color:var(--ink3);text-transform:uppercase;vertical-align:bottom}
tbody th{font-weight:600;min-width:200px}
td.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.num small{color:var(--ink3)} td.mark{color:var(--ink3)} td.phd{color:var(--ink2)}
small.what{display:block;font-weight:400;color:var(--ink3);font-size:13px;margin-top:2px;max-width:46ch}
small.ci{display:block;font-size:11px} small.brk{display:block;font-size:11px;color:var(--chg)}
tr.hi{background:var(--hot)}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px;display:grid;gap:12px;min-width:0}
.card-h{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tag{font:600 12px var(--f-mono);padding:3px 8px;border-radius:4px}
.tag-flip{background:var(--chgbg);color:var(--chg)} .tag-same{background:var(--okbg);color:var(--ok)}
.trio{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:12px}
.trio figure{margin:0;min-width:0} .trio img,.sketch{width:100%;height:auto;border-radius:6px;display:block}
figcaption{font-size:13px;color:var(--ink2);margin-top:6px}
.roles{display:flex;flex-wrap:wrap;gap:8px}
.role{font:500 13px var(--f-mono);padding:2px 8px;border-radius:4px;border:2px solid}
.role-B{border-color:var(--B)} .role-X{border-color:var(--X)} .role-Y{border-color:var(--Y)}
.prompt{margin:0;font-size:14px;color:var(--ink2);font-style:italic}
.said{margin:0;font-size:14px}
.sk-bg{fill:var(--sk-bg)} .sk-grid{stroke:var(--sk-grid);stroke-width:1}
.sk-line{stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:5 4;fill:none}
.sk-cone{stroke:none;opacity:.18} .sk-cone-A{fill:var(--ink)} .sk-cone-B{fill:var(--B)}
.sk-obj{stroke-width:2;fill-opacity:.25} .sk-B{stroke:var(--B);fill:var(--B)} .sk-X{stroke:var(--X);fill:var(--X)} .sk-Y{stroke:var(--Y);fill:var(--Y)}
.sk-lab{font:600 12px var(--f-mono);fill:var(--ink);text-anchor:middle} .sk-A{fill:var(--ink)} .sk-lab-A{fill:var(--bg)}
.sk-note{font:11px var(--f-body);fill:var(--ink3)}
.dg-t{font:600 13px var(--f-body);fill:var(--ink)} .dg-c{font:12px var(--f-body);fill:var(--ink2)} .dg-big{font:600 26px var(--f-body);fill:var(--ink2)}
.dg-l{font:600 12px var(--f-mono);fill:#fff;text-anchor:middle} .dg-lA{fill:var(--bg)}
.dg-line{stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:6 4}
.dg-X{fill:var(--X)} .dg-Y{fill:var(--Y)} .dg-B{fill:var(--B)} .dg-A{fill:var(--ink)}
.dg-cone{fill:var(--ink);opacity:.12} .dg-coneB{fill:var(--B);opacity:.22}
.dg-cell{fill:var(--sk-bg);stroke:var(--rule)} .dg-cell.hot{fill:var(--hot);stroke:var(--chg)}
.dg-p1{fill:var(--s1);opacity:.55} .dg-p2{fill:var(--s2);opacity:.55} .dg-mean{fill:var(--ink)}
.dg-arrow{stroke:var(--ink);stroke-width:2} .dg-ahead{fill:var(--ink)}
.sc-bg{fill:var(--sk-bg)} .sc-axis{stroke:var(--ink3);stroke-width:1}
.sc-L{fill:var(--s1);opacity:.6} .sc-R{fill:var(--s2);opacity:.6}
.dg-drop{stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 4}
.dg-vec{stroke-width:2.5} .dg-ink{stroke:var(--sink)} .dg-s1{stroke:var(--s1)} .dg-s2{stroke:var(--s2)}
.promptbox{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;font-size:15px;line-height:2;max-width:70ch}
.pt-img{font-family:var(--f-mono);font-size:13px;color:var(--ink3)}
.pt-mark{font-weight:700;padding:1px 5px;border-radius:4px;background:var(--hot)}
.pt-4{font-family:var(--f-mono);font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:4px 0 10px}
.controls button{font:500 13px var(--f-body);border:1px solid var(--rule);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:999px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.controls button:focus-visible,.controls input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.controls label{font-size:14px;color:var(--ink2);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.controls input[type=range]{width:min(320px,70vw)}
#steer-note{font-size:14px;color:var(--ink2);margin:6px 0 0}
.eq{font-family:var(--f-mono);font-size:14px;background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 14px;overflow-x:auto;max-width:70ch}
ul.tight{padding-left:20px;margin:0 0 12px} ul.tight li{margin:4px 0}
.diff li{margin:6px 0}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
"""

JS = r"""
(function(){
  // outline / plain photo switch
  const btns = document.querySelectorAll('[data-photo]');
  btns.forEach(b => b.addEventListener('click', () => {
    const mode = b.dataset.photo;
    btns.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    document.querySelectorAll('img.sw').forEach(i => { i.src = i.dataset[mode]; });
  }));

  // interactive steering chart (redesigned run): shift in the log-odds toward B's answer
  const S = JSON.parse(document.getElementById('steer-data').textContent);
  const NAMES = {qwen3vl:'Qwen3-VL-8B', gemma4:'Gemma 4 12B', bagel:'BAGEL-7B-MoT'};
  const DIRS = [
    {k:'B_perp', label:'B ⊥ G', cls:'s1', ci:true},
    {k:'D_perp', label:'depth ⊥ G', cls:'s3', ci:false},
    {k:'A_perp', label:'A ⊥ G', cls:'s2', ci:false},
    {k:'B', label:"B's full direction", cls:'s-ink', ci:false, dash:true},
  ];
  let model = 'qwen3vl', li = S.qwen3vl.layers.indexOf(S.qwen3vl.best);
  const svg = document.getElementById('steer-svg'), slider = document.getElementById('steer-layer'),
        lab = document.getElementById('steer-layer-lab'), note = document.getElementById('steer-note');
  const W = 640, H = 320, ml = 52, mr = 150, mt = 12, mb = 44, pw = W-ml-mr, ph = H-mt-mb;
  const X = v => ml + Math.log2(v) / 5 * pw;
  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text){ const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); if (text != null) e.textContent = text; return e; }
  function f2(v){ return (v >= 0 ? '+' : '') + v.toFixed(2); }
  function pct(v){ return Math.round(v*100) + '%'; }
  function draw(){
    const s = S[model], L = s.layers[li], ks = s.strengths;
    let lim = 1;
    DIRS.forEach(d => s.dirs[d.k][li].forEach(p => { lim = Math.max(lim, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])); }));
    s.dirs.rand[li].forEach(p => { lim = Math.max(lim, Math.abs(p[0]), Math.abs(p[1])); });
    const nice = [1, 2, 5, 10, 20, 50]; const top = nice.find(n => n >= lim * 1.05) || Math.ceil(lim);
    const Y = v => mt + (1 - (v + top) / (2 * top)) * ph;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    [-top, -top/2, 0, top/2, top].forEach(t => { svg.appendChild(el('line',{x1:ml,x2:ml+pw,y1:Y(t),y2:Y(t),class:t===0?'hl':'grid'})); svg.appendChild(el('text',{x:ml-6,y:Y(t)+4,class:'tick','text-anchor':'end'},f2(t))); });
    ks.forEach(k => svg.appendChild(el('text',{x:X(k),y:mt+ph+16,class:'tick','text-anchor':'middle'},k+'×')));
    svg.appendChild(el('text',{x:ml+pw/2,y:H-6,class:'axl','text-anchor':'middle'},"push size (multiples of B's gap)"));
    svg.appendChild(el('text',{x:14,y:mt+ph/2,class:'axl','text-anchor':'middle',transform:`rotate(-90 14 ${mt+ph/2})`},"toward B's answer (logits)"));
    const R = s.dirs.rand[li];
    const topPts = ks.map((k,i)=>X(k)+','+Y(R[i][1])).join(' '), botPts = ks.slice().reverse().map((k,j)=>{const i=ks.length-1-j; return X(k)+','+Y(R[i][0]);}).join(' ');
    svg.appendChild(el('polygon',{points:topPts+' '+botPts,class:'ci s-mut'}));
    svg.appendChild(el('path',{d:ks.map((k,i)=>(i?'L':'M')+X(k)+','+Y(R[i][2])).join(' '),class:'ln s-mut'}));
    ks.forEach((k,i)=>{ const c = el('circle',{cx:X(k),cy:Y(R[i][2]),r:7,class:'hit'}); c.appendChild(el('title',{},`10 random ⊥ G, ${k}×: mean ${f2(R[i][2])}, range ${f2(R[i][0])} to ${f2(R[i][1])}`)); svg.appendChild(c); });
    const placed = [];
    const labelAt = (y, txt, cls) => { let ly = y; placed.sort((a,b)=>a-b); for (const p of placed) if (Math.abs(ly-p) < 14) ly = p + 14; placed.push(ly); svg.appendChild(el('text',{x:X(32)+8,y:ly+4,class:'dl '+cls},txt)); };
    labelAt(Y(R[ks.length-1][2]), 'random (range, mean)', 's-mut');
    DIRS.forEach(d => {
      const pts = s.dirs[d.k][li];
      if (d.ci){
        const t = ks.map((k,i)=>X(k)+','+Y(pts[i][2])).join(' '), b = ks.slice().reverse().map((k,j)=>{const i=ks.length-1-j; return X(k)+','+Y(pts[i][1]);}).join(' ');
        svg.appendChild(el('polygon',{points:t+' '+b,class:'ci '+d.cls}));
      }
      svg.appendChild(el('path',{d:ks.map((k,i)=>(i?'L':'M')+X(k)+','+Y(pts[i][0])).join(' '),class:'ln '+d.cls+(d.dash?' dash':'')}));
      ks.forEach((k,i)=>{ const c = el('circle',{cx:X(k),cy:Y(pts[i][0]),r:7,class:'hit'}); c.appendChild(el('title',{},`${d.label}, ${k}×: ${f2(pts[i][0])} (95% interval ${f2(pts[i][1])} to ${f2(pts[i][2])}); ${pct(pts[i][3])} broken`)); svg.appendChild(c); });
      labelAt(Y(pts[ks.length-1][0]), d.label, d.cls);
    });
    const g32 = s.dirs.g[li][ks.length-1];
    lab.textContent = 'layer ' + L + (L === s.best ? ' (headline layer)' : '') + (s.reliable[li] ? '' : ' (B not reliable here yet)');
    note.textContent = `${NAMES[model]}, all 1,128 test questions. Shaded grey: the range of 10 random realistic directions. Positive control g at 32×: ${f2(g32[0])} logits. Hover a point for exact numbers.`;
  }
  function setModel(m){
    model = m; const s = S[m];
    slider.max = s.layers.length - 1; li = s.layers.indexOf(s.best); slider.value = li;
    document.querySelectorAll('[data-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.model === m)));
    draw();
  }
  document.querySelectorAll('[data-model]').forEach(b => b.addEventListener('click', () => setModel(b.dataset.model)));
  slider.addEventListener('input', () => { li = +slider.value; draw(); });
  setModel('qwen3vl');
})();
"""


def main():
    D = load()
    S = {m: steer2_table(D[m]) for m in MODELS}
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(LOCAL / "test" / "questions_test.jsonl")}
    rows = {m: {r["qid"]: r for r in D[m]["rows"]} for m in MODELS}
    examples = "".join(example(qs[q], rows) for q in EXAMPLES)
    an = {m: D[m]["an"] for m in MODELS}
    bpk = {m: peaks(an[m], "B") for m in MODELS}
    flipB = {m: sum(r["answer"] == r["relB"] for r in D[m]["rows"] if r["flip"]) / sum(1 for r in D[m]["rows"] if r["flip"]) for m in MODELS}
    nq = len(D["qwen3vl"]["rows"])
    nflip = sum(1 for r in D["qwen3vl"]["rows"] if r["flip"])
    cos_rows = ""
    for m in MODELS:
        L = an[m]["best_layer"]
        lr_ = D[m]["lens"]["layers"][L]["cos_readout"]
        cos_rows += (f'<tr><th>{NAME[m]} <small>L{L}</small></th><td class="num">{lr_["B"]:+.2f}</td>'
                     f'<td class="num">{lr_["say"]:+.2f}</td><td class="num">{lr_["B_normal"]:+.2f}</td></tr>')
    checks = " · ".join(f'{SHORT[m]} {pc(D[m]["s2"]["steer"]["check"]["cached_step_matches_full_pass"], 1)}' for m in MODELS)
    hl2 = {m: D[m]["s2"]["steer"]["headline_layer"] for m in MODELS}
    vo = {m: an[m]["vision"]["vision_objects"]["score"] for m in MODELS}
    sc_rows, sc_top, sc_all_B, sc_all_flip = shortcut_section()
    ds_all = an["qwen3vl"]["vision"].get("deepstack_all_objects", {}).get("score")

    page = f"""<meta charset="utf-8">
<title>Seeing From Where B Stands</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap">
<style>{CSS}</style>
<div class="wrap">

<header>
  <div class="kicker">Perspective probe v4 · replication and follow-ups · November 2026</div>
  <h1>Seeing From Where B Stands</h1>
  <p class="lede">Can an AI that looks at a photo imagine how the scene looks from somewhere else? We asked three
  vision-language models, then looked inside them. This page repeats the PhD student's
  <a href="{PHD_URL}">v4 probe</a> step by step, extends it to two more models, and adds two experiments of our own.
  It is written for readers who know the basics of neural networks and linear classifiers but not this project: every setting
  and every check is explained.</p>
  <ul class="summary">
    <li>When asked what an object B would see, three vision-language models (Qwen3-VL, Gemma 4, BAGEL) almost always answer from the camera's
    view instead, giving B's answer on only 9–15% of the questions where the two views differ.</li>
    <li>Yet a linear probe recovers B's correct answer from their hidden states at 80–86% in rooms it never saw, and it passes every control.
    This replicates the PhD's 86% on Qwen exactly and extends it to two more models.</li>
    <li>A logit lens shows that this signal never reaches the output: at the layers where the probe peaks, B's answer sits almost perpendicular
    to the direction the model reads its left/right choice from, so the output stays with the camera's answer.</li>
    <li>Steering shows the signal is not used. At its natural size, B's direction moves the answer no more than a random realistic
    direction, while the camera's direction has a strong route to the output. Pushed 16–32 times harder, with its answer-moving part
    removed, it still does no more than random pushes of the same size: the models compute B's view, but their output is wired to the camera's.</li>
  </ul>
  <div class="howto"><span>How to read the labels:</span><span>{ADD} something we added to the PhD's method</span>
    <span>{CHG} something we did differently</span><span>unlabelled: the same as the PhD's method</span></div>
</header>

<section id="question">
  <div class="kicker">1 · The question</div>
  <h2>What does it mean to take someone else's viewpoint?</h2>
  <p>Sit across a table from a friend. The cup on your left is on their right. You can work that out without moving,
  by imagining yourself in their seat. Psychologists call this <b>perspective taking</b>. Children learn it around age four to six,
  and it is something we would like AI systems that work with people to do as well.</p>
  <p>This study tests whether <b>vision-language models</b> can do it.</p>

  <h3>The task</h3>
  <p>A computer program builds an empty room and places three pieces of furniture on the floor at random. It takes a photo from a random
  spot, which we call <b>A</b> (think of it as the camera, or "you"). We name one of the three objects <b>B</b> and the other two
  <b>X</b> and <b>Y</b>, and ask the model:</p>
  <p class="analogy">“Imagine you are standing where B is, looking toward X and Y. From there, is X to the left or to the right of Y?”</p>
  <p>To answer correctly the model has to imagine standing at B. Here are two real questions from the test set. The coloured outlines are
  only for you; the model saw the plain photo.</p>
  <div class="controls" role="group" aria-label="Photo display">
    <button type="button" data-photo="outline" aria-pressed="true">With outlines</button>
    <button type="button" data-photo="plain" aria-pressed="false">Plain photo (what the model saw)</button>
  </div>
  <div style="display:grid;gap:16px">{examples}</div>

  <h3>Why the answer sometimes flips</h3>
  <p>Draw a straight line through X and Y. If B stands on the <b>same side</b> of that line as the camera, then left and right are the same
  from both spots. If B stands on the <b>other side</b>, left and right swap, just like you and your friend across the table.</p>
  {flip_diagram()}
  <p>The test set has {nq} questions in 9 rooms, and exactly half of them ({nflip}) flip. That balance matters: a model that always
  answers from the camera's view gets exactly 50% right, which is the same as guessing. So the task score only rises above 50%
  if a model actually uses B's viewpoint.</p>
  <h3>Bookkeeping facts, and the shortcut check</h3>
  <p>For every question we also record simple facts about the scene, called <b>bookkeeping facts</b>:</p>
  <ul class="tight">
    <li><b>B's depth rank</b>: whether B is the nearest, middle or farthest of the three objects from the camera.</li>
    <li><b>B's side of the picture</b>: whether B appears in the left or right half of A's photo.</li>
    <li><b>Distances</b>: A to B, A to X, A to Y, B to X, B to Y, X to Y, and B to the X–Y line.</li>
    <li><b>Angles</b>: how far apart X and Y look from A and from B; how far B, X and Y each sit from the centre of A's view.</li>
    <li><b>Categories</b>: what kind of furniture B, X and Y are.</li>
  </ul>
  <p><b>Why they matter.</b> If any simple fact gave away B's answer (for example "when B is far away, the answer usually flips"), a model, or one
  of our readers, could score well by using that fact instead of B's viewpoint. The data are built so that this cannot happen:</p>
  <ul class="tight">
    <li><b>Balancing.</b> The four combinations of A's answer and B's answer (left/left, left/right, right/left, right/right) appear equally
    often, and exactly half the questions flip within every depth-rank group and every picture-side group. So depth rank and picture side
    on their own say nothing about B's answer.</li>
    <li><b>The PhD's shortcut check.</b> A logistic regression is trained to predict B's answer from all bookkeeping facts on the training
    questions, then scored on the test questions within A's answer. The pass mark is 60%. Ours scores <b>{pc(sc_all_B)}</b> (the PhD's: 51%).</li>
    <li>{ADD} <b>The same check for the flip.</b> We also predict "does it flip?" from each single fact, with a pass mark of 65%. The highest
    is {esc(sc_top[0])} at {pc(sc_top[1])}. With all facts together the flip is predictable ({pc(sc_all_flip)}), which is expected: the flip
    is a geometric fact, fully set by where everything stands. What matters is that no single simple cue gives it away, so a model has to
    combine the geometry, which is the point of the task.</li>
  </ul>
  <div class="tw"><table>
    <thead><tr><th>Predicting from…</th><th>B's answer (within A's)</th><th>Flips?</th></tr></thead>
    <tbody>{sc_rows}</tbody></table></div>
  <p style="margin-top:10px">All scores are on the test set; 50% is chance.</p>

  <details><summary>How the scenes were made (and checked)</summary>
    <p>The rooms come from ReplicaCAD (a set of 3D apartment models) emptied of furniture; the objects come from the HSSD collection
    of 3D furniture. Training questions use 8 kinds of object in 25 rooms; test questions use 5 different kinds in 9 different rooms,
    so nothing in the test set was seen during training. Every answer was computed twice, from the rendered images and from the geometry,
    and the two had to agree. Questions where the answer would be hard even for a person were removed (for example when, from B's spot,
    X and Y nearly line up). The full data check is in the <a href="{DATA_URL}">data report</a>.</p>
    <p>{CHG} We added three picture-quality rules the PhD's examples did not seem to use: no object may be cut off by the photo's edge
    or more than 10% hidden behind another; in the view from B's spot, X and Y may not touch; and we kept more "B nearest" and
    "B farthest" questions, so our test set has 32 / 324 / 772 nearest / middle / farthest questions against the PhD's 24 / 352 / 756.</p>
  </details>

  <h3>The models</h3>
  <p><b>{NAME['qwen3vl']}</b> is the model the PhD tested. {ADD} We also tested <b>{NAME['gemma4']}</b> (Google) and
  <b>{NAME['bagel']}</b> (ByteDance), two models built in different ways, to see whether the result holds beyond one model.</p>
  <div class="box setup">
    <h4>Setup</h4>
    <table class="setup-t"><tbody>
      <tr><th>Checkpoints</th><td>Qwen3-VL-8B-Instruct, gemma-4-12B-it, BAGEL-7B-MoT; Hugging Face weights, bf16, no fine-tuning</td></tr>
      <tr><th>Prompt</th><td>The same text for every model (shown above), image first, then text, as one user turn in each model's own chat template.
        No system prompt. Thinking is off: Qwen is the Instruct (non-thinking) model, Gemma uses its default template with thinking off,
        and BAGEL runs in understanding mode without its thinking step.</td></tr>
      <tr><th>Answer</th><td>Greedy decoding. The answer is the model's first token, read as "left" or "right" (if it is not, a few more tokens are decoded
        and parsed). No answer was unreadable for any model.</td></tr>
      <tr><th>Images</th><td>512 × 512 renders, 90° field of view, camera 1.5 m above the floor. A faces the middle of the three objects;
        B faces the midpoint of X and Y. Each model resizes the image its own way: 256 image tokens for Qwen and Gemma.</td></tr>
      <tr><th>Data</th><td>Training: 1,516 questions, 1,436 photos, 25 rooms. Test: 1,128 questions, 1,081 photos, 9 rooms.</td></tr>
      <tr><th>Hidden states</th><td>Saved at every layer, in one forward pass per question, with no generation in between.</td></tr>
    </tbody></table>
  </div>
</section>

<section id="behaviour">
  <div class="kicker">2 · What the models say</div>
  <h2>Out loud, they answer from the camera</h2>
  <p>First we simply asked each model every question and recorded its answer.</p>
  <div class="tw"><table>
    <thead><tr><th>Test set ({nq} questions)</th>{''.join(f'<th>{SHORT[m]}</th>' for m in MODELS)}<th>PhD (Qwen)</th></tr></thead>
    <tbody>{behaviour_rows(D)}</tbody></table></div>
  <p style="margin-top:14px"><b>How to read this.</b> All three models give the camera's answer (A's) about 82–90% of the time, and B's answer
  about half the time, which is no better than guessing. On the questions that flip, they get it right only {pc(flipB['bagel'])}–{pc(max(flipB.values()))}
  of the time.</p>
  <p>The last two rows rule out two boring explanations. The models can read left and right in a photo (95–99% on the plain question),
  and when shown the actual photo taken from B's spot they answer correctly 99–100% of the time, which also confirms our answer key.</p>
  <h3>When they do depart from the camera's answer</h3>
  <p>On the 10–18% of questions where a model does not give A's answer, it is not taking B's view. Almost all of these departures happen
  when B appears on the same side of the photo as the camera's answer, and the model switches to the word it prefers anyway ("left" for Qwen
  and Gemma, "right" for BAGEL). Whether the question really flips makes no difference: only about half of the departures give B's correct
  answer. The instruction seems to push the answer away from B's side of the photo, a rough cue from the picture rather than a change of viewpoint.</p>
  <details><summary>The numbers</summary>
  <div class="tw"><table>
    <thead><tr><th>Departs from A's answer when…</th><th>A: left, B on the left of the photo</th><th>A: left, B on the right</th>
      <th>A: right, B on the left</th><th>A: right, B on the right</th><th>Departures that give B's answer</th></tr></thead>
    <tbody>{departure_rows(D)}</tbody></table></div>
  </details>
</section>

<section id="inside">
  <div class="kicker">3 · Looking inside</div>
  <h2>How the readers work</h2>
  <p><b>Where we look.</b> For every question we save the hidden state at every layer (plus the input embeddings, "layer 0") at one position:
  the last prompt token, right before the model writes its answer. For BAGEL, which handles the image and the prompt in separate passes, this
  is its first generation step. If the model works out B's answer, this is where it has to come together before being spoken. Section 6 checks
  other positions.</p>
  <p><b>The reader.</b> A reader is a linear probe on one layer's hidden state at that position: standardise each unit, reduce to 256 dimensions
  with PCA, then fit a logistic regression with balanced class weights. (PCA, principal component analysis, finds the directions along which
  the training hidden states vary the most and keeps only the top 256. It never sees the labels. With about 1,500 training questions and
  thousands of numbers per hidden state, this cuts noise and stops the regression from fitting the training rooms by chance.) One reader is trained per fact and per layer.</p>
  <p>Four choices make the scores trustworthy:</p>
  <ul class="tight">
    <li><b>Train and test on different rooms and furniture.</b> Readers learn from 25 training rooms (8 kinds of object) and are scored on 9
    other rooms with 5 other kinds of object. A reader that memorised rooms or furniture would fail.</li>
    <li><b>Linear on purpose.</b> A nonlinear classifier could compute the geometry itself from rich enough features. A linear reader can only
    find what the model has already laid out in its hidden state.</li>
    <li><b>Balanced accuracy.</b> Each score is the average of the per-class hit rates, so a reader that always says the same thing scores
    exactly chance (50% for two classes, 33% for three).</li>
    <li><b>Intervals by photo</b> {CHG}. The 95% intervals come from 1,000 resamples of test <i>photos</i>.</li>
  </ul>
  <div class="box">
    <h4>How B's answer is scored: within A's answer</h4>
    <p>We split the test questions by A's answer: one group where A's answer is "left" and one where it is "right". We score the B reader
    in each group separately and average the two. Inside a group, A's answer is the same for every question, and B's answer is "left" for half
    of them and "right" for the other half. So knowing A's answer does not help at all, and only real knowledge of B's view scores above 50%.</p>
    <p>This matters because A's answer is very easy to read from the hidden state (99%). Scoring this way guarantees that a reader that just
    picked up A's answer cannot get above chance.</p>
  </div>
</section>

<section id="readers">
  <div class="kicker">4 · The main result</div>
  <h2>B's answer is inside every model</h2>
  <p>Each panel shows, layer by layer, how well a reader recovers B's answer from the hidden state (blue, with a shaded 95% interval).
  The grey band is the <b>luck band</b>: the range scored by readers trained on shuffled answers. The dashed dark line is A's answer,
  for comparison. For Qwen the orange dashed line is the PhD's result.</p>
  {legend([("B's answer", "s1", False), ("PhD: B's answer (Qwen)", "s2", True), ("A's answer", "s-ink", True), ("same question, no picture", "s-mut", False), ("luck band", "band", False)])}
  <div class="figs" style="margin-top:10px">{reader_charts(D)}</div>
  <p style="margin-top:14px"><b>How to read this.</b> In early layers B's answer is at 50%: the model has not worked anything out yet.
  Then, partway through, it climbs well above the luck band: Qwen reaches <b>{pc(bpk['qwen3vl'][0])}</b> at layer {bpk['qwen3vl'][1]}
  (the PhD found 86% at layer 32), Gemma <b>{pc(bpk['gemma4'][0])}</b> at layer {bpk['gemma4'][1]}, and BAGEL
  <b>{pc(bpk['bagel'][0])}</b> at layer {bpk['bagel'][1]}. Our Qwen curve sits almost on top of the PhD's.</p>
  <p>So the models <b>work out B's answer internally</b>, even though, out loud, they give the camera's answer.</p>

  <h3>Six facts, read at the same spot</h3>
  <p>The same kind of reader was trained for six facts. Each number is the best layer's score on the test rooms.</p>
  <div class="tw"><table>
    <thead><tr><th>Fact</th><th>Chance</th>{''.join(f'<th>{SHORT[m]}</th>' for m in MODELS)}<th>PhD</th></tr></thead>
    <tbody>{six_rows(D)}</tbody></table></div>
  <p style="margin-top:14px">A's answer and the model's own answer are almost perfect, which shows the method works. "Flips?" is close to
  B's answer, as it should be: once you know A's answer, knowing B's answer and knowing whether it flips are the same thing. The depth-rank
  score is less reliable: it has three classes, and only 32 test questions have B as the nearest object.</p>
  <div class="box">
    <h4>Why A's answer and "flips?" together cannot explain the 86%</h4>
    <p>The hidden state holds A's answer (99%) and whether the question flips (88%). One might think the B reader just combines those two:
    B's answer is A's answer, swapped when it flips. But a linear reader cannot combine them that way. The rule is an exclusive-or: B's
    answer is "left" when A's is left and it does not flip, or when A's is right and it does flip. No single weighted sum of "A says left"
    and "it flips" gets this right. A reader leaning on the flip signal would be right among A-left questions and wrong among A-right
    questions, or the other way round, which averages to 50% within A's answer.</p>
    <p>So a linear reader scoring 80–86% within A's answer means the hidden state holds <b>B's left-or-right answer as a signal of its own</b>,
    not only as two separate facts that are never put together. The model has done the combining.</p>
  </div>
</section>

<section id="controls">
  <div class="kicker">5 · Could it be a fluke?</div>
  <h2>Six checks that the reader isn't fooling us</h2>
  <p>A high reader score could have boring causes. Each check below removes one possible cause. If B's answer were coming from that cause,
  the check would score as high as the real reader. All checks pass for all three models.</p>
  <div class="tw"><table>
    <thead><tr><th>Check</th><th>Pass mark</th>{''.join(f'<th>{SHORT[m]}</th>' for m in MODELS)}<th>PhD</th></tr></thead>
    <tbody>{controls_rows(D)}</tbody></table></div>
  <p style="margin-top:14px"><b>The added per-object check matters.</b> The PhD's picture check averages the image features over the whole
  photo, which could blur away an answer that is actually in the picture. So we gave a reader the picture features of B, X and Y
  separately. For Qwen it scores only {pc(vo['qwen3vl'])} (and {pc(ds_all) if ds_all else '–'} with Qwen's extra "DeepStack" picture features
  {ADD}), so the answer is not sitting in the picture: the language part of the model computes it. Gemma ({pc(vo['gemma4'])}) and
  BAGEL ({pc(vo['bagel'])}) get a head start from their picture features, but their language parts still add 12–15 points.</p>
  <details><summary>What "shuffled within each room" means {CHG}</summary>
    <p>We shuffle B's answers among the questions from the same training room, train a reader on those wrong answers, and score it against
    the true answers in the test rooms. Anything that only tells rooms apart survives the shuffle; real knowledge of B's view does not.
    The PhD's report does not say exactly how this control was scored, so this is our version of it.</p>
  </details>
</section>

<section id="sweep">
  <div class="kicker">6 · Where it appears</div>
  <h2>B's answer shows up only after the question is read</h2>
  <p>We also trained B-answer readers at other spots in the input, marked below. The models read left to right, and each spot can only use
  what comes before it.</p>
  {prompt_diagram()}
  <div class="tw"><table>
    <thead><tr><th>Where the reader looks</th>{''.join(f'<th>{SHORT[m]}</th>' for m in MODELS)}</tr></thead>
    <tbody>{sweep_rows(D)}</tbody></table></div>
  <p style="margin-top:14px"><b>How to read this.</b> At B's name the model has not yet heard about X and Y, so it cannot know the answer:
  the reader is at chance, as expected. By X's and Y's last mentions, the model knows where it is supposed to stand and what it is
  being asked, and B's answer appears (71–81%). It is strongest just before the answer. The picture patches come <i>before</i> the
  question, so they never see it; their 65–70% reflects general scene layout rather than this question's answer.</p>
</section>

<section id="lens">
  <div class="kicker">7 · Follow-up 1 {ADD}</div>
  <h2>Is B's answer on its way to the output?</h2>
  <p>The <b>logit lens</b> takes the hidden state at the answer position from an earlier layer and applies the model's own final norm and
  unembedding to it, as if the model stopped at that layer. We compare the score for "left" (combining the tokens "left" and "Left") with the
  score for "right" ("right" and "Right"). This shows which answer the model would give at each layer.</p>
  <p>Below, for the {nflip} flip questions (where B's and A's answers differ), we count how often the lens points to B's answer and how often
  to A's.</p>
  {legend([("lens points to A's answer", "s-ink", True), ("lens points to B's answer", "s1", False)])}
  <div class="figs" style="margin-top:10px">{lens_charts(D)}</div>
  <p style="margin-top:14px"><b>How to read this.</b> Early on, the lens is undecided (both lines near 50%). When the output starts to take
  shape (Qwen about layer 22, Gemma about 34, BAGEL about 23), it points to <b>A's answer</b>, and keeps doing so to the end. At no layer
  does it lean toward B's answer, even in the layers where the reader finds B's answer at 80–86%.</p>
  <h3>Why the lens never finds B's answer</h3>
  <p>The reader finds B's answer at these same layers, so why does the output never lean toward it? The plots below show the answer with
  real data. They use only the questions where the camera's answer is <b>"left"</b> (564 per model). Half of them are no-flip questions
  (B's answer is also left, blue) and half are flip questions (B's answer is right, orange). Each dot is one question, at the layer where
  the B reader peaks:</p>
  <ul class="tight">
    <li><b>Left to right:</b> which way the output leans if the model stopped at this layer (the logit lens score).</li>
    <li><b>Bottom to top:</b> the B reader's score: above the middle line, the reader says B's answer is left; below it, right.</li>
  </ul>
  {legend([("B's correct answer: left (no flip)", "s1", False), ("B's correct answer: right (flip)", "s2", False)])}
  <div class="figs3">{''.join(f'<figure class="panel">{lens_scatter(m)}</figure>' for m in MODELS)}</div>
  <p style="margin-top:14px"><b>How to read this.</b> Look at the colours. They separate <b>top from bottom</b>: blue dots sit mostly above the
  middle line and orange dots below it, so the reader can tell B's answer apart. They do <b>not</b> separate left from right: blue and orange
  dots sit at the same horizontal positions, mostly on the right, because the output leans toward the camera's answer ("left") whatever
  B's answer is. (In BAGEL the output has not fully decided yet at this layer, so the dots spread wider, but the two colours still overlap.)</p>
  <p>Measured over all questions, both A-left and A-right, the output's lean tells B's answer apart at chance level (45–53%), while the
  B reader manages 78–87%.</p>
  <p>So <b>the model holds B's answer, but in a part of the hidden state that the output does not use.</b> The output
  reads only the camera's answer.</p>
  <h3>Why the output cannot see B's answer</h3>
  <p><b>The model's choice comes down to one number.</b> At the end, the model scores the word "left" and the word "right" and says whichever
  scores higher. Call the difference, score(left) − score(right), the <b>lean</b>: positive means "left", negative means "right", and the
  bigger it is, the surer the model is.</p>
  <p><b>That number reads only one direction of the hidden state.</b> The lean depends on the hidden state only through how far it reaches
  along one fixed direction, which we call the <b>read-out direction</b>. In the picture, drop a line from the hidden state onto the read-out
  direction: where it lands, measured from the tie point, is the lean.</p>
  {lean_diagram()}
  <p><b>Moving sideways changes nothing.</b> If the hidden state moves perpendicular to the read-out direction (the upward arrow), it lands on
  the same spot, so the lean stays the same. The hidden state has thousands of directions; only this one decides left versus right at this layer.</p>
  <p><b>B's answer sits in a sideways direction.</b> Two measurements, at the layer where the B reader peaks:</p>
  <ul>
    <li><b>Geometry.</b> B's direction is nearly perpendicular to the read-out direction: the cosine between them is 0.13 or less in every
    model (1 would mean the same direction, 0 fully perpendicular).</li>
    <li><b>Data.</b> Among questions with the same camera answer, the lean tells B-left from B-right questions at chance (45–53%), while the
    B reader on the same hidden states gets 78–87%.</li>
  </ul>
  <p>So at this layer the output is blind to B's answer. The part of the hidden state it reads carries the camera's answer instead, and if the
  model stopped here it would answer from the camera's view, which is what the lens shows.</p>

  <details><summary>Technical details: the read-out direction, cosines and the two checks</summary>
    <h4>Where the read-out direction comes from</h4>
    <p>The model turns its last hidden state h into word scores in two steps. The <b>final norm</b> (RMSNorm) divides h by its size, rms(h),
    then multiplies each dimension by a learned weight g<sub>i</sub>. The <b>unembedding</b> gives each word w a score: the dot product of
    the normed state with that word's weight vector u<sub>w</sub>. So z<sub>w</sub> = (g ⊙ u<sub>w</sub>) · h / rms(h).</p>
    <p>The lens score combines the two spellings of each word with log-sum-exp:
    s(h) = LSE(z<sub>left</sub>, z<sub>Left</sub>) − LSE(z<sub>right</sub>, z<sub>Right</sub>). To first order the LSE of two scores is their
    average plus a constant, so with ū<sub>left</sub> = ½(u<sub>left</sub> + u<sub>Left</sub>) and likewise for right:</p>
    <p style="text-align:center"><b>s(h) ≈ (r · h) / rms(h) + const, &nbsp; r = g ⊙ (ū<sub>left</sub> − ū<sub>right</sub>)</b></p>
    <p>r is the read-out direction. The norm's weights g are part of it and tilt it away from the plain difference of word weights; the
    division by rms(h), one number for the whole vector, only scales the lean. Gemma also caps its scores with c · tanh(z / c), which only
    rises and is nearly straight for ordinary scores, so it does not change which word wins. The logit lens applies this same norm and
    unembedding to earlier layers.</p>

    <h4>What a cosine says about a push</h4>
    <p>A change δ splits into a part along r and a part perpendicular to it; only the first moves the lean, by about |r| (δ · r̂) / rms(h).
    For a push of length a along a unit direction v, the part that counts is a · cos(v, r). In the cartoon, three pushes have the same length,
    and the dashed lines show how far each one moves the reading.</p>
    {perp_diagram()}
    <div class="tw" style="margin-top:10px"><table>
      <thead><tr><th>Cosine with the read-out, at the reader's best layer</th><th>B's direction</th><th>"Say left" direction</th><th>Reader's normal</th></tr></thead>
      <tbody>{cos_rows}</tbody></table></div>
    <p style="margin-top:10px">B's direction is defined in section 8. "Say left" is the average hidden state when the model says "left"
    minus when it says "right"; the reader's normal is the B reader's weight vector mapped back to the hidden space.</p>

    <h4>Why two checks</h4>
    <p>The cosine is about <i>pushes</i>: what happens if you move the hidden state. The lean-versus-reader comparison is about the
    <i>data</i>: whether B-left and B-right questions actually sit at different read-out positions. These can disagree. Across questions, the
    reader's score v · h and the read-out r · h have correlation vᵀΣr / √(vᵀΣv · rᵀΣr), where Σ is the covariance of the hidden states, which
    can be non-zero even when v and r are perpendicular. Both checks agree here.</p>

    <h4>"Not used" in the causal sense</h4>
    <p>Perpendicular at one layer does not mean unused. "Used" means the final score, after the remaining layers F, changes when h moves
    along v: the derivative of the final read-out of F(h) along v is not zero. Steering measures this with finite pushes.</p>
  </details>
  <details><summary>A check on the lens</summary>
    <p>At the last layer, the lens must reproduce the model's real answer scores. It does: the mean difference from the saved scores is
    0.08 (Qwen), 0.06 (Gemma) and 0.04 (BAGEL) on a log scale, against left-versus-right gaps of several units.</p>
  </details>
</section>

<section id="steer">
  <div class="kicker">8 · Follow-up 2 {ADD}</div>
  <h2>Does the model use B's answer?</h2>
  <p>Readers show that B's answer is <i>present</i>. They do not show that the model <i>uses</i> it. Two questions, kept apart:</p>
  <ul class="tight">
    <li><b>Natural use:</b> at its natural size, does B's information move the answer? (Measured from the slope of the model's output.)</li>
    <li><b>Capacity:</b> if B's information is made much stronger, can the later layers act on it, beyond what any push of the same size does?
    (Measured by pushing the hidden state, as in the cartoon.)</li>
  </ul>
  {push_diagram()}
  <p style="margin-top:10px">Three stories are possible. <b>Used but outweighed:</b> B already pushes on the answer, just not enough (natural use
  would show it). <b>Suppressed:</b> the later layers can act on B, but the camera's answer wins in normal runs (capacity would show it).
  <b>Inert:</b> nothing downstream reads B (both null). All decisions below were fixed before the results; the full plan and change log are in
  <code>PLAN_V4_STEER.md</code>.</p>

  <h3>Step 1: B's direction, and checking it is a real signal</h3>
  <p>B's left/right answer here is the <b>correct</b> answer from B's spot, from the scene's geometry. Sort the training questions into four
  groups by A's correct answer and B's correct answer, average the hidden states at the token before the answer, and subtract:</p>
  {means_diagram()}
  <p style="margin-top:10px">The picture is a cartoon in two dimensions; the real states have thousands. Each dot is one question.</p>
  <div class="eq">B's direction = ½ × [ (mean of A-left, B-left − mean of A-left, B-right) + (mean of A-right, B-left − mean of A-right, B-right) ]</div>
  <p style="margin-top:12px"><b>Why split by A's answer?</b> Inside each subtraction, A's answer is the same on both sides, so anything
  about A's answer cancels. Inside the A-left group, "B left" questions are no-flip questions and "B right" questions are flip questions;
  inside the A-right group it is the other way round. So the "is it a flip?" part enters the two arrows with opposite signs and cancels
  when we average them. What is left is the part that tracks B's answer itself. Its length is <b>B's gap</b>, the natural size of the
  B difference, used as the unit for every push. Two more directions are built the same way: <b>A's direction</b> (the roles swapped) and
  <b>depth rank</b> (farthest minus middle, inside each of the four groups), which is structured but has nothing to do with left or right.</p>
  <p><b>Is it real?</b> We rebuilt B's direction separately from the test rooms (other rooms, other furniture) and from two halves of the
  training rooms, and compared. Each half has only half the data, so from the split-half cosine ρ we estimate how well the full direction
  matches the true signal, √(2ρ / (1 + ρ)); the bar was 0.9.</p>
  <div class="tw"><table>
    <thead><tr><th>At the headline layer</th><th>Training vs test rooms</th><th>Split-half ρ</th><th>Full direction (bar 0.9)</th><th>Shuffled labels</th></tr></thead>
    <tbody>{s2_validation_rows(D)}</tbody></table></div>
  <p style="margin-top:10px">The <b>headline layer</b> is chosen on training rooms only: the earliest layer whose cross-validated B reader is within
  0.01 of its best (Qwen L{hl2['qwen3vl']}, Gemma L{hl2['gemma4']}, BAGEL L{hl2['bagel']}). B's direction is real from the middle layers on, where the readers
  climb. It is also small: along it, B-left and B-right questions overlap heavily (55–60% separable, against the reader's 80–86%).</p>

  <h3>Step 2: the directions that move the answer</h3>
  <p>For every question, one forward and one backward pass gives the <b>gradient</b> of the answer with respect to the hidden state at every
  layer: for each direction, how much a tiny push changes the model's log-odds of "left" against "right". Pushing real hidden states along
  a question's own gradient moves the answer by what it predicts, to within about 10%.</p>
  <p>Most of each gradient (about 75% in Qwen and BAGEL, nearly all in Gemma) points in directions that no question's hidden state ever moves
  in. B's direction and every control only point where states do vary, so we keep only the <b>realistic</b> part: each gradient is projected
  onto the top principal directions of the training states (99% of their variance, about 350–410 directions per layer). From those:</p>
  <ul class="tight">
    <li><b>g</b>: the average realistic gradient, the single direction that most moves the answer. It is the <b>answer channel</b>.</li>
    <li><b>G</b>: a small set (2–6 directions at the headline layers) covering how that answer-moving direction varies from question to question.</li>
  </ul>

  <h3>Step 3: natural use</h3>
  <p>B's <b>overlap</b> with the answer channel (the cosine with g) is compared with 1,000 random realistic directions (random mixtures of
  the training states). Fixed in advance: B counts as used at natural size if its overlap beats 95% of them, at the headline layer. The camera's
  direction (A) is measured the same way as a reference: information the model does use.</p>
  <div class="tw"><table>
    <thead><tr><th>At the headline layer</th><th>B's overlap</th><th>Beats random</th><th>Camera's overlap</th><th>Beats random</th></tr></thead>
    <tbody>{s2_part1_rows(D)}</tbody></table></div>
  <div class="figs3" style="margin-top:12px">{s2_part1_charts(D)}</div>
  {legend([("camera's direction (A)", "s3", False), ("B's direction", "s1", False), ("95% of random realistic directions", "band", False)])}
  <p style="margin-top:10px"><b>B stays inside the random band; the camera's direction clears it in all three models.</b> The same holds at the last
  mentions of X and Y in the prompt: no layer where B's direction is reliable shows a meaningful effect. A real natural-size push along B moves
  the answer by what the slope predicts (about 0.1–0.2 logits, against typical leans of 6–8 for Qwen and Gemma), and not consistently toward
  B's answer. So <b>"used but outweighed" is ruled out</b>: in normal runs, B's information has no special route into the answer.</p>

  <h3>Step 4: capacity, pushing hard</h3>
  <p>To test whether amplified B can be acted on, every direction first has G removed ("⊥ G"), so that, to first order, none of them can move
  the answer. Removing G keeps 96–97% of B's direction: it only strips the small part that leans on the answer, which step 3 already tested.
  Any difference between the directions when pushed hard must then come from what they contain.</p>
  <ul class="tight">
    <li><b>B ⊥ G</b>: the test.</li>
    <li><b>10 random ⊥ G</b>: random realistic directions, the same treatment. Their spread shows what a push of that size does by itself.</li>
    <li><b>Depth rank ⊥ G</b>: structured but irrelevant to left and right. A real B effect must beat it too.</li>
    <li><b>A ⊥ G</b> (reference) and <b>g</b> (positive control: shows pushes at this layer can move the answer).</li>
  </ul>
  <div class="tw"><table>
    <thead><tr><th>After removing G, share of g's effect</th><th>Gradient in realistic space</th><th>G size</th><th>B (before)</th><th>B ⊥ G</th><th>Depth ⊥ G</th><th>Random ⊥ G</th><th>B kept</th></tr></thead>
    <tbody>{s2_gate_rows(D)}</tbody></table></div>
  <p style="margin-top:10px">Checked on each <i>test</i> question's own gradient: after removing G, every direction has at most about 4% of g's
  effect, so they start on equal footing.</p>
  <p><b>Pushes</b> of 1, 4, 8, 16 and 32 × B's gap, all the same length, toward each test question's B answer, at the token before the answer, one
  layer at a time. B's gap is 1–4% of the hidden state's length, so 32× is far outside natural values; that is why every direction is
  compared with controls pushed exactly as hard. <b>Measure:</b> the shift in the model's log-odds toward B's answer, over all 1,128 test
  questions, with photo-bootstrap intervals. Before trusting any result, our shortcut (reusing the model's memory of the prompt and
  re-running only the last position) matched a normal full run on {checks} of test questions.</p>
  <p><b>Fixed in advance:</b> B's content is "acted on" if, at the headline layer, at both 16× and 32×, B ⊥ G beats all 10 random seeds and depth
  rank, its interval clears the random mean, and under 10% of outputs break.</p>
  <div class="tw"><table>
    <thead><tr><th>At the headline layer</th><th>B ⊥ G, 16×</th><th>Random, 16×</th><th>Depth, 16×</th><th>B ⊥ G, 32×</th><th>Random, 32×</th><th>Depth, 32×</th><th>g, 32×</th><th>Acted on?</th></tr></thead>
    <tbody>{s2_part2_rows(D)}</tbody></table></div>
  <p style="margin-top:10px">Shift in the log-odds toward B's answer, in logits (95% interval). Pick a model and drag the slider for other layers.
  Layers where B's direction is not yet reliable are marked.</p>
  <div class="panel" style="padding:14px">
    <div class="controls" role="group" aria-label="Model">
      <button type="button" data-model="qwen3vl" aria-pressed="true">{NAME['qwen3vl']}</button>
      <button type="button" data-model="gemma4" aria-pressed="false">{NAME['gemma4']}</button>
      <button type="button" data-model="bagel" aria-pressed="false">{NAME['bagel']}</button>
    </div>
    <div class="controls"><label for="steer-layer">Layer pushed <input type="range" id="steer-layer" min="0" max="10" step="1" value="0">
      <span id="steer-layer-lab"></span></label></div>
    <svg id="steer-svg" viewBox="0 0 640 320" class="chart" role="img" aria-label="Steering results"></svg>
    <p id="steer-note"></p>
  </div>

  <h3>What the results say</h3>
  <ul class="tight">
    <li><b>Gemma and BAGEL: nothing.</b> B ⊥ G stays at about zero at every strength and every reliable layer, like the random directions. (Gemma
    at 32× edges above every seed, by 0.6 logits against typical leans of about 6, and not at 16×.)</li>
    <li><b>Qwen: a hint, inside the noise.</b> At layers 24–26, B ⊥ G moves the answer toward B's consistently (about +2.3 to +2.9 logits at
    16×, on flip and no-flip questions alike). But at these sizes random realistic directions swing Qwen's answer by ±2–5 logits either way, and
    B ⊥ G ranks 8th–9th of 10 seeds. It does not pass.</li>
    <li><b>B's full direction backfires in Gemma and BAGEL</b> (the answer moves away from B's), in exact proportion to the push, and the effect
    disappears once G is removed. It is the small, random-sized part of B's direction that happens to point toward the other word, not B's
    information being used in reverse.</li>
    <li><b>Overall: "inert" is the story the data support.</b> In normal runs B's answer has no special route to the output, and amplified it
    does nothing that a random realistic push of the same size does not. As agreed in advance, this is reported as "no route found at the token
    before the answer", not as proof that the information is unused everywhere.</li>
  </ul>
  <div class="box">
    <h4>{CHG} Why this replaces our first steering run</h4>
    <p>The first run removed only the model's average "say left" direction from B's and compared it with random directions drawn uniformly in all
    dimensions. It found Qwen giving B's answer on 86% of flip questions at 16× (BAGEL 76% at 32×) against at most 17% for random pushes. But
    "say left" is a correlational direction that covers only part of what moves the answer, and uniform random directions point where hidden states
    never go and are a weak baseline. With realistic random directions of the same length, comparable pushes move Qwen's answer just as much, so the
    first run's evidence that the later layers act on B does not hold up.</p>
  </div>
</section>

<section id="conclusion">
  <div class="kicker">9 · Putting it together</div>
  <h2>What we can and cannot conclude</h2>
  <ol class="tight">
    <li><b>The PhD's v4 result replicates.</b> Qwen3-VL keeps B's answer inside (86%, the same as the PhD's 86%) while saying the camera's
    answer. Every check passes.</li>
    <li><b>It holds in two more models</b> (Gemma 80%, BAGEL 82%).</li>
    <li><b>The language part of the model computes it.</b> The picture features alone give much less, especially in Qwen (54% vs 86%).</li>
    <li><b>The output never uses it.</b> At its natural size, B's answer has no special route to the output (the camera's answer has a strong one),
    and amplified it does nothing a random push of the same size does not. The models work out what B would see, but their output is wired to the
    camera's view.</li>
  </ol>
</section>

<section id="details">
  <div class="kicker">10 · For the record</div>
  <h2>Differences from the PhD's method</h2>
  <ul class="diff">
    <li>{ADD} Two more models (Gemma 4 12B, BAGEL-7B-MoT); per-object picture check and Qwen DeepStack check; the three-object reader;
    the angle-from-B breakdown; the logit lens; steering (natural use from gradients in the realistic space; capacity with B ⊥ G against realistic
    random and depth-rank controls).</li>
    <li>{CHG} Picture-quality rules: no cut-off or mostly hidden objects in A's photo; X and Y may not touch in B's view.</li>
    <li>{CHG} The depth mix: every usable nearest and farthest question kept (test 32 / 324 / 772; PhD 24 / 352 / 756).</li>
    <li>{CHG} Intervals resample test photos rather than single questions; "shuffled within each room" is scored against the true test answers.</li>
  </ul>
  <p>Code: <code>repro/v4/</code> in the project repository (data: render.py, balance.py; models: extract.py, extract_bagel.py; readers:
  analyze.py; follow-ups: lens.py; steer2_dirs.py, grads.py, grads_bagel.py, steer2.py, steer2_bagel.py, steer2_common.py; plan and change log:
  PLAN_V4_STEER.md). This page: <code>repro/report/build_v4_results_report.py</code>.</p>
</section>

</div>
<script type="application/json" id="steer-data">{json.dumps(S)}</script>
<script>{JS}</script>
"""
    out = LOCAL / "v4_results_report.html"
    out.write_text(page)
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
