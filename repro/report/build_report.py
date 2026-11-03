"""Replication report (Sections 1-2 of "Perspective Pilot Reproduction"), as one HTML page.

Reads data pulled from Turbo into local_data/ (see the rsync commands in the README/chat):
  local_data/data/{test,train}/examples_labeled.jsonl, local_data/data/scene_index/*.json,
  local_data/data/test/<scene>/<id>_{A,B}_rgb[_marked].png, local_data/results/<model>/test/*.jsonl

    python repro/report/build_report.py --out local_data/report.html
"""
import argparse
import base64
import html
import json
import math
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
MODELS = [("qwen3vl", "Qwen3-VL-8B"), ("bagel", "BAGEL-7B-MoT"), ("gemma4", "Gemma 4 12B")]
REPORT_MODELS = {"qwen3vl", "bagel"}
SETTINGS = [
    ("s1", "1", "Original", "A's photo + sentence about B", "B"),
    ("s1_identity", "1·id", "Identity variant", "A's photo + “B stands where you stand”", "A"),
    ("s2", "2", "Ground-truth target", "B's photo, direct question", "B"),
    ("s4", "4", "Control (own view)", "A's photo, direct question", "A"),
]
# Report, large cross_line set (92 clean): (correct, answered from A)
REPORT = {
    ("s1", "qwen3vl"): (18, 74), ("s1", "bagel"): (31, 61),
    ("s1_identity", "qwen3vl"): (84, 84), ("s1_identity", "bagel"): (88, 88),
    ("s2", "qwen3vl"): (88, 4), ("s2", "bagel"): (89, 3),
    ("s4", "qwen3vl"): (86, 86), ("s4", "bagel"): (87, 87),
}
REPORT_N = 92
# Report 2x2 tables for setting 1: correct answers by B's answer, and class sizes (left 56, right 36)
REPORT_S1_CLASS = {"qwen3vl": (17, 1), "bagel": (0, 31)}
REPORT_CLASS_N = (56, 36)

e = html.escape


def pct(a, n):
    return f"{100 * a / n:.0f}%" if n else "–"


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def thumb(path, width=420):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def furniture(data, scene):
    p = data / "scene_index" / f"{scene}.json"
    if not p.exists():
        return []
    return [(o["aabb_min"][0], o["aabb_max"][0], o["aabb_min"][2], o["aabb_max"][2])
            for o in json.loads(p.read_text())["objects"]
            if o["size"][1] >= 0.25 and max(o["size"][0], o["size"][2]) < 5]


def map_svg(rec, objs, hfov=90.0, size=260):
    pts = [rec["A"]["pos"], rec["B"]["pos"], rec["X_center"], rec["Y_center"]]
    xs = [p[0] for p in pts] + [v for o in objs for v in o[:2]]
    zs = [p[2] for p in pts] + [v for o in objs for v in o[2:]]
    x0, x1, z0, z1 = min(xs) - 0.6, max(xs) + 0.6, min(zs) - 0.6, max(zs) + 0.6
    s = size / max(x1 - x0, z1 - z0)
    xy = lambda x, z: ((x - x0) * s, (z - z0) * s)
    out = [f'<svg viewBox="0 0 {(x1 - x0) * s:.0f} {(z1 - z0) * s:.0f}" class="map" role="img" '
           f'aria-label="Floor plan from above with A, B and the two objects">']
    for ox0, ox1, oz0, oz1 in objs:
        out.append(f'<rect x="{(ox0 - x0) * s:.1f}" y="{(oz0 - z0) * s:.1f}" width="{(ox1 - ox0) * s:.1f}" '
                   f'height="{(oz1 - oz0) * s:.1f}" class="furn"/>')
    (ax, ay), (bx, by) = xy(rec["X_center"][0], rec["X_center"][2]), xy(rec["Y_center"][0], rec["Y_center"][2])
    out.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" class="xyline"/>')
    for view, cls in (("A", "obsA"), ("B", "obsB")):
        p, yaw = rec[view]["pos"], rec[view]["yaw"]
        cx, cy = xy(p[0], p[2])
        wedge = [f"{cx:.1f},{cy:.1f}"]
        for d in (-hfov / 2, hfov / 2):
            r = math.radians(yaw + d)
            wx, wy = xy(p[0] - 1.3 * math.sin(r), p[2] - 1.3 * math.cos(r))
            wedge.append(f"{wx:.1f},{wy:.1f}")
        out.append(f'<polygon points="{" ".join(wedge)}" class="{cls} wedge"/>'
                   f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" class="{cls}"/>'
                   f'<text x="{cx + 8:.1f}" y="{cy - 6:.1f}" class="lbl">{view}</text>')
    for c, cls in ((rec["X_center"], "objX"), (rec["Y_center"], "objY")):
        cx, cy = xy(c[0], c[2])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="{cls}"/>')
    out.append("</svg>")
    return "".join(out)


def stats(rows):
    n = len(rows)
    ok = sum(r["answer"] == r["truth"] for r in rows)
    from_a = sum(r["answer"] == r["relA"] for r in rows)
    bad = sum(r["answer"] == "invalid" for r in rows)
    said_left = sum(r["answer"] == "left" for r in rows)
    cls = {}
    for w in ("left", "right"):
        g = [r for r in rows if r["truth"] == w]
        cls[w] = (sum(r["answer"] == w for r in g), len(g))
    return dict(n=n, ok=ok, from_a=from_a, bad=bad, said_left=said_left, cls=cls)


def build(data, results):
    test = load_jsonl(data / "test" / "examples_labeled.jsonl")
    train = load_jsonl(data / "train" / "examples_labeled.jsonl")
    clean = [r for r in test if r["clean"]]
    res = {}
    for m, _ in MODELS:
        for s, *_ in SETTINGS:
            for suffix in ("", "_marked"):
                p = results / m / "test" / f"{s}{suffix}.jsonl"
                if p.exists():
                    res[(m, s + suffix)] = {r["id"]: r for r in load_jsonl(p)}
    st = {k: stats(list(v.values())) for k, v in res.items()}
    n = len(clean)

    # ---------- Section 1 numbers ----------
    reasons = Counter()
    for r in test:
        for t in r["reject_reasons"]:
            key = ("overlap" if "overlap" in t else "too close in px" if "px apart" in t
                   else "observer < 1.5 m" if "stands" in t else "no flip")
            reasons[key] += 1
    disagree = {v: sum(r[f"rel{v}"] != r[f"rel{v}_centroid"] for r in test) for v in "AB"}
    b_left = sum(r["relB"] == "left" for r in clean)
    rooms = len({r["scene"] for r in clean})
    next_to = sum("next to" in r["sentence_B"] for r in clean)
    facing = sum("is facing" in r["sentence_B"] for r in clean)

    # ---------- headline ----------
    q1, b1 = st[("qwen3vl", "s1")], st[("bagel", "s1")]
    ctrl = [st[(m, s)]["ok"] / st[(m, s)]["n"] for m in ("qwen3vl", "bagel")
            for s in ("s1_identity", "s2", "s4")]

    # ---------- results table ----------
    def model_cells(key, report_key=None):
        s_ = st.get(key)
        if not s_:
            return '<td class="num muted" colspan="2">not run</td>'
        flag = f' <span class="flag" title="{s_["bad"]} answers could not be read as left/right">{s_["bad"]} unread</span>' if s_["bad"] else ""
        return (f'<td class="num"><b>{pct(s_["ok"], s_["n"])}</b> <span class="frac">{s_["ok"]}/{s_["n"]}</span>{flag}</td>'
                f'<td class="num">{pct(s_["from_a"], s_["n"])}</td>')

    rows_html = []
    for marked in (False, True):
        if marked:
            rows_html.append('<tr class="group"><th colspan="7">Outlined photos (added in this replication): '
                             'X outlined orange, Y blue, and the question names the colours</th></tr>')
        for s, code, name, gets, truth in SETTINGS:
            key_s = s + ("_marked" if marked else "")
            for i, (m, label) in enumerate(MODELS):
                rep = REPORT.get((s, m)) if not marked else None
                rep_cells = (f'<td class="num rep">{pct(rep[0], REPORT_N)}</td><td class="num rep">{pct(rep[1], REPORT_N)}</td>'
                             if rep else '<td class="num rep muted">–</td><td class="num rep muted">–</td>')
                head = (f'<td rowspan="{len(MODELS)}" class="setting"><span class="code">{code}</span> {e(name)}'
                        f'<div class="muted small">{e(gets)} · scored against {truth}</div></td>') if i == 0 else ""
                extra = '' if m in REPORT_MODELS else ' class="ext"'
                rows_html.append(f'<tr{extra}>{head}<td class="model">{e(label)}</td>{model_cells((m, key_s))}{rep_cells}</tr>')

    # ---------- bias table (setting 1) ----------
    bias_rows = []
    for m, label in MODELS:
        for suffix, tag in (("", ""), ("_marked", " · outlined")):
            s_ = st.get((m, "s1" + suffix))
            if not s_:
                continue
            (hl, nl), (hr, nr) = s_["cls"]["left"], s_["cls"]["right"]
            bal = (hl / nl + hr / nr) / 2
            bias_rows.append(f'<tr><td class="model">{e(label)}{tag}</td>'
                             f'<td class="num">{pct(s_["said_left"], s_["n"])}</td>'
                             f'<td class="num">{hl}/{nl} <span class="muted">({pct(hl, nl)})</span></td>'
                             f'<td class="num">{hr}/{nr} <span class="muted">({pct(hr, nr)})</span></td>'
                             f'<td class="num"><b>{bal:.0%}</b></td></tr>')
    for m, label in MODELS:
        if m in REPORT_S1_CLASS:
            hl, hr = REPORT_S1_CLASS[m]
            nl, nr = REPORT_CLASS_N
            bal = (hl / nl + hr / nr) / 2
            bias_rows.append(f'<tr class="rep-row"><td class="model">{e(label)} · report</td><td class="num muted">–</td>'
                             f'<td class="num">{hl}/{nl} <span class="muted">({pct(hl, nl)})</span></td>'
                             f'<td class="num">{hr}/{nr} <span class="muted">({pct(hr, nr)})</span></td>'
                             f'<td class="num"><b>{bal:.0%}</b></td></tr>')

    # ---------- example cards ----------
    def outcome(m, r, s="s1"):
        a = res.get((m, s), {}).get(r["id"])
        return None if a is None else a["answer"]

    picked, seen = [], set()
    # prefer variety: Qwen right, Qwen wrong & BAGEL right, all wrong; distinct rooms
    buckets = [lambda r: outcome("qwen3vl", r) == r["relB"],
               lambda r: outcome("qwen3vl", r) != r["relB"] and outcome("bagel", r) == r["relB"],
               lambda r: all(outcome(m, r) == r["relA"] for m, _ in MODELS)]
    for want in (1, 2, 3):
        for f in buckets:
            for r in clean:
                if r["scene"] not in seen and f(r) and len([p for p in picked if f(p)]) < want:
                    picked.append(r)
                    seen.add(r["scene"])
                    break
    picked = picked[:6]

    cards = []
    for r in picked:
        d = data / "test" / r["scene"]
        imgs = {v: thumb(d / f"{r['id']}_{v}_rgb_marked.png") for v in "AB"}
        ans_rows = []
        for m, label in MODELS:
            cells = []
            for s in ("s1", "s1_marked", "s2", "s4"):
                a = outcome(m, r, s)
                truth = r["relB"] if s in ("s1", "s1_marked", "s2") else r["relA"]
                cls = "ok" if a == truth else "bad"
                cells.append(f'<td class="ans {cls}">{e(a or "–")}</td>')
            ans_rows.append(f'<tr><td class="model">{e(label)}</td>{"".join(cells)}</tr>')
        cards.append(f"""
<article class="ex">
  <header><h3>{e(r['id'])}</h3><p class="muted small">{e(r['scene'])}</p></header>
  <p class="q">From B's point of view, is the <span class="x">{e(r['X'])}</span> to the left or to the right of the <span class="y">{e(r['Y'])}</span>?
     <span class="truths">From A's spot: <b>{r['relA']}</b> · from B's spot: <b>{r['relB']}</b></span></p>
  <p class="sentence">{e(r['sentence_B'])}</p>
  <div class="views">
    <figure><img src="{imgs['A']}" alt="A's photo with the two objects outlined"><figcaption>A's photo</figcaption></figure>
    <figure><img src="{imgs['B']}" alt="B's photo with the two objects outlined"><figcaption>B's photo</figcaption></figure>
    <figure>{map_svg(r, furniture(data, r['scene']))}<figcaption>From above: <span class="a">A</span>, <span class="b">B</span>, view wedges, dashed X–Y line</figcaption></figure>
  </div>
  <div class="scroll"><table class="answers">
    <thead><tr><th>Model</th><th>1 Original</th><th>1 outlined</th><th>2 B's photo</th><th>4 A's photo</th></tr></thead>
    <tbody>{''.join(ans_rows)}</tbody>
  </table></div>
</article>""")

    gemma_bad = st.get(("gemma4", "s1_identity_marked"), {}).get("bad", 0)
    return PAGE.format(
        n=n, rooms=rooms, rendered=len(test), clean_pct=pct(n, len(test)),
        train_clean=sum(r["clean"] for r in train), train_rooms=len({r["scene"] for r in train if r["clean"]}),
        q1_ok=pct(q1["ok"], q1["n"]), q1_a=pct(q1["from_a"], q1["n"]),
        b1_ok=pct(b1["ok"], b1["n"]), b1_left=pct(b1["said_left"], b1["n"]),
        ctrl_lo=f"{min(ctrl):.0%}", ctrl_hi=f"{max(ctrl):.0%}",
        rej_close=reasons["observer < 1.5 m"], rej_overlap=reasons["overlap"],
        rej_px=reasons["too close in px"], rej_flip=reasons["no flip"],
        dis_a=disagree["A"], dis_b=disagree["B"], b_left=b_left, b_right=n - b_left,
        next_to=next_to, facing=facing,
        results_rows="".join(rows_html), bias_rows="".join(bias_rows), cards="".join(cards),
        gemma_note=(f"Gemma's outlined identity row has {gemma_bad} answers that could not be read: "
                    "it began to explain or to think and hit the 5-token answer limit. "
                    "Rerunning that setting with a longer answer limit should fix this; it is not yet included here." if gemma_bad else ""),
    )


PAGE = """<title>Perspective Replication</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: one reading column (~68ch) with wide breakouts for tables and example cards */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --ink: #18202b; --muted: #5e6a79; --line: #dfe3e9;
  --accent: #2f5f8a; --x: #d9661f; --y: #2b7fd4; --a: #2f5fb3; --b: #c23b3b;
  --ok: #1e7a4a; --ok-soft: #e3f3ea; --bad: #b3362d; --bad-soft: #fbe7e5; --rep: #eef1f6; --furn: #d3d9e1;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12161c; --panel: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --line: #2c3542;
  --accent: #8fb6dd; --x: #f08a4b; --y: #62a6ec; --a: #7aa2e8; --b: #ec7a72;
  --ok: #6fd39c; --ok-soft: #173326; --bad: #f08a80; --bad-soft: #3a1d1b; --rep: #1f2733; --furn: #333d4b;
  color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12161c; --panel: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --line: #2c3542;
  --accent: #8fb6dd; --x: #f08a4b; --y: #62a6ec; --a: #7aa2e8; --b: #ec7a72;
  --ok: #6fd39c; --ok-soft: #173326; --bad: #f08a80; --bad-soft: #3a1d1b; --rep: #1f2733; --furn: #333d4b;
  color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 17px/1.6 var(--body); }}
main {{ max-width: 1080px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }}
.prose {{ max-width: 68ch; }}
h1, h2, h3 {{ font-family: var(--display); text-wrap: balance; line-height: 1.15; margin: 0; }}
h1 {{ font-size: clamp(2rem, 5vw, 3.1rem); font-weight: 700; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.6rem; font-weight: 700; margin-top: 24px; }}
h3 {{ font-size: 0.95rem; font-family: var(--mono); font-weight: 600; }}
p {{ margin: 0; }}
.stack {{ display: grid; gap: 14px; }}
.eyebrow {{ font: 600 0.78rem/1.4 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent); }}
.muted {{ color: var(--muted); }} .small {{ font-size: 0.85rem; }}
.lede {{ font-size: 1.2rem; }}
.headline {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.headline div {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
.headline b {{ display: block; font: 700 1.9rem/1.1 var(--display); font-variant-numeric: tabular-nums; }}
.scroll {{ overflow-x: auto; min-width: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.93rem; }}
th, td {{ padding: 7px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }}
thead th {{ font: 600 0.75rem/1.3 var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
td.num {{ font-family: var(--mono); font-size: 0.86rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.frac {{ color: var(--muted); font-size: 0.78rem; }}
td.rep, th.rep {{ background: var(--rep); }}
tr.group th {{ font: 600 0.85rem/1.4 var(--body); text-transform: none; letter-spacing: 0; color: var(--ink); padding-top: 18px; }}
tr.ext td.model {{ color: var(--muted); }}
tr.rep-row td {{ background: var(--rep); }}
.setting {{ min-width: 190px; }} .code {{ font-family: var(--mono); font-weight: 600; color: var(--accent); }}
.flag {{ font-size: 0.72rem; color: var(--bad); }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; }}
pre.prompt {{ white-space: pre-wrap; font: 0.84rem/1.5 var(--mono); margin: 0; background: var(--rep); padding: 10px 12px; border-radius: 6px; }}
.x {{ color: var(--x); font-weight: 600; }} .y {{ color: var(--y); font-weight: 600; }}
.a {{ color: var(--a); font-weight: 700; }} .b {{ color: var(--b); font-weight: 700; }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
.ex {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; display: grid; gap: 12px; }}
.ex header {{ display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
.q .truths {{ display: block; color: var(--muted); font-size: 0.9rem; }}
.sentence {{ font-style: italic; color: var(--muted); }}
.views {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
figure {{ margin: 0; min-width: 0; }}
figure img, .map {{ width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid var(--line); }}
figcaption {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
.map {{ background: var(--panel); }} .furn {{ fill: var(--furn); }}
.xyline {{ stroke: var(--muted); stroke-dasharray: 4 3; stroke-width: 1.5; fill: none; }}
.obsA {{ fill: var(--a); }} .obsB {{ fill: var(--b); }} .wedge {{ opacity: 0.22; }}
.objX {{ fill: var(--x); }} .objY {{ fill: var(--y); }} .lbl {{ fill: var(--ink); font: 700 12px var(--body); }}
td.ans {{ font-family: var(--mono); font-size: 0.85rem; }}
td.ans.ok {{ background: var(--ok-soft); color: var(--ok); }} td.ans.bad {{ background: var(--bad-soft); color: var(--bad); }}
</style>
<main>
<header class="stack prose">
  <p class="eyebrow">Spatial Perspectives · Replication · ReplicaCAD in Habitat-Sim · {n} clean cross_line examples · {rooms} rooms</p>
  <h1>The models still answer from A's spot.</h1>
  <p class="lede">We rebuilt the cross_line test from the pilot reproduction on new renders and asked the same questions.
  A takes a photo; B stands on the far side of two objects and looks back, so left and right are swapped for B.
  Given A's photo and a sentence saying where B stands, the models are asked what B sees.</p>
</header>

<section class="headline" aria-label="Headline numbers">
  <div><span class="muted small">Qwen3-VL, original task</span><b>{q1_ok}</b><span class="small">correct · {q1_a} gave A's answer (report: 20% · 80%)</span></div>
  <div><span class="muted small">BAGEL, original task</span><b>{b1_ok}</b><span class="small">correct · says “left” only {b1_left} of the time (report: 34%)</span></div>
  <div><span class="muted small">Controls, both report models</span><b>{ctrl_lo}–{ctrl_hi}</b><span class="small">identity variant, B's photo, A's photo (report: 91–97%)</span></div>
</section>

<section class="stack prose">
  <h2>What is asked</h2>
  <p>Every clean example is asked four ways, with the report's wording. X and Y are the two objects.</p>
</section>
<div class="scroll"><table>
  <thead><tr><th>Setting</th><th>The model gets</th><th>Prompt</th><th>Correct when the answer matches</th></tr></thead>
  <tbody>
  <tr><td><span class="code">1</span> Original</td><td>A's photo</td><td><pre class="prompt">This image shows what you (observer A) currently see. Another observer, B, is in the same room. [sentence about B] From B's point of view, is the X to the left or to the right of the Y? Answer with exactly one word: left or right.</pre></td><td>B's spot</td></tr>
  <tr><td><span class="code">1·id</span> Identity variant</td><td>A's photo</td><td><pre class="prompt">Same, with the sentence: Observer B is standing where you are standing, facing the same direction as you.</pre></td><td>A's spot</td></tr>
  <tr><td><span class="code">2</span> Ground-truth target</td><td>B's photo</td><td><pre class="prompt">In this image, is the X to the left or to the right of the Y? Answer with exactly one word: left or right.</pre></td><td>B's spot</td></tr>
  <tr><td><span class="code">4</span> Control (own view)</td><td>A's photo</td><td><pre class="prompt">Same direct question as 2.</pre></td><td>A's spot</td></tr>
  </tbody>
</table></div>

<section class="stack prose">
  <h2>Ground truth, and which examples count</h2>
  <p>As in the report, the correct answer for a view is read from the rendered instance masks: X is left of Y when the centre of X's visible pixels is left of the centre of Y's.
  An example counts only when, in both views, the objects' outlines do not overlap horizontally, their centres are at least 40 px apart, each observer stands at least 1.5 m from both objects, and the answer differs between A and B.</p>
  <p>We rendered {rendered} cross_line examples (B turned at least 120° from A, as in the report's examples) and {n} pass ({clean_pct}).
  Rejections: observer closer than 1.5 m in {rej_close}, overlapping outlines in {rej_overlap}, centres too close in {rej_px}, no flip in {rej_flip}.
  The report kept 92 of 177.</p>
  <p>The report's audit found that labels from the objects' 3D centres disagreed with the photo in 50 of 177 B views, nearly always because B stood right beside a large object (median 0.2 m from its centre).
  In our renders the two labels disagree in <b>{dis_a} A views and {dis_b} B views</b>. The difference has a known cause: in the apartment scenes the walkable-floor map treated furniture as floor, so observers could be placed inside a sofa.
  We fixed that before rendering, so no observer stands inside furniture, and the disagreement the report describes does not arise.</p>
  <p>Of the {n} clean examples, B's answer is “left” in {b_left} and “right” in {b_right}. The sentence about B adds “B stands next to …” in {next_to} and “B is facing …” in {facing}.
  A separate set of {train_clean} clean examples from {train_rooms} other rooms is set aside for the probe in the report's Section 3.</p>
</section>

<section class="stack prose">
  <h2>Results</h2>
  <p>Share of correct answers, and share of answers that equal the correct answer from A's spot. On cross_line these two always add up to 100%, minus unreadable answers.
  Shaded columns are the report's numbers on its 92 examples. Gemma 4 is an extra model, not in the report.</p>
</section>
<div class="scroll"><table>
  <thead><tr><th>Setting</th><th>Model</th><th>Correct</th><th>Answered from A</th><th class="rep">Report correct</th><th class="rep">Report from A</th></tr></thead>
  <tbody>{results_rows}</tbody>
</table></div>
<p class="prose small muted">{gemma_note}</p>

<section class="stack prose">
  <h2>Two different failures</h2>
  <p>Both report models fail the original task, but in different ways. The table splits setting 1 by B's correct answer.
  “Balanced” averages the two classes, so a model that always gives the same word scores 50% whatever the mix of the test set.</p>
</section>
<div class="scroll"><table>
  <thead><tr><th>Setting 1</th><th>Says “left”</th><th>Correct when B's answer is left</th><th>Correct when B's answer is right</th><th>Balanced</th></tr></thead>
  <tbody>{bias_rows}</tbody>
</table></div>
<section class="stack prose">
  <p><b>Qwen3-VL answers from A's spot.</b> It is right on few examples in both classes and far below 50% balanced, in our data as in the report's.</p>
  <p><b>BAGEL mostly says “right”, whatever the scene.</b> In the report all 31 of its correct answers were “right”, and here it says “left” in only {b1_left} of setting-1 answers.
  Its raw score therefore follows how often “right” happens to be correct: 39% of the report's examples, 54% of ours. That is why it reads 34% there and {b1_ok} here.
  Balanced, both datasets put BAGEL near chance. The report's “BAGEL answers from A's spot 66% of the time” mostly reflects this preference for one word.</p>
  <p><b>Outlining the objects does not help.</b> With X and Y outlined, every model reads left and right almost perfectly on A's or B's own photo (99–100%), so finding the objects is not the obstacle.
  On the original task the outlines leave Qwen3-VL at the same level, and BAGEL answers “right” to every outlined version of the original task.</p>
</section>

<section class="stack prose">
  <h2>Examples</h2>
  <p>Photos are shown with the outlines from the outlined condition so the objects are easy to find; in settings 1, 2 and 4 the models see them without outlines.
  Green cells are correct, red wrong.</p>
</section>
{cards}

<section class="stack prose">
  <h2>What differs from the report</h2>
  <ul>
    <li>New renders, rooms and object poses; {n} clean examples instead of 92. Images are 512×512.</li>
    <li>Observers cannot stand inside furniture, so the label audit finds no disagreements.</li>
    <li>The blue unit the report calls “tv stand” sits under the TV only in the apartment layouts, so we call it “blue shelf”, and the brown unit under the TV in the other layouts “sideboard”.</li>
    <li>The rule for adding “next to” and “facing” to the sentence about B is ours; the report does not state its rule.</li>
    <li>The report describes the identity sentence but does not quote it; ours is “Observer B is standing where you are standing, facing the same direction as you.”</li>
    <li>All models answer with greedy decoding; the report does not state its decoding settings.</li>
    <li>Added: the outlined condition, and Gemma 4 as a third model.</li>
    <li>Not done yet: the left/right probe in the report's Section 3, setting 3 (BAGEL drawing B's view).</li>
  </ul>
</section>
</main>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=str(ROOT / "local_data"))
    ap.add_argument("--out", default=str(ROOT / "local_data" / "report.html"))
    args = ap.parse_args()
    local = Path(args.local)
    page = build(local / "data", local / "results")
    Path(args.out).write_text(page)
    print(f"-> {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
