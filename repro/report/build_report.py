"""Replication report (Sections 1-3 of "Perspective Pilot Reproduction"), as one HTML page.

Reads data pulled from Turbo into local_data/ (see the rsync commands in the README/chat):
  local_data/data/{test,train}/examples_labeled.jsonl, local_data/data/scene_index/*.json,
  local_data/data/test/<scene>/<id>_{A,B}_rgb[_marked].png, local_data/results/<model>/test/*.jsonl,
  local_data/results/probe/<model>/{analysis.json,test/s1.jsonl} (Section 3, see probe_section.py)

    python repro/report/build_report.py --out local_data/report.html
"""
import argparse
import base64
import html
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import cv2

from probe_section import PROBE_CSS, build_probe

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


MARK = {None: "", "added": ' class="added"', "changed": ' class="changed"'}
TAG = {"added": '<span class="tag added">added</span>', "changed": '<span class="tag changed">changed</span>'}

TELLS = {
    "s1": "the real task: below the 50% of a coin flip for Qwen3-VL and Gemma",
    "s1_identity": "the same prompt is answered almost perfectly when no viewpoint change is needed",
    "s2": "B's photo, labelled from what it shows, is read about as well as A's",
    "s4": "left/right on A's own photo is easy",
}


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
            key = ("overlap" if "overlap" in t else "px" if "px apart" in t
                   else "close" if "stands" in t else "flip")
            reasons[key] += 1
    disagree = {v: sum(r[f"rel{v}"] != r[f"rel{v}_centroid"] for r in test) for v in "AB"}
    b_left = sum(r["relB"] == "left" for r in clean)

    # ---------- main results table (report layout) ----------
    def cells(key):
        s_ = st.get(key)
        if not s_:
            return '<td class="num muted">not run</td><td class="num muted">–</td>'
        flag = f'<div class="flag">{s_["bad"]} not one word</div>' if s_["bad"] else ""
        return (f'<td class="num">{s_["ok"]}/{s_["n"]} <b>{pct(s_["ok"], s_["n"])}</b>{flag}</td>'
                f'<td class="num">{s_["from_a"]}/{s_["n"]} <b>{pct(s_["from_a"], s_["n"])}</b></td>')

    def results_rows(marked):
        out = []
        for s, code, name, gets, truth in SETTINGS:
            for i, (m, label) in enumerate(MODELS):
                key = (m, s + ("_marked" if marked else ""))
                rep = None if marked else REPORT.get((s, m))
                rep_cells = (f'<td class="num rep">{rep[0]}/{REPORT_N} <b>{pct(rep[0], REPORT_N)}</b></td>'
                             f'<td class="num rep">{rep[1]}/{REPORT_N} <b>{pct(rep[1], REPORT_N)}</b></td>'
                             if rep else '<td class="num rep muted">–</td><td class="num rep muted">–</td>')
                head = (f'<td rowspan="{len(MODELS)}"><span class="code">{code}</span> <b>{e(name)}</b>'
                        f'{" · outlined" if marked else ""}</td>') if i == 0 else ""
                tells = (f'<td rowspan="{len(MODELS)}" class="tells">{e(TELLS[s])}</td>'
                         if i == 0 and not marked else ("<td rowspan=\"3\"></td>" if i == 0 else ""))
                cls = MARK["added"] if (marked or m not in REPORT_MODELS) else ""
                out.append(f'<tr{cls}>{head}<td class="model"><span class="m {m}">{e(label)}</span></td>'
                           f'{cells(key)}{rep_cells}{tells}</tr>')
        return "".join(out)

    # ---------- bias table ----------
    bias_rows = []
    for m, label in MODELS:
        for suffix, tag in (("", ""), ("_marked", " · outlined")):
            s_ = st.get((m, "s1" + suffix))
            if not s_:
                continue
            (hl, nl), (hr, nr) = s_["cls"]["left"], s_["cls"]["right"]
            bias_rows.append(f'<tr><td><span class="m {m}">{e(label)}</span>{tag}</td>'
                             f'<td class="num">{pct(s_["said_left"], s_["n"])}</td>'
                             f'<td class="num">{hl}/{nl} ({pct(hl, nl)})</td><td class="num">{hr}/{nr} ({pct(hr, nr)})</td>'
                             f'<td class="num"><b>{(hl / nl + hr / nr) / 2:.0%}</b></td></tr>')
    for m, label in MODELS:
        if m in REPORT_S1_CLASS:
            (hl, hr), (nl, nr) = REPORT_S1_CLASS[m], REPORT_CLASS_N
            bias_rows.append(f'<tr class="rep-row"><td><span class="m {m}">{e(label)}</span> · report</td><td class="num muted">–</td>'
                             f'<td class="num">{hl}/{nl} ({pct(hl, nl)})</td><td class="num">{hr}/{nr} ({pct(hr, nr)})</td>'
                             f'<td class="num"><b>{(hl / nl + hr / nr) / 2:.0%}</b></td></tr>')

    # ---------- examples ----------
    def outcome(m, r, s="s1"):
        a = res.get((m, s), {}).get(r["id"])
        return None if a is None else a["answer"]

    picked, seen = [], set()
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
    cards = []
    for r in picked[:6]:
        d = data / "test" / r["scene"]
        imgs = {v: thumb(d / f"{r['id']}_{v}_rgb_marked.png") for v in "AB"}
        ans_rows = []
        for m, label in MODELS:
            tds = []
            for s in ("s1", "s1_identity", "s2", "s4", "s1_marked"):
                a = outcome(m, r, s)
                truth = r["relB"] if s in ("s1", "s1_marked", "s2") else r["relA"]
                tds.append(f'<td class="ans {"ok" if a == truth else "bad"}">{e(a or "–")}</td>')
            ans_rows.append(f'<tr{MARK["added"] if m not in REPORT_MODELS else ""}><td><span class="m {m}">{e(label)}</span></td>{"".join(tds)}</tr>')
        cards.append(f"""
<article class="ex">
  <header><h4>{e(r['id'])}</h4><span class="muted small">{e(r['scene'])}</span></header>
  <p>Question: is the <span class="x">{e(r['X'])}</span> left or right of the <span class="y">{e(r['Y'])}</span>?<br>
     Correct answer from A's spot: <b>{r['relA']}</b> · from B's spot: <b>{r['relB']}</b></p>
  <p class="sentence">Sentence about B: {e(r['sentence_B'])}</p>
  <div class="views">
    <figure><img src="{imgs['A']}" alt="A's photo with X and Y outlined"><figcaption><b>A's photo</b> · input for 1 and 4</figcaption></figure>
    <figure><img src="{imgs['B']}" alt="B's photo with X and Y outlined"><figcaption><b>B's real photo</b> · input for 2</figcaption></figure>
    <figure>{map_svg(r, furniture(data, r['scene']))}<figcaption><b>Map from above</b> (<span class="a">A</span>, <span class="b">B</span>, dashed X–Y line)</figcaption></figure>
  </div>
  <div class="scroll"><table class="answers">
    <thead><tr><th>setting</th><th>1 Original</th><th>1 Identity</th><th>2 Ground-truth target</th><th>4 Control</th><th class="addhead">1 outlined</th></tr></thead>
    <tbody>{''.join(ans_rows)}</tbody>
  </table></div>
</article>""")

    probe_compare, probe_models = build_probe(results, MODELS)
    marked_compare, marked_models = build_probe(results, MODELS, "_marked")
    g = st.get(("gemma4", "s1_identity_marked"), {})
    q1, b1 = st[("qwen3vl", "s1")], st[("bagel", "s1")]
    ctrl = [st[(m, s)]["ok"] / st[(m, s)]["n"] for m in REPORT_MODELS for s in ("s1_identity", "s2", "s4")]
    return PAGE.format(
        n=n, rooms=len({r["scene"] for r in clean}), rendered=len(test), clean_pct=pct(n, len(test)),
        train_clean=sum(r["clean"] for r in train), train_rooms=len({r["scene"] for r in train if r["clean"]}),
        q1_ok=pct(q1["ok"], q1["n"]), q1_a=pct(q1["from_a"], q1["n"]), b1_ok=pct(b1["ok"], b1["n"]),
        b1_left=pct(b1["said_left"], b1["n"]), ctrl_lo=f"{min(ctrl):.0%}", ctrl_hi=f"{max(ctrl):.0%}",
        rej_close=reasons["close"], rej_overlap=reasons["overlap"], rej_px=reasons["px"], rej_flip=reasons["flip"],
        dis_a=disagree["A"], dis_b=disagree["B"], b_left=b_left, b_right=n - b_left,
        next_to=sum("next to" in r["sentence_B"] for r in clean), facing=sum("is facing" in r["sentence_B"] for r in clean),
        rows_plain=results_rows(False), rows_marked=results_rows(True), bias_rows="".join(bias_rows),
        cards="".join(cards), probe_compare=probe_compare, probe_models=probe_models, marked_compare=marked_compare, marked_models=marked_models, gemma_bad=g.get("bad", 0), date=date.today().isoformat(),
    )


PAGE = """<title>Perspective Replication</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout follows the report being replicated: header, legend, placement table, settings table, results, examples, notes.
   Violet rule = added in this replication; amber rule = changed from the report's method. */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --ink: #18202b; --muted: #5e6a79; --line: #dfe3e9; --rep: #eef1f6;
  --accent: #2f5f8a; --violet: #7c4dce; --violet-soft: #f1ebfb; --amber: #b7791f; --amber-soft: #fbf3e4;
  --x: #d9661f; --y: #2b7fd4; --a: #2f5fb3; --b: #c23b3b; --qwen: #6b4fbb; --bagel: #b15a1e; --gemma4: #2a7d6b;
  --ok: #1e7a4a; --ok-soft: #e3f3ea; --bad: #b3362d; --bad-soft: #fbe7e5; --furn: #d3d9e1;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12161c; --panel: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --line: #2c3542; --rep: #1f2733;
  --accent: #8fb6dd; --violet: #b08cf0; --violet-soft: #2a2140; --amber: #e0a94e; --amber-soft: #33281a;
  --x: #f08a4b; --y: #62a6ec; --a: #7aa2e8; --b: #ec7a72; --qwen: #ab95ec; --bagel: #e59a63; --gemma4: #5cc4ab;
  --ok: #6fd39c; --ok-soft: #173326; --bad: #f08a80; --bad-soft: #3a1d1b; --furn: #333d4b; --c1: #3987e5; --c2: #d95926; --c3: #199e70; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12161c; --panel: #1a2029; --ink: #e5e9ef; --muted: #98a3b2; --line: #2c3542; --rep: #1f2733;
  --accent: #8fb6dd; --violet: #b08cf0; --violet-soft: #2a2140; --amber: #e0a94e; --amber-soft: #33281a;
  --x: #f08a4b; --y: #62a6ec; --a: #7aa2e8; --b: #ec7a72; --qwen: #ab95ec; --bagel: #e59a63; --gemma4: #5cc4ab;
  --ok: #6fd39c; --ok-soft: #173326; --bad: #f08a80; --bad-soft: #3a1d1b; --furn: #333d4b; --c1: #3987e5; --c2: #d95926; --c3: #199e70; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 16.5px/1.6 var(--body); }}
main {{ max-width: 1120px; margin: 0 auto; padding-block: 36px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 22px; }}
.prose {{ max-width: 72ch; display: grid; gap: 12px; }}
h1, h2, h3, h4 {{ font-family: var(--display); text-wrap: balance; line-height: 1.15; margin: 0; }}
h1 {{ font-size: clamp(2rem, 5vw, 3rem); font-weight: 700; }}
h2 {{ font-size: 1.7rem; font-weight: 700; margin-top: 30px; padding-top: 18px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 1.25rem; font-weight: 700; margin-top: 10px; }}
h4 {{ font: 600 0.95rem var(--mono); }}
p {{ margin: 0; }}
.meta {{ font: 600 0.78rem/1.5 var(--mono); letter-spacing: 0.04em; color: var(--muted); }}
.lede {{ font-size: 1.12rem; }}
.muted {{ color: var(--muted); }} .small {{ font-size: 0.85rem; }}
.headline {{ font-size: 1.08rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 0.88rem; align-items: center;
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
.swatch {{ display: inline-block; width: 4px; height: 16px; margin-right: 6px; vertical-align: middle; }}
.m {{ font-weight: 600; }} .m.qwen3vl {{ color: var(--qwen); }} .m.bagel {{ color: var(--bagel); }} .m.gemma4 {{ color: var(--gemma4); }}
.x {{ color: var(--x); font-weight: 600; }} .y {{ color: var(--y); font-weight: 600; }}
.a {{ color: var(--a); font-weight: 700; }} .b {{ color: var(--b); font-weight: 700; }}
.added {{ border-left: 4px solid var(--violet); padding-left: 14px; }}
.changed {{ border-left: 4px solid var(--amber); padding-left: 14px; }}
tr.added > td:first-child, tr.added > th:first-child {{ box-shadow: inset 4px 0 0 var(--violet); }}
tr.changed > td:first-child {{ box-shadow: inset 4px 0 0 var(--amber); }}
th.addhead {{ box-shadow: inset 0 -3px 0 var(--violet); }}
.tag {{ font: 600 0.68rem/1 var(--mono); text-transform: uppercase; letter-spacing: 0.05em; padding: 3px 6px; border-radius: 4px; margin-left: 6px; vertical-align: 2px; }}
.tag.added {{ background: var(--violet-soft); color: var(--violet); border: 0; padding-left: 6px; }}
.tag.changed {{ background: var(--amber-soft); color: var(--amber); border: 0; padding-left: 6px; }}
.scroll {{ overflow-x: auto; min-width: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; background: var(--panel); }}
th, td {{ padding: 8px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }}
thead th {{ font: 600 0.74rem/1.3 var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); background: var(--bg); }}
td.num {{ font-family: var(--mono); font-size: 0.84rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.rep, th.rep, tr.rep-row td {{ background: var(--rep); }}
td.tells {{ color: var(--muted); font-size: 0.88rem; max-width: 30ch; }}
.code {{ font-family: var(--mono); font-weight: 600; color: var(--accent); }}
.flag {{ font-size: 0.72rem; color: var(--bad); }}
tr.notrun td {{ color: var(--muted); }}
pre.prompt {{ white-space: pre-wrap; font: 0.82rem/1.5 var(--mono); margin: 0; }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
li.added, li.changed {{ list-style: none; margin-left: -1.2em; }}
.ex {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; display: grid; gap: 10px; }}
.ex header {{ display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }}
.sentence {{ color: var(--muted); font-size: 0.93rem; }}
.views {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
figure {{ margin: 0; min-width: 0; }}
figure img, .map {{ width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid var(--line); }}
figcaption {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
.map {{ background: var(--panel); }} .furn {{ fill: var(--furn); }}
.xyline {{ stroke: var(--muted); stroke-dasharray: 4 3; stroke-width: 1.5; fill: none; }}
.obsA {{ fill: var(--a); }} .obsB {{ fill: var(--b); }} .wedge {{ opacity: 0.22; }}
.objX {{ fill: var(--x); }} .objY {{ fill: var(--y); }} .lbl {{ fill: var(--ink); font: 700 12px var(--body); }}
td.ans {{ font-family: var(--mono); font-size: 0.84rem; }}
td.ans.ok {{ background: var(--ok-soft); color: var(--ok); }} td.ans.bad {{ background: var(--bad-soft); color: var(--bad); }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; display: grid; gap: 10px; }}
code {{ font-family: var(--mono); font-size: 0.86em; }}
</style>
<main>
<header class="prose">
  <p class="meta">Spatial Perspectives · Replication · {date} · ReplicaCAD in Habitat-Sim · {n} clean of {rendered} rendered cross_line examples · {rooms} rooms</p>
  <h1>Can the model see from B's spot? Replicated.</h1>
  <p class="lede">The setup is the report's: person A takes a photo of a room, person B stands somewhere else in the same room. The model gets A's photo and a sentence that says where B stands, and is asked: “From B's point of view, is object X on the left or on the right of object Y?”
  We rebuilt the data pipeline from scratch on new ReplicaCAD renders and asked <span class="m qwen3vl">Qwen3-VL</span> and <span class="m bagel">BAGEL</span> the report's questions.
  Additions to the report carry a <b style="color:var(--violet)">violet rule</b>; changes to its method carry an <b style="color:var(--amber)">amber rule</b>.</p>
  <p class="headline"><b>Headline.</b> On {n} clean cross_line examples, Qwen3-VL is right {q1_ok} of the time and gives A's answer {q1_a} of the time (report: 20% and 80%), while both report models score {ctrl_lo}–{ctrl_hi} on the three controls (report: 91–97%).
  BAGEL scores {b1_ok} (report: 34%), but it says “left” in only {b1_left} of its answers: in both datasets it mostly answers “right” whatever it is shown, so its score tracks the test set's left/right mix.
  Inside the models, left/right readers find A's answer or nothing readable at every trusted layer, never B's, as the report found.</p>
</header>

<div class="legend" role="note" aria-label="Legend">
  <span><span class="dot" style="background:var(--a)"></span>A (the photo the model gets)</span>
  <span><span class="dot" style="background:var(--b)"></span>B (the view we ask about)</span>
  <span><span class="dot" style="background:var(--x)"></span>object X</span>
  <span><span class="dot" style="background:var(--y)"></span>object Y</span>
  <span class="m qwen3vl">Qwen3-VL-8B-Instruct</span><span class="m bagel">BAGEL-7B-MoT</span><span class="m gemma4">Gemma 4 12B</span>
  <span><span class="swatch" style="background:var(--violet)"></span>added in this replication</span>
  <span><span class="swatch" style="background:var(--amber)"></span>changed from the report</span>
</div>

<h2>Four ways to place B</h2>
<div class="prose"><p>The report places B in four ways. This replication covers the report's cross_line test at scale, the one placement where A and B disagree; the other three were not rendered.</p></div>
<div class="scroll"><table>
  <thead><tr><th>Mode</th><th>Where B stands</th><th>Answer differs for A and B?</th><th>Examples</th></tr></thead>
  <tbody>
  <tr class="notrun"><td><b>rotate_in_place</b></td><td>Same spot as A, only turns</td><td>no</td><td>not run</td></tr>
  <tr class="notrun"><td><b>general</b></td><td>0.5–3 m away, on A's side of the X–Y line</td><td>no</td><td>not run</td></tr>
  <tr class="notrun"><td><b>shared_target</b></td><td>0.5–3 m away, looking at A's target</td><td>no</td><td>not run</td></tr>
  <tr class="changed"><td><b>cross_line</b><span class="tag changed">changed</span></td><td>Far side of the two objects, looking back. Here B must also turn at least 120° from A's heading, as in the report's examples (the report states no rule).</td><td>yes, always</td><td>{n} clean</td></tr>
  </tbody>
</table></div>

<h2>Experimental settings</h2>
<div class="prose"><p>Settings keep the report's numbers. X and Y are the two objects; [where B stands and looks] is the sentence about B (distance and direction from A, B's turn, and sometimes what B is next to or facing).
The “can you see both objects?” pre-question is not used, as in the report's large cross_line set, because B sees both objects by construction.</p></div>
<div class="scroll"><table>
  <thead><tr><th>Setting</th><th>The model gets</th><th>What we test</th><th>Question template</th><th>Correct when</th></tr></thead>
  <tbody>
  <tr><td><span class="code">1</span> <b>Original</b></td><td>A's photo + sentence about B</td><td>Direct understanding: imagine B's view and answer from it.</td><td><pre class="prompt">This image shows what you (observer A) currently see. Another observer, B, is in the same room. [where B stands and looks] From B's point of view, is the X to the left or to the right of the Y? Answer with exactly one word: left or right.</pre></td><td>the answer matches the truth from <b>B's</b> spot</td></tr>
  <tr class="changed"><td><span class="code">1</span> <b>Identity variant</b><span class="tag changed">changed</span></td><td>A's photo + “B stands where you stand”</td><td>Is it the change of viewpoint that hurts, or the extra sentence?</td><td><pre class="prompt">Same as 1, with: Observer B is standing where you are standing, facing the same direction as you.</pre><span class="muted small">The report describes this sentence but does not quote it; this wording is ours.</span></td><td><b>A's</b> spot</td></tr>
  <tr><td><span class="code">2</span> <b>Ground-truth target</b></td><td>B's photo</td><td>Control: understanding B's view when it is simply given.</td><td><pre class="prompt">In this image, is the X to the left or to the right of the Y? Answer with exactly one word: left or right.</pre></td><td><b>B's</b> spot</td></tr>
  <tr class="notrun"><td><span class="code">3</span> <b>Generated target</b></td><td>BAGEL's drawing of B's view</td><td>Generation + understanding</td><td>not run (the report also skipped it on the large set)</td><td>–</td></tr>
  <tr><td><span class="code">4</span> <b>Control (own view)</b></td><td>A's photo</td><td>Can the model read left/right at all?</td><td><pre class="prompt">Same direct question as 2.</pre></td><td><b>A's</b> spot</td></tr>
  <tr class="added"><td><b>1, 1 identity, 2, 4 · outlined</b><span class="tag added">added</span></td><td>The same photos with X outlined orange and Y outlined blue</td><td>Does finding the objects explain the errors?</td><td><pre class="prompt">Same templates, naming the outlines: … is the X (outlined in orange) to the left or to the right of the Y (outlined in blue)? …</pre></td><td>as in the plain setting</td></tr>
  </tbody>
</table></div>

<h2>Experimental results</h2>
<h3>Ground truth, and which examples count</h3>
<div class="prose">
  <p>As in the report, the correct answer for a view is read from the rendered instance masks: X is “left of” Y when the centre of X's visible pixels is left of the centre of Y's.
  An example is kept only when, in both views, the two objects' outlines do not overlap horizontally, their centres are at least 40 px apart, each observer stands at least 1.5 m from both objects, and the answer differs for A and B.</p>
  <p>Of {rendered} rendered cross_line examples, {n} pass ({clean_pct}); the report kept 92 of 177. Rejections: an observer within 1.5 m of an object in {rej_close}, overlapping outlines in {rej_overlap}, centres too close in {rej_px}, no flip in {rej_flip}.
  Of the {n}, B's answer is “left” in {b_left} and “right” in {b_right}.</p>
  <div class="changed prose"><p><span class="tag changed" style="margin-left:0">changed</span> The report's audit found that labels from each object's 3D centre disagreed with the photo in 50 of 177 B views, almost always because B stood right beside a large object (median 0.2 m from its centre).
  We traced this to the walkable-floor map: in the apartment scenes it treats furniture as floor, so an observer can be placed inside a sofa. We block all furniture when building the map and reject any pose inside an object's footprint.
  With that fix the 3D-centre and mask labels disagree in <b>{dis_a} of {rendered} A views and {dis_b} of {rendered} B views</b>, so the effect the audit measured does not occur here, and the clean rate is higher than the report's.</p></div>
</div>

<h3>The cross_line test at scale: {n} clean examples</h3>
<div class="prose"><p>The four settings the report ran on its 92 examples, for both report models. Shaded columns are the report's numbers. On cross_line, “correct” and “answered from A's spot” add up to 100% except for answers that are not one word, which count as wrong.</p></div>
<div class="scroll"><table>
  <thead><tr><th>Setting</th><th>Model</th><th>Correct</th><th>Answered from A's spot</th><th class="rep">Report correct</th><th class="rep">Report from A</th><th>What it tells us</th></tr></thead>
  <tbody>{rows_plain}</tbody>
</table></div>
<div class="prose">
  <p><b>How to read it.</b> Every answer is one of two words, so a model that guessed would be right about half the time.
  In the real task (1) Qwen3-VL is right {q1_ok} of the time, and {q1_a} of its answers are the correct answer for A's spot: it reads the left/right relation off the photo it holds and reports it as B's, as the report found.
  The identity variant uses the same prompt with a sentence about B and is answered almost perfectly, so the sentence is not what confuses the models; on A's and B's own photos (4, 2) they read left and right reliably.</p>
  <p class="added"><span class="tag added" style="margin-left:0">added</span> <span class="m gemma4">Gemma 4</span> is a third model, run with the same prompts. It shows the same failure as Qwen3-VL, less strongly, and is weaker at reading left/right on B's photo.</p>
</div>

<h3 class="added">Why BAGEL scores near 50%<span class="tag added">added</span></h3>
<div class="prose added"><p>Splitting setting 1 by B's correct answer separates two kinds of failure. “Balanced” averages the two classes, so a model that always gives the same word scores 50% whatever the left/right mix of the test set.</p></div>
<div class="scroll"><table>
  <thead><tr><th>Setting 1</th><th>Says “left”</th><th>Correct when B's answer is left</th><th>Correct when B's answer is right</th><th>Balanced</th></tr></thead>
  <tbody>{bias_rows}</tbody>
</table></div>
<div class="prose added">
  <p><b>Qwen3-VL answers from A's spot</b>: it is rarely right in either class, far below 50% balanced, in our data as in the report's.</p>
  <p><b>BAGEL mostly says “right”, whatever it is shown.</b> In the report all 31 of its correct answers were “right”; here it says “left” in {b1_left} of setting-1 answers.
  Its raw score therefore follows how often “right” happens to be correct: 39% of the report's examples and 54% of ours, hence 34% there and {b1_ok} here. Balanced, both put BAGEL near chance.
  The report's figure of 66% answered from A's spot mostly reflects this preference for one word, not a reading of A's view.</p>
</div>

<h3 class="added">Outlined photos: finding the objects is not the obstacle<span class="tag added">added</span></h3>
<div class="prose added"><p>The same four settings with X outlined orange and Y outlined blue, and the colours named in the question. If the models failed because they could not find the objects, the outlines would fix it.</p></div>
<div class="scroll"><table>
  <thead><tr><th>Setting</th><th>Model</th><th>Correct</th><th>Answered from A's spot</th><th class="rep">Report correct</th><th class="rep">Report from A</th><th></th></tr></thead>
  <tbody>{rows_marked}</tbody>
</table></div>
<div class="prose added">
  <p>With outlines, every model reads left and right on A's or B's own photo almost perfectly (99–100%); Gemma's 84% on B's plain photo rises to 100%.
  The original task does not improve: Qwen3-VL stays far below chance and BAGEL answers “right” every time. The failure is in taking B's viewpoint, not in recognising the objects.</p>
  <p class="small muted">Gemma's outlined identity row has {gemma_bad} answers that are not one word: Gemma ignored “Answer with exactly one word” and described both objects (“The kitchen counter (orange) is on the left side of the image. The blue shelf (blue) is on the right…”). They count as wrong, as for every model. This setting was rerun with a 32-token answer limit; all others use 5.</p>
</div>

<h2>Inside the model: does it ever work out B's answer?</h2>
<div class="prose">
  <h3>Why look inside</h3>
  <p>In setting 1 the models mostly answer from A's spot. Two stories fit that behaviour. <b>Never imagined</b>: the model reads left/right off A's photo and never works out what B would see.
  <b>Imagined, then overwritten</b>: somewhere in the middle layers it does work out B's answer, but A's answer wins by the last layer. Behaviour cannot separate them; reading the hidden states layer by layer can.</p>
  <h3>The tool: a left/right reader</h3>
  <p>As in the report, a reader is a linear classifier on one layer's hidden state at the position where the model is about to answer (for BAGEL, the start token of its first generation step). One reader per layer is trained on direct questions on A's and B's photos (settings 4 and 2) from the training rooms,
  labelled with the model's own answer, then applied unchanged to setting 1 on the {n} clean test examples. Readers are class-weighted logistic regressions on 256 principal components.
  A second reader per layer is trained on the identity prompt instead (the task's wording, but B at A's spot).</p>
  <p><b>B score</b>: how often the reader's verdict on setting 1 is B's answer rather than A's, averaged over B's two answer classes, so 50% is no information, 0% is A's answer everywhere, 100% B's.
  <b>Check 1</b>: at the last layer the reader must match what the model actually said in setting 1, to at least 95% of its own ceiling (held-out agreement on the questions it was trained on).
  <b>Check 2</b>: on the identity prompt, where B's answer equals A's, the B score must be at least 80%; only layers that pass are read.
  Outcomes, per trusted layer: below the chance band of 100 shuffled-label readers = holds A's answer; inside = near chance; above = holds B's answer.</p>
  <div class="changed prose"><p><span class="tag changed" style="margin-left:0">changed</span> Details the report does not state, chosen by us:
  training examples are the {train_clean} clean cross_line examples from {train_rooms} training rooms, each asked on A's and on B's photo; features are standardized before PCA;
  the ceiling is 5-fold cross-validation grouped by room; logistic regression uses C = 1; the chance band uses 100 shuffled-label readers and the confidence interval 1000 bootstrap resamples of the test examples.
  For Qwen3-VL and Gemma, layer 0 is the input embedding; for BAGEL, layer 0 is the token embedding and the states are read before the final norm.</p></div>
  <h3>What we found</h3>
  <p>On the two report models the conclusion replicates: wherever a reader can be trusted, the state holds A's answer or no readable answer, never B's. Shaded rows are the report's figures.</p>
</div>
{probe_compare}
<div class="prose">
  <p>Qwen3-VL matches closely, including the split between the two readers: the direct reader fails Check 1 but the identity-trained reader passes, and every trusted layer holds A's answer.
  BAGEL reaches the same outcome (about half the trusted layers hold A's answer, half are near chance), with lower Check 1 agreement than the report's. Check 1 is class-balanced over the model's own setting-1 answers, and our BAGEL says “left” in only a few of them, so half of that score rests on a handful of examples.</p>
  <p class="added"><span class="tag added" style="margin-left:0">added</span> <span class="m gemma4">Gemma 4</span> fails Check 1 with both readers, like BAGEL, and its trusted layers (the last 15) hold A's answer or nothing readable; none holds B's.</p>
  <p class="small muted">Hover a chart for each layer's numbers; the same numbers are in the table under each pair. In the B-score chart the shaded columns are untrusted layers, the grey band is chance, the blue wash the bootstrap interval; the dashed line is the report's curve.</p>
</div>
{probe_models}

<h3 class="added">The probe on outlined photos<span class="tag added">added</span></h3>
<div class="prose added">
  <p>The same readers, trained and tested on the outlined photos (X orange, Y blue, colours named in the question): direct questions on outlined A and B photos and the outlined identity prompt in the training rooms; outlined setting 1 and identity prompt in the test rooms.
  With outlines every model reads left/right on its own or B's photo almost perfectly, so if B's answer appears anywhere inside, it should appear here.</p>
  <p><b>Result: still no B's answer, in any model or layer.</b> The outlines make the readers sharper but do not change the conclusion.</p>
  <ul>
    <li><span class="m qwen3vl">Qwen3-VL</span> gives the clearest picture of the study: from layer 22 on, the reader's verdict is A's answer on almost every example (B score 0–4%).
    Both checks miss the 95% bar narrowly (90% and 94%), so the reader transfers less than perfectly, but nothing in the trusted layers points to B.</li>
    <li><span class="m bagel">BAGEL</span> answers “right” on every outlined setting-1 example, so Check 1 is undefined. Inside, though, layers 19–25 hold A's answer almost perfectly (B score 1–12%):
    the model works out A's view and then says “right” regardless. It never works out B's.</li>
    <li><span class="m gemma4">Gemma 4</span> is the weakest read: at most trusted layers the reader says “right” for every example, so they carry no information (Outcome C).
    At the last layer the B score is 33%, on A's side. Among the examples Gemma answers correctly, B's answer does appear at the last layer (72%), as it did for Qwen3-VL on plain photos: when the model gets it right, the correction comes late.</li>
  </ul>
  <p class="small muted">The Gemma run uses a 32-token answer limit, since on outlined photos it sometimes answers in a sentence; its setting-1 answers still match the Section 2 eval exactly. Twelve of its outlined identity-prompt training answers are not one word and are left out of the identity reader's training.</p>
</div>
{marked_compare}
{marked_models}

<h2>Examples</h2>
<div class="prose"><p>Six clean examples from different rooms, chosen to show each kind of outcome. Photos are shown with the outlines from the outlined condition; in settings 1, 2 and 4 the models see them without outlines. Green is correct, red wrong.</p></div>
{cards}

<h2>Notes for the next iteration</h2>
<div class="prose">
  <h3>What differs from the report, and why</h3>
  <ul>
    <li class="changed"><b>Examples are new.</b> Rendered from scratch in 14 test rooms ({n} clean); the report had 92. Images are 512×512.</li>
    <li class="changed"><b>No observer inside furniture.</b> See the ground-truth section; the label audit therefore finds no disagreements.</li>
    <li class="changed"><b>Object names.</b> The blue unit the report calls “tv stand” sits under the TV only in the apartment layouts, so we call it “blue shelf”, and the brown unit under the TV in the other layouts “sideboard”. The two are never paired with the TV screen.</li>
    <li class="changed"><b>Sentence about B.</b> “next to” is added when another named object is within 1 m of B ({next_to} of {n}), “facing” when one is within 10° of B's view centre ({facing} of {n}). The report does not state its rule.</li>
    <li class="changed"><b>B turns at least 120°</b> from A's heading, matching the report's cross_line examples.</li>
    <li class="changed"><b>Identity sentence</b> worded by us; the report only describes it.</li>
    <li class="changed"><b>Decoding.</b> Greedy, at most 5 new tokens (32 for one Gemma setting). The report does not state its settings.</li>
  </ul>
  <h3>What was added</h3>
  <ul>
    <li class="added"><b>Outlined condition</b> for all four settings.</li>
    <li class="added"><b>Gemma 4 12B</b> as a third model.</li>
    <li class="added"><b>Per-class breakdown</b> of setting 1, which explains BAGEL's score.</li>
    <li class="added"><b>Left/right probe on Gemma 4</b>, and the probe on outlined photos for all three models.</li>
  </ul>
  <h3>Not done</h3>
  <ul>
    <li>Setting 3 (BAGEL drawings), the person-in-photo variants, and the three placements other than cross_line.</li>
    <li>Molmo2-8B was tried and dropped: its code does not run under the current transformers version.</li>
  </ul>
  <h3>Knobs and files</h3>
  <ul>
    <li>Render: <code>repro/render/render_examples.py</code>; labels, clean rule and sentences: <code>label_and_clean.py</code>; outlines: <code>make_marked.py</code>; settings and paths: <code>repro/common/config.py</code>, <code>vocab.py</code>.</li>
    <li>Models: <code>repro/eval/run_vlm.py</code> (Qwen3-VL, Gemma 4), <code>run_bagel.py</code>; prompts in <code>prompts.py</code>; tables: <code>summarize.py</code>; this page: <code>repro/report/build_report.py</code>.</li>
    <li>Probe: hidden states <code>repro/probing/extract_vlm.py</code>, <code>extract_bagel.py</code>; readers and checks <code>analyze.py</code>; Section 3 of this page <code>repro/report/probe_section.py</code>.</li>
  </ul>
</div>
</main>
"""

PAGE = PAGE.replace("</style>", PROBE_CSS + "</style>", 1)


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
