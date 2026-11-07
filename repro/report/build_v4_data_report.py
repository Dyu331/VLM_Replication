"""Visual check of the v4 data (Phases 0-2): object pool, example scenes, bookkeeping, checks.

Runs locally on data copied from Turbo into local_data/v4/ (needs opencv + numpy):

    python repro/report/build_v4_data_report.py --list > /tmp/files.txt   # images it needs
    rsync -a --files-from=/tmp/files.txt greatlakes:$PP_DATA/v4/ local_data/v4/
    python repro/report/build_v4_data_report.py                       # -> local_data/v4/v4_data_report.html

Display only: the outlines (B green, X orange, Y blue) are drawn here from the masks; the model's
input is the plain photo, which the "plain photo" switch shows.
"""
import argparse
import base64
import html
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "local_data/v4"
COL = {"B": (46, 160, 67), "X": (235, 120, 30), "Y": (40, 110, 230)}  # RGB, as in the PhD's cards
PROMPT = ("This image shows what you currently see. Imagine you are standing where the {B} is, looking toward "
          "the {X} and the {Y}. From there, is the {X} to the left or to the right of the {Y}? Answer with "
          "exactly one word: left or right.")
HFOV = 90.0


def esc(s):
    return html.escape(str(s))


def load_questions(set_name, split):
    p = LOCAL / set_name / f"questions_{split}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def pick_cards(qs, n, seed):
    """Up to n questions covering flip/no-flip x depth rank, from different photos."""
    rnd = random.Random(seed)
    groups = {}
    for q in qs:
        groups.setdefault((q["flip"], q["depth_rank"]), []).append(q)
    order = [(f, r) for f in (True, False) for r in ("nearest", "middle", "farthest")]
    out, seen = [], set()
    while len(out) < n and any(groups.get(k) for k in order):
        for k in order:
            pool = [q for q in groups.get(k, []) if q["photo"] not in seen]
            if pool and len(out) < n:
                q = rnd.choice(pool)
                out.append(q)
                seen.add(q["photo"])
                groups[k].remove(q)
    return out


def pick_gallery(qs, n, seed, exclude):
    rnd = random.Random(seed + 1)
    pool = [q for q in qs if q["photo"] not in exclude]
    rnd.shuffle(pool)
    out, seen = [], set()
    for q in pool:
        if q["photo"] not in seen:
            out.append(q)
            seen.add(q["photo"])
        if len(out) == n:
            break
    return out


def needed_files(set_name, cards, gallery):
    files = set()
    for q in cards:
        files |= {q["img_A"], q["sem_A"], q["img_B"], q["sem_B"]}
    for q in gallery:
        files |= {q["img_A"], q["sem_A"]}
    return sorted(f"{set_name}/{f}" for f in files)


def jpeg(img_rgb, size, q=82):
    im = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def outlined(img, sem, roles):
    out = img.copy()
    for role, sid in roles.items():
        m = (sem == sid).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, COL[role], 3)
    return out


# ---------- top-down sketch ----------

def sketch(q, size=300):
    objs = {o["sid"]: o for o in q["objects"]}
    role_of = {q["sid"][r]: r for r in "BXY"}
    pts = [p for o in q["objects"] for p in o["foot"]] + [[q["A_pos"][0], q["A_pos"][2]]]
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    span = max(max(xs) - min(xs), max(zs) - min(zs)) + 2.4
    s = size / span

    def P(x, z):
        return (size / 2 + (x - cx) * s, size / 2 + (z - cz) * s)

    el = [f'<svg viewBox="0 0 {size} {size}" class="sketch" role="img" aria-label="Top-down sketch">',
          f'<rect width="{size}" height="{size}" class="sk-bg"/>']
    g0, g1 = math.floor(cx - span / 2), math.ceil(cx + span / 2)
    for g in range(g0, g1 + 1):
        x, _ = P(g, 0)
        el.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{size}" class="sk-grid"/>')
    for g in range(math.floor(cz - span / 2), math.ceil(cz + span / 2) + 1):
        _, y = P(0, g)
        el.append(f'<line x1="0" y1="{y:.1f}" x2="{size}" y2="{y:.1f}" class="sk-grid"/>')
    # line through X and Y, extended
    X, Y = objs[q["sid"]["X"]]["centre"], objs[q["sid"]["Y"]]["centre"]
    dx, dz = Y[0] - X[0], Y[2] - X[2]
    L = math.hypot(dx, dz) or 1
    a, b = P(X[0] - dx / L * span, X[2] - dz / L * span), P(Y[0] + dx / L * span, Y[2] + dz / L * span)
    el.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" class="sk-line"/>')
    # view cones
    for who, pos, yaw in (("A", q["A_pos"], q["A_yaw"]), ("B", q["B_pos"], q["B_yaw"])):
        o = P(pos[0], pos[2])
        ray = []
        for d in (-HFOV / 2, HFOV / 2):
            r = math.radians(yaw + d)
            ray.append(P(pos[0] - math.sin(r) * 1.6, pos[2] - math.cos(r) * 1.6))
        el.append(f'<path d="M{o[0]:.1f},{o[1]:.1f} L{ray[0][0]:.1f},{ray[0][1]:.1f} L{ray[1][0]:.1f},{ray[1][1]:.1f} Z" '
                  f'class="sk-cone sk-cone-{who}"/>')
    for sid, o in objs.items():
        role = role_of[sid]
        d = " ".join(f"{'M' if i == 0 else 'L'}{P(*p)[0]:.1f},{P(*p)[1]:.1f}" for i, p in enumerate(o["foot"])) + " Z"
        el.append(f'<path d="{d}" class="sk-obj sk-{role}"/>')
        c = P(o["centre"][0], o["centre"][2])
        el.append(f'<text x="{c[0]:.1f}" y="{c[1] + 4:.1f}" class="sk-lab">{role}</text>')
    a = P(q["A_pos"][0], q["A_pos"][2])
    el.append(f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="6" class="sk-A"/>'
              f'<text x="{a[0]:.1f}" y="{a[1] + 4:.1f}" class="sk-lab sk-lab-A">A</text>')
    el.append(f'<text x="6" y="{size - 8}" class="sk-note">1 m grid · dashed: line through X and Y</text></svg>')
    return "".join(el)


# ---------- page pieces ----------

def chip(ok, text=None):
    return f'<span class="chip {"ok" if ok else "bad"}">{esc(text or ("pass" if ok else "fail"))}</span>'


def card(q, set_name):
    d = LOCAL / set_name
    A, semA = read_rgb(d / q["img_A"]), cv2.imread(str(d / q["sem_A"]), cv2.IMREAD_UNCHANGED)
    Bv, semB = read_rgb(d / q["img_B"]), cv2.imread(str(d / q["sem_B"]), cv2.IMREAD_UNCHANGED)
    a_out = jpeg(outlined(A, semA, q["sid"]), 320)
    a_plain = jpeg(A, 320)
    b_out = jpeg(outlined(Bv, semB, {"X": q["sid"]["X"], "Y": q["sid"]["Y"]}), 320)
    b_plain = jpeg(Bv, 320)
    flip = "B across the line: answer flips" if q["flip"] else "B on A's side: same answer"
    side = {"left": "left of X and Y", "right": "right of X and Y"}.get(q["image_side"], q["image_side"])
    prompt = PROMPT.format(B=q["B"], X=q["X"], Y=q["Y"])
    return f"""
<article class="card">
  <header class="card-h">
    <span class="tag {'tag-flip' if q['flip'] else 'tag-same'}">{esc(flip)}</span>
    <span class="meta">{esc(q['split'])} · {esc(q['room'])} · B is the {esc(q['depth_rank'])} of the three from A · B appears {esc(side)} in the photo</span>
  </header>
  <div class="trio">
    <figure><img class="sw" data-plain="{a_plain}" data-outline="{a_out}" src="{a_out}" alt="A's photo">
      <figcaption>A's photo (input) · from A, <b>X is {esc(q['relA'])} of Y</b></figcaption></figure>
    <figure><img class="sw" data-plain="{b_plain}" data-outline="{b_out}" src="{b_out}" alt="View from B's spot">
      <figcaption>From B's spot (hidden from the model) · <b>X is {esc(q['relB'])} of Y</b></figcaption></figure>
    <figure>{sketch(q)}<figcaption>Top-down sketch</figcaption></figure>
  </div>
  <div class="roles"><span class="role role-B">B = {esc(q['B'])}</span><span class="role role-X">X = {esc(q['X'])}</span><span class="role role-Y">Y = {esc(q['Y'])}</span></div>
  <p class="prompt">“{esc(prompt)}”</p>
  <dl class="facts">
    <div><dt>A's answer</dt><dd>{esc(q['relA'])}</dd></div>
    <div><dt>B's answer</dt><dd>{esc(q['relB'])}</dd></div>
    <div><dt>A to B</dt><dd>{q['dist']['AB']:.1f} m</dd></div>
    <div><dt>B to X–Y line</dt><dd>{q['dist']['B_line']:.1f} m</dd></div>
    <div><dt>X–Y angle from A</dt><dd>{q['angle_A']:.0f}°</dd></div>
    <div><dt>X–Y angle from B</dt><dd>{q['angle_B']:.0f}°</dd></div>
  </dl>
</article>"""


def gallery_item(q, set_name):
    d = LOCAL / set_name
    A, semA = read_rgb(d / q["img_A"]), cv2.imread(str(d / q["sem_A"]), cv2.IMREAD_UNCHANGED)
    out, plain = jpeg(outlined(A, semA, q["sid"]), 220, 78), jpeg(A, 220, 78)
    return (f'<figure class="g"><img class="sw" data-plain="{plain}" data-outline="{out}" src="{out}" alt="A photo">'
            f'<figcaption><span class="dot dB"></span>{esc(q["B"])} <span class="dot dX"></span>{esc(q["X"])} '
            f'<span class="dot dY"></span>{esc(q["Y"])}<br>{esc(q["room"])} · {"flips" if q["flip"] else "same"}'
            f'</figcaption></figure>')


def bookkeeping(summary):
    rows = []
    for split in ("train", "test"):
        s = summary["splits"].get(split)
        if not s:
            continue
        t = s["selected"]
        cells = ", ".join(f"{k}: {v}" for k, v in sorted(t["cells"].items()))
        rank = ", ".join(f"{k} {v['flip']:.0%} ({v['n']})" for k, v in t["by_rank"].items() if v["n"])
        side = ", ".join(f"{k} {v['flip']:.0%} ({v['n']})" if v["n"] else f"{k} – (0)" for k, v in t["by_side"].items())
        rows.append(f"<tr><td>{split}</td><td class='n'>{t['questions']}</td><td class='n'>{t['photos']}</td>"
                    f"<td class='n'>{t['rooms']}</td><td>{esc(', '.join(t['categories']))}</td><td>{esc(cells)}</td>"
                    f"<td>{esc(rank)}</td><td>{esc(side)}</td></tr>")
    return ("<div class='tw'><table><thead><tr><th>split</th><th>questions</th><th>photos</th><th>rooms</th>"
            "<th>categories</th><th>A's answer / B's answer</th><th>flip rate by B depth rank</th>"
            "<th>flip rate by B image side</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


THEIRS_SHORTCUT = {"all": 0.51, "rank": 0.50, "side": 0.51, "distances+angles": 0.50, "categories": 0.50, "rank+side": 0.50}


def shortcut_table(summary):
    sc = summary.get("shortcut")
    if not sc:
        return "<p>Shortcut check needs both splits.</p>"
    rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='n'>{THEIRS_SHORTCUT[k]:.0%}</td><td class='n'>{v['relB_within_A']:.0%}</td>"
        f"<td class='n add'>{v['flip']:.0%}</td></tr>" for k, v in sc.items())
    return ("<div class='tw'><table><thead><tr><th>features</th><th>PhD: B's answer, within A's</th>"
            "<th>ours: B's answer, within A's</th><th class='add'>added: flips?</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


REASONS = {  # sampler counter -> (chart, plain label)
    "A: distance": ("A", "not 1.5–7 m from all three objects"),
    "A: view blocked by wall": ("A", "a wall blocks one of the objects"),
    "A: too small": ("A", "an object covers < 0.3% of the photo"),
    "A: cut off": ("A", "an object is cut off at the edge"),
    "A: overlap": ("A", "two objects overlap left to right"),
    "A: inside an object": ("A", "spot is inside an object"),
    "A: object hidden behind another": ("A", "an object is >10% hidden behind another"),
    "layout: objects do not fit": ("Q", "layout: objects do not fit in the room"),
    "A: X and Y < 10 deg apart": ("Q", "X and Y < 10° apart from A"),
    "B < 0.5 m from the X-Y line": ("Q", "B < 0.5 m from the X–Y line"),
    "B < 0.8 m from X or Y": ("Q", "B < 0.8 m from X or Y"),
    "B: view blocked by wall": ("Q", "B's spot: a wall blocks X or Y"),
    "B: too small": ("Q", "B's spot: X or Y < 0.3% of the photo"),
    "B: overlap": ("Q", "B's spot: X and Y overlap left to right"),
    "B: X and Y < 15 deg apart": ("Q", "X and Y < 15° apart from B's spot"),
    "mask label != geometry label": ("Q", "mask label disagrees with geometry"),
    "flip != side of line": ("Q", "flip disagrees with side of line"),
}


def bars(items, title):
    vmax = max(max(v.values()) for _, v in items) or 1
    W, LW, row = 720, 290, 28
    H = row * len(items) + 34
    el = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = LW + t * (W - LW - 70)
        el.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H - 24}" class="grid"/>'
                  f'<text x="{x:.1f}" y="{H - 8}" class="ax" text-anchor="middle">{int(round(t * vmax)):,}</text>')
    for i, (k, v) in enumerate(items):
        y = i * row + 4
        el.append(f'<text x="{LW - 8}" y="{y + 15}" class="lab" text-anchor="end">{esc(k)}</text>')
        for j, split in enumerate(("train", "test")):
            n = v.get(split, 0)
            w = (W - LW - 70) * n / vmax
            el.append(f'<rect x="{LW}" y="{y + j * 11}" width="{max(w, 0.5):.1f}" height="9" rx="2" class="bar bar-{split}">'
                      f'<title>{esc(k)} · {split} rooms: {n:,}</title></rect>'
                      f'<text x="{LW + w + 4:.1f}" y="{y + j * 11 + 8.5}" class="val">{n:,}</text>')
    el.append("</svg>")
    return f'<h3>{esc(title)}</h3>' + "".join(el)


def funnel_chart(summary):
    """Why camera spots for A, and then questions, were rejected, per split."""
    import re
    groups = {"A": {}, "Q": {}}
    for split in ("train", "test"):
        s = summary["splits"].get(split)
        if not s:
            continue
        for k, v in s["funnel"].items():
            k = re.sub(r": \d ", ": ", k)
            if k not in REASONS:
                continue
            chart, label = REASONS[k]
            groups[chart].setdefault(label, {})
            groups[chart][label][split] = groups[chart][label].get(split, 0) + v
    legend = ('<div class="legend"><span><i class="sw-train"></i>train rooms</span>'
              '<span><i class="sw-test"></i>test rooms</span></div>')
    out = [legend]
    for chart, title in (("A", "Camera spots for A rejected"), ("Q", "Layouts and questions rejected")):
        items = sorted(groups[chart].items(), key=lambda kv: -sum(kv[1].values()))
        if items:
            out.append(bars(items, title))
    return "".join(out)


def pool_section(assets, set_counts):
    rows = []
    for split, cats in (("train", ["bench", "cabinet", "chair", "couch", "dresser", "potted plant", "stool", "table"]),
                        ("test", ["bed", "bookshelf", "office chair", "refrigerator", "trash can"])):
        for cat in cats:
            tiles = []
            for a in [a for a in assets if a["category"] == cat]:
                p = LOCAL / "thumbs" / f"{a['id']}.png"
                if p.exists():
                    tiles.append(f'<img src="{jpeg(read_rgb(p), 120, 80)}" alt="{esc(cat)}" title="{esc(a["hssd_name"])}">')
            n = set_counts.get(cat, 0)
            rows.append(f'<div class="pool-row"><div class="pool-name"><b>{esc(cat)}</b><span>{split} · '
                        f'{n:,} mentions</span></div><div class="pool-tiles">{"".join(tiles)}</div></div>')
    return "".join(rows)


CSS = """
:root{
  /* layout: one reading column (max 1180px); cards stack; tables scroll inside their own box */
  --bg:#f6f7f9; --panel:#ffffff; --ink:#15181d; --ink2:#4a5260; --ink3:#6f7785; --rule:#dde1e7;
  --accent:#2a6fdb; --ok:#1d7a46; --okbg:#e3f3e9; --bad:#b42318; --badbg:#fdecea;
  --add:#6250d6; --addbg:#efedfc; --chg:#9a6700; --chgbg:#fdf3dc;
  --B:#2ea043; --X:#eb7a1e; --Y:#286ee6; --train:#2a78d6; --test:#eb6834;
  --sk-bg:#fbfbfa; --sk-grid:#e7e9ec;
  --f-display:"Fraunces",Georgia,serif; --f-body:"Public Sans",system-ui,sans-serif; --f-mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#121417; --panel:#1b1e23; --ink:#eceef1; --ink2:#b9c0ca; --ink3:#8d95a1; --rule:#2e333b;
  --accent:#6ea0f0; --ok:#5cc98a; --okbg:#173325; --bad:#f2837a; --badbg:#3a1d1b;
  --add:#a59bf0; --addbg:#26224a; --chg:#e0b048; --chgbg:#3a2f12; --train:#3987e5; --test:#d95926;
  --sk-bg:#22262c; --sk-grid:#30353d; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121417; --panel:#1b1e23; --ink:#eceef1; --ink2:#b9c0ca; --ink3:#8d95a1; --rule:#2e333b;
  --accent:#6ea0f0; --ok:#5cc98a; --okbg:#173325; --bad:#f2837a; --badbg:#3a1d1b;
  --add:#a59bf0; --addbg:#26224a; --chg:#e0b048; --chgbg:#3a2f12; --train:#3987e5; --test:#d95926;
  --sk-bg:#22262c; --sk-grid:#30353d; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:36px}
h1{font-family:var(--f-display);font-weight:600;font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0;text-wrap:balance}
h2{font-family:var(--f-display);font-weight:600;font-size:24px;margin:0 0 6px;text-wrap:balance}
.lede{max-width:72ch;color:var(--ink2);margin:8px 0 0}
.kicker{font:500 12px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
section>p{max-width:72ch;color:var(--ink2);margin:4px 0 14px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:18px}
.stat{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.stat b{display:block;font:600 24px/1.1 var(--f-mono);font-variant-numeric:tabular-nums}
.stat span{color:var(--ink3);font-size:13px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.controls button{font:500 13px var(--f-body);border:1px solid var(--rule);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:999px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.controls button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.cards{display:grid;gap:18px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px;display:grid;gap:12px;min-width:0}
.card-h{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:baseline}
.tag{font:600 12px var(--f-mono);padding:3px 8px;border-radius:4px}
.tag-flip{background:var(--chgbg);color:var(--chg)} .tag-same{background:var(--okbg);color:var(--ok)}
.meta{color:var(--ink3);font-size:13px}
.trio{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
@media (max-width:760px){.trio{grid-template-columns:minmax(0,1fr)}}
figure{margin:0;display:grid;gap:6px;min-width:0}
figure img,.sketch{width:100%;aspect-ratio:1;border-radius:6px;display:block}
figcaption{font-size:13px;color:var(--ink2)}
.roles{display:flex;gap:8px;flex-wrap:wrap}
.role{font:500 13px var(--f-mono);padding:2px 8px;border-radius:4px;border:2px solid}
.role-B{border-color:var(--B)} .role-X{border-color:var(--X)} .role-Y{border-color:var(--Y)}
.prompt{margin:0;font-size:14px;color:var(--ink2);font-style:italic;max-width:95ch}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:0}
.facts div{border-top:1px solid var(--rule);padding-top:6px}
.facts dt{font-size:12px;color:var(--ink3)} .facts dd{margin:0;font:500 15px var(--f-mono);font-variant-numeric:tabular-nums}
.sk-bg{fill:var(--sk-bg)} .sk-grid{stroke:var(--sk-grid);stroke-width:1}
.sk-line{stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:5 4;fill:none}
.sk-cone{stroke:none;opacity:.18} .sk-cone-A{fill:var(--ink)} .sk-cone-B{fill:var(--B)}
.sk-obj{stroke-width:2;fill-opacity:.25} .sk-B{stroke:var(--B);fill:var(--B)} .sk-X{stroke:var(--X);fill:var(--X)} .sk-Y{stroke:var(--Y);fill:var(--Y)}
.sk-A{fill:var(--ink)} .sk-lab{font:600 12px var(--f-mono);text-anchor:middle;fill:var(--ink)} .sk-lab-A{fill:var(--bg)}
.sk-note{font:11px var(--f-body);fill:var(--ink3)}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:12.5px;white-space:nowrap}
tr:last-child td{border-bottom:0}
td.n{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right}
th.add,td.add{color:var(--add)}
.chip{font:600 12px var(--f-mono);padding:2px 8px;border-radius:999px;white-space:nowrap}
.chip.ok{background:var(--okbg);color:var(--ok)} .chip.bad{background:var(--badbg);color:var(--bad)}
.pool-row{display:grid;grid-template-columns:160px minmax(0,1fr);gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--rule)}
@media (max-width:560px){.pool-row{grid-template-columns:minmax(0,1fr)}}
.pool-name span{display:block;font-size:12.5px;color:var(--ink3)}
.pool-tiles{display:flex;gap:6px;flex-wrap:wrap} .pool-tiles img{width:96px;height:96px;border-radius:4px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.g figcaption{font-size:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 3px 0 4px} .dB{background:var(--B)} .dX{background:var(--X)} .dY{background:var(--Y)}
.chart{width:100%;max-width:760px;height:auto;display:block}
.chart .grid{stroke:var(--rule)} .chart .ax,.chart .val{font:11px var(--f-mono);fill:var(--ink3)} .chart .lab{font:12.5px var(--f-body);fill:var(--ink2)}
.bar-train{fill:var(--train)} .bar-test{fill:var(--test)}
h3{font:600 14px var(--f-body);margin:18px 0 6px;color:var(--ink2)}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin-bottom:6px}
.legend i{display:inline-block;width:12px;height:8px;border-radius:2px;margin-right:6px;vertical-align:middle}
.sw-train{background:var(--train)} .sw-test{background:var(--test)}
.diff{display:grid;gap:8px;padding:0;margin:0;list-style:none}
.diff li{padding:8px 12px;border-radius:6px;max-width:100ch}
.diff .a{background:var(--addbg)} .diff .c{background:var(--chgbg)} .diff .t{background:var(--panel);border:1px solid var(--rule)}
.diff b{font:600 12px var(--f-mono);margin-right:8px}
.diff .a b{color:var(--add)} .diff .c b{color:var(--chg)} .diff .t b{color:var(--ink3)}
@media (prefers-reduced-motion:no-preference){.sw{transition:opacity .15s}}
"""

JS = """
const btns=document.querySelectorAll('[data-mode]');
btns.forEach(b=>b.addEventListener('click',()=>{
  const m=b.dataset.mode;
  btns.forEach(x=>x.setAttribute('aria-pressed',String(x===b)));
  document.querySelectorAll('img.sw').forEach(i=>{i.src=i.dataset[m]});
}));
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", default="train")
    ap.add_argument("--test-set", default="test")
    ap.add_argument("--cards", type=int, default=6, help="example cards per split")
    ap.add_argument("--gallery", type=int, default=12, help="extra photos per split")
    ap.add_argument("--list", action="store_true", help="print the image files needed, relative to $PP_DATA/v4")
    ap.add_argument("--out", default=str(LOCAL / "v4_data_report.html"))
    args = ap.parse_args()

    picks = {}
    for split, set_name in (("train", args.train_set), ("test", args.test_set)):
        qs = load_questions(set_name, split)
        c = pick_cards(qs, args.cards, 7 if split == "test" else 3)
        g = pick_gallery(qs, args.gallery, 7 if split == "test" else 3, {q["photo"] for q in c})
        picks[split] = (set_name, qs, c, g)
    if args.list:
        files = set()
        for set_name, _, c, g in picks.values():
            files |= set(needed_files(set_name, c, g))
        print("\n".join(sorted(files)))
        return

    sfx = "" if args.train_set == "train" else "_" + args.train_set
    summary = json.loads((LOCAL / f"v4_data_summary{sfx}.json").read_text())
    assets = json.loads((LOCAL / "assets.json").read_text())
    mentions = {}
    for split in ("train", "test"):
        for k, v in summary["splits"].get(split, {}).get("selected", {}).get("category_mentions", {}).items():
            mentions[k] = mentions.get(k, 0) + v
    tr, te = summary["splits"].get("train", {}).get("selected", {}), summary["splits"].get("test", {}).get("selected", {})
    checks = summary["checks"]
    n_pass = sum(c["pass"] for c in checks)

    check_rows = "".join(f"<tr><td>{esc(c['check'])}</td><td>{esc(c['value'])}</td><td>{chip(c['pass'])}</td></tr>"
                         for c in checks)
    cards = "".join(card(q, set_name) for split in ("test", "train") for q in picks[split][2]
                    for set_name in [picks[split][0]])
    gallery = "".join(gallery_item(q, picks[split][0]) for split in ("test", "train") for q in picks[split][3])
    smoke = " (smoke test)" if args.train_set != "train" else ""

    page = f"""<title>v4 Scene Inspection</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="kicker">Perspective probe replication · v4 data{esc(smoke)}</div>
  <h1>Three random objects in an emptied room</h1>
  <p class="lede">Phases 0–2 of PLAN_V4.md: the HSSD object pool, the rendered scenes, the balanced question set and the data checks, before any model sees them.
  Each question names one object as B and asks whether X is left or right of Y from B's spot. B's answer equals A's when B is on A's side of the line through X and Y, and flips when B is across it.</p>
  <div class="stats">
    <div class="stat"><b>{tr.get('questions', 0):,}</b><span>train questions · {tr.get('photos', 0):,} photos · {tr.get('rooms', 0)} rooms</span></div>
    <div class="stat"><b>{te.get('questions', 0):,}</b><span>test questions · {te.get('photos', 0):,} photos · {te.get('rooms', 0)} rooms</span></div>
    <div class="stat"><b>{len(assets)}</b><span>HSSD models in 13 categories</span></div>
    <div class="stat"><b>{n_pass}/{len(checks)}</b><span>data checks pass</span></div>
  </div>
</header>

<section>
  <h2>Example scenes</h2>
  <p>Each card: A's photo (the model's input), the view from B's spot with B hidden (used for the label and the direct-B question), a top-down sketch, the exact question and the facts. Outlines are drawn for this page only: the model gets the plain photo.</p>
  <div class="controls" role="group" aria-label="Photo style"><button data-mode="outline" aria-pressed="true">Outlines (B green, X orange, Y blue)</button><button data-mode="plain" aria-pressed="false">Plain photo (what the model sees)</button></div>
  <div class="cards" style="margin-top:14px">{cards}</div>
</section>

<section>
  <h2>More photos</h2>
  <p>Further A photos drawn at random from the selected questions, test rooms first.</p>
  <div class="gallery">{gallery}</div>
</section>

<section>
  <h2>Bookkeeping</h2>
  <p>The selected questions per split, in the same layout as the PhD's table. Flip rates should be 50% in every group; "between" is empty by design.</p>
  {bookkeeping(summary)}
</section>

<section>
  <h2>Data checks</h2>
  <p>Balance checks and the metadata shortcut check: a logistic regression from bookkeeping facts (depth rank, image side, distances and angles, categories) trained on the train split and scored on the test split. The PhD's version predicts B's answer and scores it within A's answer. The added column predicts the flip directly.</p>
  <div class="tw"><table><thead><tr><th>check</th><th>value</th><th>result</th></tr></thead><tbody>{check_rows}</tbody></table></div>
  <div style="height:14px"></div>
  {shortcut_table(summary)}
</section>

<section>
  <h2>Why candidates were dropped</h2>
  <p>Counts from the sampler, summed over rooms. The first chart counts random camera spots tried for A and turned down; the second counts whole layouts and single questions. Each A photo offers three questions, one per choice of B. Balancing then keeps equal numbers of the four A/B answer combinations in each depth-rank × image-side group.</p>
  {funnel_chart(summary)}
</section>

<section>
  <h2>Object pool</h2>
  <p>{len(assets)} HSSD models kept after a visual check (5 per category; 4 beds). Train and test use different categories. Mentions count how often a category appears as B, X or Y in the selected questions.</p>
  {pool_section(assets, mentions)}
</section>

<section>
  <h2>Differences from the PhD's v4</h2>
  <ul class="diff">
    <li class="t"><b>THEIRS</b>13 categories and their split; 34 rooms (25 train, 9 test); A 1.5–7 m from the objects looking at their centroid; X–Y ≥ 10° from A and ≥ 15° from B; B ≥ 0.5 m from the X–Y line and ≥ 0.8 m from X and Y; mask labels checked against geometry; grey background with white floor grid; balanced 50% flips; shortcut check ≤ 60%.</li>
    <li class="c"><b>CHANGED</b>A's photo: no object may be cut off by the image edge or more than 10% hidden behind another, and the X and Y masks may not overlap left to right. Their examples show cut-off objects.</li>
    <li class="c"><b>CHANGED</b>"B between X and Y in the picture" is excluded. Their table has none, so we assume they did the same.</li>
    <li class="a"><b>ADDED</b>Shortcut check also run on the flip itself, which the PhD's version (B's answer without A's answer as an input) cannot see directly.</li>
    <li class="t"><b>OURS</b>Which rooms; which HSSD models (5 per category, chosen by eye); camera 512×512, 90° field of view, 1.5 m eye height, no tilt; grid every 1 m; objects ≥ 1 m apart and randomly turned; at most 2 A photos per layout; each object visible on at least 0.3% of the frame.</li>
  </ul>
</section>
</div>
<script>{JS}</script>
"""
    Path(args.out).write_text(page)
    print(f"-> {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
