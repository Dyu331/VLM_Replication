"""Self-contained HTML gallery of rendered examples, for visual checks before the eval.

Per example: A and B photos with X (orange) / Y (blue) outlined from the instance masks, a
top-down map (A, B, view wedges, X-Y line, furniture from the scene index), labels, the
clean-rule verdict, and the exact prompt each setting sends (built by eval/prompts.py).

    python repro/report/build_gallery.py --set test --n 12 --n-rejected 4
    -> $PP_RESULTS/gallery/<set>_gallery.html   (copy to your laptop and open)
"""
import argparse
import base64
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import HFOV, PP_DATA, PP_RESULTS
from eval.prompts import SETTINGS, build

X_BGR, Y_BGR = (40, 120, 255), (255, 150, 40)  # orange, blue
SETTING_INFO = {
    "s1": ("1 Original", "A's photo + sentence about B", "B"),
    "s1_identity": ("1 Identity", "A's photo + 'B stands where you stand'", "A"),
    "s2": ("2 Ground-truth target", "B's photo, direct question", "B"),
    "s4": ("4 Control (own view)", "A's photo, direct question", "A"),
}


# ---------- images ----------

def find_sid(sem, stats):
    """The mask id whose pixel stats match the saved stats for one object."""
    best, best_err = None, float("inf")
    for sid in np.unique(sem):
        if sid == 0:
            continue
        m = sem == sid
        xs = np.nonzero(m)[1]
        err = abs(100.0 * m.sum() / m.size - stats["vis_pct"]) + abs(xs.mean() - stats["cx"]) / 100
        if err < best_err:
            best, best_err = sid, err
    return best


def outlined(rgb_path, sem_path, view):
    img = cv2.imread(str(rgb_path))
    sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
    k = np.ones((5, 5), np.uint8)
    for key, color in (("X", X_BGR), ("Y", Y_BGR)):
        sid = find_sid(sem, view[key])
        if sid is None:
            continue
        m = (sem == sid).astype(np.uint8)
        edge = (cv2.dilate(m, k) > 0) & ~(cv2.erode(m, k) > 0)
        img[edge] = color
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ---------- top-down map (x right, z down: looking down from above) ----------

def map_svg(rec, scene_objs, size=300):
    pts = [rec["A"]["pos"], rec["B"]["pos"], rec["X_center"], rec["Y_center"]]
    xs = [p[0] for p in pts] + [v for o in scene_objs for v in (o[0], o[1])]
    zs = [p[2] for p in pts] + [v for o in scene_objs for v in (o[2], o[3])]
    x0, x1, z0, z1 = min(xs) - 0.5, max(xs) + 0.5, min(zs) - 0.5, max(zs) + 0.5
    s = size / max(x1 - x0, z1 - z0)
    w, h = (x1 - x0) * s, (z1 - z0) * s
    xy = lambda x, z: ((x - x0) * s, (z - z0) * s)
    P = lambda x, z: "{:.1f},{:.1f}".format(*xy(x, z))
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" class="map" role="img" aria-label="top-down map">']
    for ox0, ox1, oz0, oz1 in scene_objs:
        out.append(f'<rect x="{(ox0 - x0) * s:.1f}" y="{(oz0 - z0) * s:.1f}" width="{(ox1 - ox0) * s:.1f}" '
                   f'height="{(oz1 - oz0) * s:.1f}" class="furn"/>')
    xc, yc = rec["X_center"], rec["Y_center"]
    (lx1, ly1), (lx2, ly2) = xy(xc[0], xc[2]), xy(yc[0], yc[2])
    out.append(f'<line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" class="xyline"/>')
    for view, cls in (("A", "obsA"), ("B", "obsB")):
        p, yaw = rec[view]["pos"], rec[view]["yaw"]
        wedge = [P(p[0], p[2])]
        for d in (-HFOV / 2, HFOV / 2):
            r = math.radians(yaw + d)
            wedge.append(P(p[0] - 1.2 * math.sin(r), p[2] - 1.2 * math.cos(r)))
        out.append(f'<polygon points="{" ".join(wedge)}" class="{cls} wedge"/>')
        cx, cy = xy(p[0], p[2])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" class="{cls}"/>'
                   f'<text x="{cx + 7:.1f}" y="{cy - 7:.1f}" class="lbl">{view}</text>')
    for c, cls in ((xc, "objX"), (yc, "objY")):
        cx, cy = xy(c[0], c[2])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="{cls}"/>')
    out.append("</svg>")
    return "".join(out)


def load_furniture(scene):
    path = PP_DATA / "scene_index" / f"{scene}.json"
    if not path.exists():
        return []
    objs = json.loads(path.read_text())["objects"]
    return [(o["aabb_min"][0], o["aabb_max"][0], o["aabb_min"][2], o["aabb_max"][2])
            for o in objs if o["size"][1] >= 0.25 and max(o["size"][0], o["size"][2]) < 5]


# ---------- page ----------

def pick(recs, n, rng):
    """Round-robin over rooms so the sample covers many scenes."""
    by_scene = defaultdict(list)
    for r in recs:
        by_scene[r["scene"]].append(r)
    for v in by_scene.values():
        rng.shuffle(v)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    out = []
    while len(out) < n and any(by_scene.values()):
        for sc in scenes:
            if by_scene[sc] and len(out) < n:
                out.append(by_scene[sc].pop())
    return out


def card(rec, set_name):
    d = PP_DATA / set_name / rec["scene"]
    imgs = {v: outlined(d / f"{rec['id']}_{v}_rgb.png", d / f"{rec['id']}_{v}_sem.png", rec[v])
            for v in "AB"}
    e = html.escape
    status = ('<span class="ok">clean</span>' if rec["clean"] else
              '<span class="bad">rejected</span>: ' + e("; ".join(rec["reject_reasons"])))
    stats = lambda v: (f'X {rec[v]["X"]["vis_pct"]:.1f}% · {rec[v]["X"]["dist"]:.1f} m &nbsp; '
                       f'Y {rec[v]["Y"]["vis_pct"]:.1f}% · {rec[v]["Y"]["dist"]:.1f} m')
    rows = []
    for s in SETTINGS:
        view, text, truth_view = build(rec, s)
        name, desc, _ = SETTING_INFO[s]
        rows.append(f'<tr><td><b>{e(name)}</b><br><span class="muted">{e(desc)}</span></td>'
                    f'<td>{view}\'s photo</td><td><pre>{e(text)}</pre></td>'
                    f'<td><b>{rec["rel" + truth_view]}</b> <span class="muted">(from {truth_view})</span></td></tr>')
    return f"""
<section class="card">
  <h3>{e(rec['id'])}</h3>
  <p class="muted">{e(rec['scene'])} · {status}</p>
  <p>Question objects: <span class="x">X = {e(rec['X'])}</span>, <span class="y">Y = {e(rec['Y'])}</span>.
     Mask answer: from A <b>{rec['relA']}</b>, from B <b>{rec['relB']}</b>
     <span class="muted">(3D-centre labels: {rec['relA_centroid']} / {rec['relB_centroid']})</span></p>
  <div class="views">
    <figure><img src="{imgs['A']}" alt="A's photo"><figcaption>A's photo · {stats('A')}</figcaption></figure>
    <figure><img src="{imgs['B']}" alt="B's photo"><figcaption>B's photo · {stats('B')}</figcaption></figure>
    <figure>{map_svg(rec, load_furniture(rec['scene']))}<figcaption>Map from above:
      <span class="a">A</span>, <span class="b">B</span>, wedges = view direction ({HFOV:.0f}°),
      dashed = X–Y line, grey = furniture</figcaption></figure>
  </div>
  <p><b>Sentence about B:</b> {e(rec['sentence_B'])}</p>
  <div class="scroll"><table>
    <tr><th>Setting</th><th>Image</th><th>Prompt sent to the model</th><th>Correct</th></tr>
    {''.join(rows)}
  </table></div>
</section>"""


CSS = """
:root{--bg:#fafaf7;--panel:#fff;--fg:#1d2228;--muted:#69727d;--line:#dde1e6;--x:#e8641e;--y:#2b8be6;
--a:#2b6fd6;--b:#d63b3b;--ok:#1f8a4c;--bad:#c0392b;--furn:#cfd4da}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181d;--panel:#1c2229;--fg:#e6e9ee;
--muted:#9aa5b4;--line:#2e3845;--furn:#3a4450;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14181d;--panel:#1c2229;--fg:#e6e9ee;--muted:#9aa5b4;--line:#2e3845;--furn:#3a4450;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0}
main{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:1.6rem;margin:.2em 0}.muted{color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:18px 0}
.card h3{margin:0;font-size:1rem;font-family:ui-monospace,monospace}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
figure{margin:0}figure img,.map{width:100%;height:auto;border-radius:6px;border:1px solid var(--line);display:block}
figcaption{font-size:.82rem;color:var(--muted);margin-top:4px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{border-top:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;margin:0;font-size:.82rem;font-family:ui-monospace,monospace}
.x{color:var(--x);font-weight:600}.y{color:var(--y);font-weight:600}.a{color:var(--a);font-weight:600}
.b{color:var(--b);font-weight:600}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.furn{fill:var(--furn)}.xyline{stroke:var(--muted);stroke-dasharray:4 3;stroke-width:1.5}
.obsA{fill:var(--a)}.obsB{fill:var(--b)}.wedge{opacity:.22}.objX{fill:var(--x)}.objY{fill:var(--y)}
.lbl{fill:var(--fg);font-size:12px;font-weight:700}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--n", type=int, default=12, help="clean examples to show")
    ap.add_argument("--n-rejected", type=int, default=4, help="rejected examples to show")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    recs = [json.loads(l) for l in open(PP_DATA / args.set / "examples_labeled.jsonl")]
    clean = [r for r in recs if r["clean"]]
    shown = pick(clean, args.n, rng) + pick([r for r in recs if not r["clean"]], args.n_rejected, rng)
    legend = "".join(f"<tr><td><b>{n}</b></td><td>{d}</td><td>answer from {t}</td></tr>"
                     for n, d, t in SETTING_INFO.values())
    page = f"""<title>Perspective gallery: {args.set}</title><style>{CSS}</style><main>
<h1>Perspective replication: {html.escape(args.set)} set</h1>
<p class="muted">{len(recs)} cross_line examples rendered in {len({r['scene'] for r in recs})} rooms,
{len(clean)} clean. Showing {len(shown) - args.n_rejected} clean and up to {args.n_rejected} rejected,
sampled across rooms. In every photo <span class="x">X is outlined orange</span> and
<span class="y">Y blue</span> (from the instance masks); the model sees the photo without outlines.</p>
<div class="scroll"><table><tr><th>Setting</th><th>What the model gets</th><th>Scored against</th></tr>{legend}</table></div>
{''.join(card(r, args.set) for r in shown)}
</main>"""
    out = PP_RESULTS / "gallery" / f"{args.set}_gallery.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{len(shown)} examples -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
