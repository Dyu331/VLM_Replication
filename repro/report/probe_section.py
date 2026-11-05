"""Section 3 of the replication report ("Inside the model"), from the probe analyses.

Reads local_data/results/probe/<model>/analysis.json and test/s1.jsonl (pulled from Turbo), and the
report's own per-layer numbers from repro/report/report_probe.json (extract_report_probe.py).
"""
import html
import itertools
import json
from pathlib import Path

e = html.escape
_ids = itertools.count()
CHECK2_MIN = 0.80
# The report's "What we found" figures, quoted from its text
REPORT_FOUND = {
    "qwen3vl": dict(check1=0.85, ceiling=0.99, bar=0.94, id_check1=0.96, id_pass=True, trusted="20–36", n_trusted=17,
                    b_last=0.23, b_ci=(0.15, 0.33), id_b_last=0.17, holds_a=16, near=1, holds_b=0),
    "bagel": dict(check1=0.71, ceiling=0.97, bar=0.92, id_check1=0.89, id_pass=False, trusted="20–28", n_trusted=9,
                  b_last=0.48, b_ci=(0.37, 0.58), id_b_last=0.19, holds_a=4, near=5, holds_b=0),
}


def p0(x):
    return "–" if x is None or x != x else f"{100 * x:.0f}%"


def runs(layers):
    """[17, 22, 23, 24] -> '17, 22–24'"""
    out, start = [], None
    for i, L in enumerate(layers):
        if start is None:
            start = L
        if i == len(layers) - 1 or layers[i + 1] != L + 1:
            out.append(f"{start}" if start == L else f"{start}–{L}")
            start = None
    return ", ".join(out) or "none"


def span(vals):
    vals = [v for v in vals if v == v]
    return f"{min(vals):.0%}–{max(vals):.0%}" if vals else "–"


# ---------------------------------------------------------------- charts
W, H, L0, R0, T0, B0 = 560, 250, 44, 14, 14, 40   # viewBox and plot margins


def frame(n, ylabel):
    x = lambda i: L0 + (W - L0 - R0) * i / (n - 1)
    y = lambda v: T0 + (H - T0 - B0) * (1 - v)
    out = []
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{L0}" x2="{W - R0}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"base" if v == 0 else "grid"}"/>'
                   f'<text x="{L0 - 6}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0%}</text>')
    step = 4 if n > 30 else 3
    for i in range(0, n, step):
        out.append(f'<text x="{x(i):.1f}" y="{H - B0 + 16}" class="tick" text-anchor="middle">{i}</text>')
    out.append(f'<text x="{(L0 + W - R0) / 2:.0f}" y="{H - 6}" class="tick" text-anchor="middle">layer</text>'
               f'<text transform="translate(11 {(T0 + H - B0) / 2:.0f}) rotate(-90)" class="tick" text-anchor="middle">{ylabel}</text>')
    return x, y, out


def path(xs, ys, vals):
    pts = [(xs(i), ys(v)) for i, v in enumerate(vals) if v is not None and v == v]
    return "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts)


def band(xs, ys, lo, hi):
    top = [f"{xs(i):.1f} {ys(v):.1f}" for i, v in enumerate(hi)]
    bot = [f"{xs(i):.1f} {ys(v):.1f}" for i, v in reversed(list(enumerate(lo)))]
    return "M" + " L".join(top + bot) + " Z"


def hover(xs, n, tips):
    """One transparent column per layer, wider than the marks, carrying the tooltip."""
    w = (W - L0 - R0) / (n - 1)
    return "".join(f'<rect x="{xs(i) - w / 2:.1f}" y="{T0}" width="{w:.1f}" height="{H - T0 - B0}" class="hit">'
                   f"<title>{e(t)}</title></rect>" for i, t in enumerate(tips))


def trust_chart(label, layers):
    n = len(layers)
    xs, ys, out = frame(n, "agreement (class-balanced)")
    d = [l["direct"] for l in layers]
    out.append(f'<path d="{path(xs, ys, [0.95 * v["ceiling"] for v in d])}" class="bar1"/>'
               f'<line x1="{L0}" x2="{W - R0}" y1="{ys(CHECK2_MIN):.1f}" y2="{ys(CHECK2_MIN):.1f}" class="bar2"/>')
    for key, cls in (("ceiling", "s1"), ("check1", "s2"), ("check2", "s3")):
        out.append(f'<path d="{path(xs, ys, [v[key] for v in d])}" class="ln {cls}"/>')
    tips = [f"layer {i}\nCeiling: {p0(v['ceiling'])}\nCheck 1: {p0(v['check1'])}\nCheck 2: {p0(v['check2'])}"
            for i, v in enumerate(d)]
    out.append(hover(xs, n, tips))
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{e(label)}: can the reader be trusted at this layer?">'
            + "".join(out) + "</svg>")


def bscore_chart(label, layers, report):
    n = len(layers)
    hid = f"hatch{next(_ids)}"
    xs, ys, out = frame(n, "B score")
    d = [l["direct"] for l in layers]
    w = (W - L0 - R0) / (n - 1)
    untrusted = [i for i, v in enumerate(d) if v["check2"] < CHECK2_MIN]
    for r in runs(untrusted).split(", "):
        if r == "none":
            break
        a, b = (int(r), int(r)) if "–" not in r else map(int, r.split("–"))
        x0, x1 = max(L0, xs(a) - w / 2), min(W - R0, xs(b) + w / 2)
        out.append(f'<rect x="{x0:.1f}" y="{T0}" width="{x1 - x0:.1f}" height="{H - T0 - B0}" class="untr-zone" style="fill: url(#{hid})"/>')
    out.insert(0, f'<defs><pattern id="{hid}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                  f'<line x1="0" y1="0" x2="0" y2="7" class="hatch"/></pattern></defs>')
    out.append(f'<path d="{band(xs, ys, [v["chance"][0] for v in d], [v["chance"][1] for v in d])}" class="chance"/>'
               f'<path d="{band(xs, ys, [v["b_ci"][0] for v in d], [v["b_ci"][1] for v in d])}" class="ci"/>'
               f'<line x1="{L0}" x2="{W - R0}" y1="{ys(0.5):.1f}" y2="{ys(0.5):.1f}" class="mid"/>'
               f'<text x="{L0 + 6}" y="{ys(1) + 13:.1f}" class="zone">↑ holds B’s answer</text>'
               f'<text x="{L0 + 6}" y="{ys(0) - 6:.1f}" class="zone">↓ holds A’s answer</text>')
    if report:
        out.append(f'<path d="{path(xs, ys, [r.get("b_score") for r in report])}" class="ln rep"/>')
    # whole curve faded; trusted layers drawn again at full strength
    trusted = [i for i in range(n) if i not in untrusted]
    for vals, cls in (([l["identity"]["b_score"] for l in layers], "s2"), ([v["b_score"] for v in d], "s1")):
        out.append(f'<path d="{path(xs, ys, vals)}" class="ln {cls} faint"/>')
        for r in runs(trusted).split(", "):
            if r == "none":
                break
            a, b = (int(r), int(r)) if "–" not in r else map(int, r.split("–"))
            if a == b:
                out.append(f'<circle cx="{xs(a):.1f}" cy="{ys(vals[a]):.1f}" r="3.5" class="pt {cls}"/>')
            else:
                out.append(f'<path d="{path(lambda k: xs(a + k), ys, vals[a:b + 1])}" class="ln {cls}"/>')
    tips = []
    for i, l in enumerate(layers):
        v = l["direct"]
        t = (f"layer {i}\nB score: {p0(v['b_score'])} (95% CI {p0(v['b_ci'][0])}–{p0(v['b_ci'][1])})\n"
             f"Identity-trained reader: {p0(l['identity']['b_score'])}\n"
             f"chance band: {p0(v['chance'][0])}–{p0(v['chance'][1])}\nreader says “left”: {p0(v['left_rate'])}")
        if report and i < len(report) and "b_score" in report[i]:
            t += f"\nreport: {p0(report[i]['b_score'])}"
        if v["check2"] < CHECK2_MIN:
            t += "\nreader not trusted here"
        tips.append(t)
    out.append(hover(xs, n, tips))
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{e(label)}: whose answer does the trusted reader see?">'
            + "".join(out) + "</svg>")


def key(items):
    return '<div class="key">' + "".join(f'<span><i class="{c}"></i>{e(t)}</span>' for c, t in items) + "</div>"


def layer_table(layers):
    rows = []
    for l in layers:
        d, i = l["direct"], l["identity"]
        cls = "" if d["check2"] >= CHECK2_MIN else ' class="untr"'
        rows.append(f'<tr{cls}><td class="num">{l["layer"]}</td><td class="num">{p0(d["ceiling"])}</td><td class="num">{p0(d["check1"])}</td>'
                    f'<td class="num">{p0(d["check2"])}</td><td class="num"><b>{p0(d["b_score"])}</b></td>'
                    f'<td class="num">{p0(d["b_ci"][0])}–{p0(d["b_ci"][1])}</td><td class="num">{p0(d["chance"][0])}–{p0(d["chance"][1])}</td>'
                    f'<td class="num">{p0(i["ceiling"])}</td><td class="num">{p0(i["check1"])}</td><td class="num">{p0(i["b_score"])}</td>'
                    f'<td class="num">{p0(d["b_right"])}</td><td class="num">{p0(d["b_wrong"])}</td></tr>')
    return ('<details><summary>Per-layer numbers</summary><div class="scroll"><table class="layers"><thead><tr>'
            '<th>layer</th><th>ceiling</th><th>check 1</th><th>check 2</th><th>B score</th><th>95% CI</th><th>chance band</th>'
            '<th>id. ceiling</th><th>id. check 1</th><th>id. B score</th><th>B · right</th><th>B · wrong</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
            '<p class="small muted">Greyed rows: reader not trusted (check 2 below 80%). “id.” = the reader trained on the identity prompt. '
            'B · right / wrong: B score on the examples the model answers correctly / wrongly.</p></details>')


# ---------------------------------------------------------------- text
def findings(m, a, rows1, sfx):
    """Short bullet notes for one model and photo type."""
    s, layers = a["summary"], a["layers"]
    d = layers[-1]["direct"]
    sd, si = s["direct"], s["identity"]
    trusted = s["trusted_layers"]
    n_a, n_c, n_b = len(s["holds_A"]), len(s["near_chance"]), len(s["holds_B"])
    ok = lambda x: '<b class="pass">passes</b>' if x else '<b class="fail">fails</b>'

    if sd["check1_last"] != sd["check1_last"]:
        c1 = f"undefined: the model answers “{rows1[0]['answer']}” on every example, so there is nothing to agree with on the other side."
    else:
        c1 = (f"direct reader {p0(sd['check1_last'])} (bar {p0(0.95 * sd['ceiling_last'])}), {ok(sd['check1_pass'])}; "
              f"identity reader {p0(si['check1_last'])} (bar {p0(0.95 * si['ceiling_last'])}), {ok(si['check1_pass'])}.")
    if n_a == len(trusted):
        outcome = f"all {n_a} trusted layers hold A's answer."
    else:
        outcome = (f"{n_a} of {len(trusted)} trusted layers hold A's answer, {n_c} {'is' if n_c == 1 else 'are'} near chance, "
                   f"{'none' if n_b == 0 else n_b} hold{'s' if n_b == 0 else ''} B's.")
    items = [f"<b>Check 1</b>: {c1}",
             f"<b>Check 2</b>: {len(trusted)} trusted layers ({runs(trusted)}).",
             f"<b>Outcome</b>: {outcome} Last-layer B score {p0(d['b_score'])} ({p0(d['b_ci'][0])}–{p0(d['b_ci'][1])})."]

    flat = [k for k in s["near_chance"] if not 0.05 <= layers[k]["direct"]["left_rate"] <= 0.95]
    if flat:
        items.append(f"Near-chance layer{'s' if len(flat) > 1 else ''} {runs(flat)}: the reader gives the same word for almost every example, so {'they say' if len(flat) > 1 else 'it says'} nothing either way.")
    ncorr = [r for r in rows1 if r["answer"] == r["relB"]]
    cl = sum(r["relB"] == "left" for r in ncorr)
    if 0 < cl < len(ncorr):
        note = f"; only {min(cl, len(ncorr) - cl)} correct example(s) in one class, so read loosely" if min(cl, len(ncorr) - cl) <= 3 else ""
        items.append(f"Right vs wrong: on the {len(ncorr)} examples it answers correctly the last-layer B score is {p0(d['b_right'])}, "
                     f"on the {len(rows1) - len(ncorr)} it gets wrong {p0(d['b_wrong'])}{note}.")
    r = REPORT_FOUND.get(m) if not sfx else None
    if r:
        items.append(f"<span class=\"muted\">Report: direct reader {p0(r['check1'])} fails, identity reader {p0(r['id_check1'])} "
                     f"{'passes' if r['id_pass'] else 'fails'}; {r['n_trusted']} trusted layers ({r['trusted']}); "
                     f"{r['holds_a']} hold A's answer, {r['near']} near chance; last-layer B score {p0(r['b_last'])}.</span>")
    return '<ul class="facts">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def compare_table(models, analyses):
    """One table: per model, our plain run, the report's figures (shaded), our outlined run (violet)."""
    def c1(x):
        if x["check1_last"] != x["check1_last"]:
            return '<td class="num">undefined</td>'
        return f'<td class="num">{p0(x["check1_last"])} / {p0(0.95 * x["ceiling_last"])} {"✓" if x["check1_pass"] else "✗"}</td>'

    rows = []
    for m, label in models:
        group = []
        if m in analyses:
            group.append(("ours", "" if m in REPORT_FOUND else "added", analyses[m]))
        if m in REPORT_FOUND:
            group.append(("report", "rep-row", None))
        if m + "_marked" in analyses:
            group.append(("ours · outlined", "added", analyses[m + "_marked"]))
        for k, (run, cls, a) in enumerate(group):
            head = f'<td rowspan="{len(group)}" class="mcell"><span class="m {m}">{e(label)}</span></td>' if k == 0 else ""
            if a:
                s, last = a["summary"], a["layers"][-1]["direct"]
                cells = (f'{c1(s["direct"])}{c1(s["identity"])}'
                         f'<td class="num">{len(s["trusted_layers"])} ({runs(s["trusted_layers"])})</td>'
                         f'<td class="num"><b>{len(s["holds_A"])} / {len(s["near_chance"])} / {len(s["holds_B"])}</b></td>'
                         f'<td class="num">{p0(last["b_score"])} <span class="muted">({p0(last["b_ci"][0])}–{p0(last["b_ci"][1])})</span></td>')
            else:
                r = REPORT_FOUND[m]
                cells = (f'<td class="num">{p0(r["check1"])} / {p0(r["bar"])} ✗</td>'
                         f'<td class="num">{p0(r["id_check1"])} {"✓" if r["id_pass"] else "✗"}</td>'
                         f'<td class="num">{r["n_trusted"]} ({r["trusted"]})</td>'
                         f'<td class="num"><b>{r["holds_a"]} / {r["near"]} / {r["holds_b"]}</b></td>'
                         f'<td class="num">{p0(r["b_last"])} <span class="muted">({p0(r["b_ci"][0])}–{p0(r["b_ci"][1])})</span></td>')
            rows.append(f'<tr class="{cls}{" grp" if k == 0 else ""}">{head}<td class="run">{run}</td>{cells}</tr>')
    return ('<div class="scroll"><table class="probe-table"><thead><tr><th>Model</th><th></th>'
            '<th>Check 1 · direct<br>score / bar</th><th>Check 1 · identity<br>score / bar</th>'
            '<th>Trusted layers (Check 2)</th><th>Trusted layers:<br>A / chance / B</th><th>Last-layer B score<br>(95% CI)</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def build_probe(results, models):
    """Returns (table, switchable chart panel, layers holding B's answer over all runs, number of runs)."""
    root = Path(results) / "probe"
    rep = json.loads((Path(__file__).with_name("report_probe.json")).read_text())
    analyses, views = {}, []
    for m, label in models:
        for sfx in ("", "_marked"):
            p = root / m / f"analysis{sfx}.json"
            if not p.exists():
                continue
            a = analyses[m + sfx] = json.loads(p.read_text())
            rows1 = [json.loads(l) for l in open(root / m / "test" / f"s1{sfx}.jsonl") if l.strip()]
            added = m not in REPORT_FOUND or bool(sfx)
            overlay = None if sfx else rep.get(m)
            k2 = [("s1", "B score, direct reader"), ("s2", "B score, identity-trained reader")]
            if overlay:
                k2.append(("rep", "B score in the report"))
            k2 += [("ci", "bootstrap 95% interval"), ("chance", "chance band (shuffled-label readers)"), ("untrusted", "reader not trusted (lines faded)")]
            views.append(f"""
<div class="pview{' added' if added else ''}" data-m="{m}" data-p="{sfx or 'plain'}"{'' if (m, sfx) == (models[0][0], '') else ' hidden'}>
  <h4 class="vtitle">{e(label)} · {"outlined" if sfx else "plain"} photos{'<span class="tag added">added</span>' if added else ''}</h4>
  {findings(m, a, rows1, sfx)}
  <div class="pair">
    <figure><figcaption class="ctitle">Can the reader be trusted at this layer?</figcaption>
      {key([("s1", "Ceiling"), ("s2", "Check 1"), ("bar1", "Check 1 bar (95% of ceiling), last layer"),
            ("s3", "Check 2"), ("bar2", "Check 2 bar (80%), per layer")])}
      {trust_chart(label, a["layers"])}</figure>
    <figure><figcaption class="ctitle">Whose answer does the trusted reader see?</figcaption>
      {key(k2)}
      {bscore_chart(label, a["layers"], overlay)}</figure>
  </div>
  {layer_table(a["layers"])}
</div>""")
    btn = lambda grp, val, text, on: f'<button type="button" data-{grp}="{val}" aria-pressed="{"true" if on else "false"}">{text}</button>'
    panel = (f'<div class="panel-probe"><div class="switches">'
             f'<div class="seg" role="group" aria-label="Model">'
             + "".join(btn("m", m, e(label), i == 0) for i, (m, label) in enumerate(models) if m in analyses)
             + '</div><div class="seg" role="group" aria-label="Photos">'
             + btn("p", "plain", "plain photos", True) + btn("p", "_marked", '<i class="vdot"></i>outlined photos', False)
             + "</div></div>" + "".join(views) + "</div>" + PANEL_JS)
    held_b = sum(len(a["summary"]["holds_B"]) for a in analyses.values())
    return compare_table(models, analyses), panel, held_b, len(analyses)


PANEL_JS = """<script>
(() => {
  const root = document.currentScript.previousElementSibling, cur = { m: "", p: "plain" };
  const views = root.querySelectorAll(".pview"), btns = root.querySelectorAll("button");
  cur.m = views[0].dataset.m;
  const show = () => {
    views.forEach(v => { v.hidden = !(v.dataset.m === cur.m && v.dataset.p === cur.p); });
    btns.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.m ? b.dataset.m === cur.m : b.dataset.p === cur.p)));
  };
  btns.forEach(b => b.addEventListener("click", () => { if (b.dataset.m) cur.m = b.dataset.m; else cur.p = b.dataset.p; show(); }));
})();
</script>"""


PROBE_CSS = """
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.pair figure {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px 6px; display: grid; gap: 6px; }}
.ctitle {{ font: 600 0.9rem var(--body); color: var(--ink); margin: 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
.chart .grid {{ stroke: var(--line); stroke-width: 1; }} .chart .base {{ stroke: var(--muted); stroke-width: 1; }}
.chart .mid {{ stroke: var(--muted); stroke-width: 1; }}
.chart .tick {{ fill: var(--muted); font: 11px var(--mono); }}
.chart .ln {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.ln.s1, .key i.s1 {{ stroke: var(--c1); background: var(--c1); }}
.ln.s2, .key i.s2 {{ stroke: var(--c2); background: var(--c2); }}
.ln.s3, .key i.s3 {{ stroke: var(--c3); background: var(--c3); }}
.ln.rep {{ stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 4; }}
.key i.rep {{ background: repeating-linear-gradient(90deg, var(--muted) 0 5px, transparent 5px 8px); }}
.chart .bar1 {{ fill: none; stroke: var(--c2); stroke-width: 1.5; stroke-dasharray: 2 3; }}
.chart .bar2 {{ stroke: var(--c3); stroke-width: 1.5; stroke-dasharray: 2 3; }}
.key i.bar1 {{ background: repeating-linear-gradient(90deg, var(--c2) 0 2px, transparent 2px 5px); }}
.key i.bar2 {{ background: repeating-linear-gradient(90deg, var(--c3) 0 2px, transparent 2px 5px); }}

.chart .chance, .key i.chance {{ fill: var(--muted); background: var(--muted); opacity: 0.16; }}
.chart .ci, .key i.ci {{ fill: var(--c1); background: var(--c1); opacity: 0.16; }}
.chart .hatch {{ stroke: var(--muted); stroke-width: 1; opacity: 0.35; }}
.key i.untrusted {{ background: repeating-linear-gradient(45deg, var(--muted) 0 1px, transparent 1px 4px); opacity: 0.7; }}
.chart .faint {{ opacity: 0.28; }}
.chart .pt.s1 {{ fill: var(--c1); }} .chart .pt.s2 {{ fill: var(--c2); }}
.chart .zone {{ fill: var(--muted); font: 600 10.5px var(--body); }}
.chart .hit {{ fill: transparent; }} .chart .hit:hover {{ fill: var(--ink); opacity: 0.06; }}
.key {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.78rem; color: var(--muted); }}
.key span {{ display: inline-flex; align-items: center; gap: 6px; }}
.key i {{ display: inline-block; width: 16px; height: 3px; border-radius: 2px; }}
.key i.chance, .key i.ci, .key i.untrusted {{ height: 10px; }}
.probe {{ display: grid; gap: 14px; }}
details summary {{ cursor: pointer; font-size: 0.88rem; color: var(--accent); }}
details {{ display: grid; gap: 8px; }}
table.layers td, table.layers th {{ padding: 4px 8px; }}
tr.untr td {{ color: var(--muted); }}
b.pass {{ color: var(--ok); }} b.fail {{ color: var(--bad); }}
.answer {{ font-size: 1.12rem; background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--ok); border-radius: 8px; padding: 12px 16px; }}
table.probe-table td.mcell {{ vertical-align: middle; }}
table.probe-table tr.grp td {{ border-top: 2px solid var(--line); }}
table.probe-table td.run {{ font-size: 0.84rem; color: var(--muted); white-space: nowrap; }}
.panel-probe {{ display: grid; gap: 14px; }}
.switches {{ display: flex; flex-wrap: wrap; gap: 10px 18px; }}
.seg {{ display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }}
.seg button {{ font: 600 0.88rem var(--body); color: var(--muted); background: transparent; border: 0; padding: 8px 14px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }}
.seg button + button {{ border-left: 1px solid var(--line); }}
.seg button[aria-pressed="true"] {{ background: var(--rep); color: var(--ink); }}
.seg button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
.vdot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--violet); }}
.pview {{ display: grid; gap: 12px; }} .pview[hidden] {{ display: none; }}
.vtitle {{ font: 700 1.1rem var(--display); }}
ul.facts {{ font-size: 0.95rem; }}
"""
