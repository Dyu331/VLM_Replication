"""Section 3 of the replication report ("Inside the model"), from the probe analyses.

Reads local_data/results/probe/<model>/analysis.json and test/s1.jsonl (pulled from Turbo), and the
report's own per-layer numbers from repro/report/report_probe.json (extract_report_probe.py).
"""
import html
import json
from pathlib import Path

e = html.escape
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
    out.append(f'<path d="{path(xs, ys, [0.95 * v["ceiling"] for v in d])}" class="bar95"/>')
    for key, cls in (("ceiling", "s1"), ("check1", "s2"), ("check2", "s3")):
        out.append(f'<path d="{path(xs, ys, [v[key] for v in d])}" class="ln {cls}"/>')
    tips = [f"layer {i}\nCeiling: {p0(v['ceiling'])}\nCheck 1: {p0(v['check1'])}\nCheck 2: {p0(v['check2'])}"
            for i, v in enumerate(d)]
    out.append(hover(xs, n, tips))
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{e(label)}: can the reader be trusted at this layer?">'
            + "".join(out) + "</svg>")


def bscore_chart(label, layers, report):
    n = len(layers)
    xs, ys, out = frame(n, "B score")
    d = [l["direct"] for l in layers]
    w = (W - L0 - R0) / (n - 1)
    untrusted = [i for i, v in enumerate(d) if v["check2"] < CHECK2_MIN]
    for r in runs(untrusted).split(", "):
        if r == "none":
            break
        a, b = (int(r), int(r)) if "–" not in r else map(int, r.split("–"))
        x0, x1 = max(L0, xs(a) - w / 2), min(W - R0, xs(b) + w / 2)
        out.append(f'<rect x="{x0:.1f}" y="{T0}" width="{x1 - x0:.1f}" height="{H - T0 - B0}" class="untrusted"/>')
    out.append(f'<path d="{band(xs, ys, [v["chance"][0] for v in d], [v["chance"][1] for v in d])}" class="chance"/>'
               f'<path d="{band(xs, ys, [v["b_ci"][0] for v in d], [v["b_ci"][1] for v in d])}" class="ci"/>'
               f'<line x1="{L0}" x2="{W - R0}" y1="{ys(0.5):.1f}" y2="{ys(0.5):.1f}" class="mid"/>')
    if report:
        out.append(f'<path d="{path(xs, ys, [r.get("b_score") for r in report])}" class="ln rep"/>')
    out.append(f'<path d="{path(xs, ys, [l["identity"]["b_score"] for l in layers])}" class="ln s2"/>'
               f'<path d="{path(xs, ys, [v["b_score"] for v in d])}" class="ln s1"/>')
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
def findings(m, label, a, rows1, is_report_model):
    s, layers = a["summary"], a["layers"]
    last = layers[-1]
    d, i = last["direct"], last["identity"]
    trusted = s["trusted_layers"]
    tl = [layers[k] for k in trusted]
    sd, si = s["direct"], s["identity"]
    verdict = lambda ok: '<b class="pass">passes</b>' if ok else '<b class="fail">fails</b>'
    ncorr = [r for r in rows1 if r["answer"] == r["relB"]]
    cl = sum(r["relB"] == "left" for r in ncorr)
    first = lambda key: next((l["layer"] for l in layers if l[key]["ceiling"] >= 0.8), None)
    fd, fi = first("direct"), first("identity")
    n_a, n_c, n_b = len(s["holds_A"]), len(s["near_chance"]), len(s["holds_B"])
    outcome = ("Outcome A" if n_b == 0 and n_a >= n_c else "Outcome C" if n_b == 0 else "Outcome B/D")
    flat = [k for k in s["near_chance"] if not 0.05 <= layers[k]["direct"]["left_rate"] <= 0.95]
    flat_note = (f" At near-chance layer{'s' if len(flat) > 1 else ''} {runs(flat)} the reader gives the same word for almost every example "
                 f"(“left” rate under 5% or over 95%), so its B score there carries no information either way."
                 if flat else "")

    if 0 < cl < len(ncorr):
        split = (f'Class-balanced within each group, the {len(ncorr)} examples the model answers correctly (B = left in {cl}, right in {len(ncorr) - cl}) '
                 f'score {p0(d["b_right"])} at the last layer and the {len(rows1) - len(ncorr)} it gets wrong {p0(d["b_wrong"])}.')
        if min(cl, len(ncorr) - cl) <= 3:
            split += f" Caution: one class of the correct group has only {min(cl, len(ncorr) - cl)} example(s), so that figure is fragile."
    else:
        split = (f"The {len(ncorr)} examples this model answers correctly all have the same B answer, "
                 "so a class-balanced score is undefined for that group and the split cannot be read.")

    if sd["check1_last"] != sd["check1_last"]:
        one = rows1[0]["answer"]
        check1 = (f'<p><b>Check 1</b> cannot be computed: the model answers “{one}” on every setting-1 example, so its answers have one class and '
                  f'a class-balanced agreement is undefined. Both readers are still accurate on held-out direct questions (ceiling {p0(sd["ceiling_last"])} and {p0(si["ceiling_last"])}).</p>')
    else:
        check1 = None
    return (check1 or f"""
  <p><b>Check 1</b>: at the last layer the direct-question reader agrees with the model's actual setting-1 answer in {p0(sd['check1_last'])} of examples (class-balanced)
  against its own ceiling of {p0(sd['ceiling_last'])}; the bar is {p0(0.95 * sd['ceiling_last'])}, so it {verdict(sd['check1_pass'])}.
  The reader trained on the identity prompt agrees in {p0(si['check1_last'])} (bar {p0(0.95 * si['ceiling_last'])}): it {verdict(si['check1_pass'])}.</p>""") + f"""
  <p><b>Check 2</b>: the identity control is {p0(d['check2'])} at the last layer and at least 80% at layers {runs(trusted)} ({len(trusted)} layers); only those are read.
  B score over the trusted layers: {span([l['direct']['b_score'] for l in tl])} with the direct reader, {span([l['identity']['b_score'] for l in tl])} with the identity-trained reader;
  at the last layer {p0(d['b_score'])} (95% CI {p0(d['b_ci'][0])}–{p0(d['b_ci'][1])}) and {p0(i['b_score'])}.
  {"No trusted layer rises above the permutation band." if n_b == 0 else f"{n_b} trusted layers rise above the permutation band."}</p>
  <p><b>{outcome}</b>: {n_a} of {len(trusted)} trusted layers hold A's answer, {n_c} near chance, {n_b} B's.
  {"<i>Never imagined</i>, as far as these readers can see." if outcome == "Outcome A" else "Nothing can be concluded from most of these layers, but none holds B's answer." if n_b == 0 else ""}{flat_note}
  Left/right first becomes readable (ceiling ≥ 80%) at layer {fd} for direct questions and {fi} under the task prompt.</p>
  <p><b>Right vs wrong examples.</b> {split}</p>"""


def compare_table(models, analyses, with_report=True):
    def c1(x):
        if x["check1_last"] != x["check1_last"]:
            return '<td class="num">undefined (one answer)</td>'
        return (f'<td class="num">{p0(x["check1_last"])} / bar {p0(0.95 * x["ceiling_last"])} '
                f'{"✓" if x["check1_pass"] else "✗"}</td>')
    rows = []
    for m, label in models:
        a = analyses.get(m)
        if not a:
            continue
        s, last = a["summary"], a["layers"][-1]["direct"]
        cls = "" if m in REPORT_FOUND and with_report else ' class="added"'
        rows.append(f'<tr{cls}><td><span class="m {m}">{e(label)}</span></td>'
                    f'{c1(s["direct"])}{c1(s["identity"])}'
                    f'<td class="num">{len(s["trusted_layers"])} ({runs(s["trusted_layers"])})</td>'
                    f'<td class="num">{p0(last["b_score"])} ({p0(last["b_ci"][0])}–{p0(last["b_ci"][1])})</td>'
                    f'<td class="num">{len(s["holds_A"])} / {len(s["near_chance"])} / {len(s["holds_B"])}</td></tr>')
        r = REPORT_FOUND.get(m) if with_report else None
        if r:
            rows.append(f'<tr class="rep-row"><td><span class="m {m}">{e(label)}</span> · report</td>'
                        f'<td class="num">{p0(r["check1"])} / bar {p0(r["bar"])} ✗</td>'
                        f'<td class="num">{p0(r["id_check1"])} {"✓" if r["id_pass"] else "✗"}</td>'
                        f'<td class="num">{r["n_trusted"]} ({r["trusted"]})</td>'
                        f'<td class="num">{p0(r["b_last"])} ({p0(r["b_ci"][0])}–{p0(r["b_ci"][1])})</td>'
                        f'<td class="num">{r["holds_a"]} / {r["near"]} / {r["holds_b"]}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Model</th><th>Check 1 · direct reader</th><th>Check 1 · identity reader</th>'
            '<th>Trusted layers</th><th>Last-layer B score (95% CI)</th><th>Trusted layers: A / chance / B</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def build_probe(results, models, sfx=""):
    """sfx="_marked": the outlined-photo run (no report overlay; all of it is an addition)."""
    root = Path(results) / "probe"
    rep = json.loads((Path(__file__).with_name("report_probe.json")).read_text())
    analyses, blocks = {}, []
    for m, label in models:
        p = root / m / f"analysis{sfx}.json"
        if not p.exists():
            continue
        a = analyses[m] = json.loads(p.read_text())
        rows1 = [json.loads(l) for l in open(root / m / "test" / f"s1{sfx}.jsonl") if l.strip()]
        added = m not in REPORT_FOUND or bool(sfx)
        k2 = [("s1", "B score, direct reader"), ("s2", "B score, identity-trained reader")]
        if not sfx and m in REPORT_FOUND:
            k2.append(("rep", "B score in the report"))
        k2 += [("ci", "bootstrap 95% interval"), ("chance", "chance band"), ("untrusted", "reader not trusted")]
        blocks.append(f"""
<section class="probe{' added' if added else ''}">
  <h3>{e(label)}{" · outlined photos" if sfx else ""}{'<span class="tag added">added</span>' if added else ''}</h3>
  <div class="prose">{findings(m, label, a, rows1, not added)}</div>
  <div class="pair">
    <figure><figcaption class="ctitle">Can the reader be trusted at this layer?</figcaption>
      {key([("s1", "Ceiling"), ("s2", "Check 1"), ("s3", "Check 2"), ("bar95", "95% of ceiling")])}
      {trust_chart(label, a["layers"])}</figure>
    <figure><figcaption class="ctitle">Whose answer does the trusted reader see?</figcaption>
      {key(k2)}
      {bscore_chart(label, a["layers"], None if sfx else rep.get(m))}</figure>
  </div>
  {layer_table(a["layers"])}
</section>""")
    return compare_table(models, analyses, not sfx), "".join(blocks)


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
.chart .bar95 {{ fill: none; stroke: var(--muted); stroke-width: 1.25; stroke-dasharray: 3 3; }}
.key i.bar95 {{ background: repeating-linear-gradient(90deg, var(--muted) 0 3px, transparent 3px 6px); }}
.chart .chance, .key i.chance {{ fill: var(--muted); background: var(--muted); opacity: 0.16; }}
.chart .ci, .key i.ci {{ fill: var(--c1); background: var(--c1); opacity: 0.16; }}
.chart .untrusted, .key i.untrusted {{ fill: var(--ink); background: var(--ink); opacity: 0.06; }}
.key i.untrusted {{ opacity: 0.12; }}
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
"""
