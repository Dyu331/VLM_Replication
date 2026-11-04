"""Per-layer probe numbers from the original report's HTML (its chart tooltips), for overlaying.

    python repro/report/extract_report_probe.py <report.html>
Writes repro/report/report_probe.json: {model: [{layer, ceiling, check1, check2, b_score, chance, identity_b, right, wrong}]}.
"""
import json
import re
import sys
from pathlib import Path

NAMES = {"Qwen3-VL-8B-Instruct": "qwen3vl", "BAGEL-7B-MoT": "bagel"}
KEYS = {"Ceiling": "ceiling", "Check 1": "check1", "Check 2": "check2", "B score": "b_score",
        "Identity-trained": "identity_b", "Right": "right", "Wrong": "wrong"}

t = open(sys.argv[1]).read()
out = {}
for svg in re.findall(r'<svg class="chart".*?</svg>', t, flags=re.S):
    label = re.search(r'aria-label="([^"]+)"', svg).group(1)
    name = label.split(":")[0]
    if name not in NAMES:
        continue
    layers = out.setdefault(NAMES[name], {})
    for tip in re.findall(r"<title>(.*?)</title>", svg, flags=re.S):
        m = re.match(r"layer (\d+)\n", tip)
        if not m:
            continue
        rec = layers.setdefault(int(m.group(1)), {"layer": int(m.group(1))})
        for k, v in re.findall(r"^([\w\- ]+): (\d+)%$", tip, flags=re.M):
            if k in KEYS:
                rec[KEYS[k]] = int(v) / 100
        c = re.search(r"chance band: (\d+)–(\d+)%", tip)
        if c:
            rec["chance"] = [int(c.group(1)) / 100, int(c.group(2)) / 100]
out = {m: [v[k] for k in sorted(v)] for m, v in out.items()}
path = Path(__file__).with_name("report_probe.json")
path.write_text(json.dumps(out, indent=1))
print({m: (len(v), v[-1]) for m, v in out.items()})
