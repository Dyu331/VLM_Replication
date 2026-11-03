"""Model-independent eval loop: every clean example x every setting -> one jsonl per setting.

A model only supplies answer(image_path, prompt) -> raw text. Imports nothing heavy, so it
works in both the transformers-v5 env (run_vlm.py) and the BAGEL env (run_bagel.py).
"""
import json

from common.config import PP_DATA, PP_RESULTS
from eval.prompts import build, image_name, parse


def load_clean(set_name):
    path = PP_DATA / set_name / "examples_labeled.jsonl"
    return [r for r in map(json.loads, open(path)) if r["clean"]]


def run_settings(model_name, answer, sets, settings, limit=None):
    for set_name in sets:
        recs = load_clean(set_name)[:limit]
        out_dir = PP_RESULTS / model_name / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for setting in settings:
            rows = []
            for rec in recs:
                view, text, truth_view = build(rec, setting)
                img = PP_DATA / set_name / rec["scene"] / image_name(rec, setting, view)
                raw = answer(img, text)
                rows.append({"id": rec["id"], "scene": rec["scene"], "setting": setting,
                             "raw": raw, "answer": parse(raw), "truth": rec[f"rel{truth_view}"],
                             "relA": rec["relA"], "relB": rec["relB"], "prompt": text})
            with open(out_dir / f"{setting}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            acc = sum(r["answer"] == r["truth"] for r in rows) / max(len(rows), 1)
            bad = sum(r["answer"] == "invalid" for r in rows)
            print(f"{model_name} {set_name} {setting}: {len(rows)} prompts, accuracy {acc:.1%}, "
                  f"invalid {bad}  e.g. {rows[0]['raw']!r}", flush=True)
