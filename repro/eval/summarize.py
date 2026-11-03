"""Section-2 table: correct / answered-from-A's-spot per setting and model.

    python repro/eval/summarize.py --set test --models qwen3vl bagel gemma4 molmo2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PP_RESULTS
from eval.prompts import SETTINGS

LABELS = {"s1": "1 Original", "s1_identity": "1 Identity variant",
          "s2": "2 Ground-truth target", "s4": "4 Control (own view)"}
LABELS.update({k + "_marked": v + " [marked]" for k, v in list(LABELS.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--models", nargs="+", default=["qwen3vl", "bagel", "gemma4", "molmo2"])
    ap.add_argument("--detail", action="store_true",
                    help="also print answer bias, per-class accuracy and sample invalid answers")
    args = ap.parse_args()

    print(f"{'setting':33s} {'model':8s} {'correct':>14s} {'answered from A':>18s} {'invalid':>8s}")
    for setting in SETTINGS:
        for model in args.models:
            path = PP_RESULTS / model / args.set / f"{setting}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in open(path)]
            n = len(rows)
            ok = sum(r["answer"] == r["truth"] for r in rows)
            from_a = sum(r["answer"] == r["relA"] for r in rows)
            bad = sum(r["answer"] == "invalid" for r in rows)
            print(f"{LABELS[setting]:33s} {model:8s} {ok:>4d}/{n} {100*ok/n:4.0f}%"
                  f" {from_a:>6d}/{n} {100*from_a/n:4.0f}% {bad:>8d}")
            if args.detail:
                said_left = sum(r["answer"] == "left" for r in rows)
                per_class = []
                for word in ("left", "right"):
                    group = [r for r in rows if r["truth"] == word]
                    hit = sum(r["answer"] == r["truth"] for r in group)
                    per_class.append(f"truth={word}: {hit}/{len(group)}")
                print(f"{'':42s}says left {said_left}/{n}; " + "; ".join(per_class))
                for r in [r for r in rows if r["answer"] == "invalid"][:3]:
                    print(f"{'':42s}invalid raw: {r['raw']!r}")


if __name__ == "__main__":
    main()
