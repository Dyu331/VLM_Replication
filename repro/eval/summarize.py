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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--models", nargs="+", default=["qwen3vl", "bagel", "gemma4", "molmo2"])
    args = ap.parse_args()

    print(f"{'setting':24s} {'model':8s} {'correct':>14s} {'answered from A':>18s} {'invalid':>8s}")
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
            print(f"{LABELS[setting]:24s} {model:8s} {ok:>4d}/{n} {100*ok/n:4.0f}%"
                  f" {from_a:>6d}/{n} {100*from_a/n:4.0f}% {bad:>8d}")


if __name__ == "__main__":
    main()
