"""Write marked copies of every clean example's photos: X outlined orange, Y blue.

CPU only, from the saved RGB + mask; nothing is re-rendered.
Output: <id>_A_rgb_marked.png and <id>_B_rgb_marked.png next to the originals.

    python repro/render/make_marked.py --set test
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PP_DATA
from common.marks import outline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    args = ap.parse_args()

    root = PP_DATA / args.set
    recs = [r for r in map(json.loads, open(root / "examples_labeled.jsonl")) if r["clean"]]
    for rec in recs:
        d = root / rec["scene"]
        for v in "AB":
            img = outline(d / f"{rec['id']}_{v}_rgb.png", d / f"{rec['id']}_{v}_sem.png", rec[v])
            cv2.imwrite(str(d / f"{rec['id']}_{v}_rgb_marked.png"), img)
    print(f"{len(recs)} clean examples -> {2 * len(recs)} marked photos under {root}")


if __name__ == "__main__":
    main()
