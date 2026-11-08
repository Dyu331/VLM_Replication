"""Prompts, saved positions, mask -> image-token pooling and the behaviour summary, shared by
extract.py (Qwen3-VL, Gemma 4; transformers v5) and extract_bagel.py (BAGEL env). No model imports."""
import numpy as np
from PIL import Image

TASK = ("This image shows what you currently see. Imagine you are standing where the {B} is, looking toward "
        "the {X} and the {Y}. From there, is the {X} to the left or to the right of the {Y}? "
        "Answer with exactly one word: left or right.")
DIRECT = "In this image, is the {X} to the left or to the right of the {Y}? Answer with exactly one word: left or right."
POSITIONS = ["answer", "name_B", "name_X", "name_Y", "patch_B", "patch_X", "patch_Y"]
WORDS = ["left", "Left", "right", "Right"]
MIN_CELL = 0.2  # an image token belongs to an object when its mask covers >= 20% of the token's cell [ours]


def cells(sem, sid, rows, cols):
    """Raster indices of image-token cells whose area is >= MIN_CELL covered by the object's mask
    (the photo is square and scaled uniformly, so cells map to equal pixel blocks)."""
    m = Image.fromarray((sem == sid).astype(np.float32), mode="F")
    frac = np.asarray(m.resize((cols, rows), Image.BOX))
    idx = np.flatnonzero(frac.ravel() >= MIN_CELL)
    return idx if len(idx) else np.array([int(frac.argmax())])


def find_last(ids, sub, last=True):
    """Index of the last token of the last (or first) occurrence of token list sub in ids."""
    hits = [i for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub]
    assert hits, sub
    return (hits[-1] if last else hits[0]) + len(sub) - 1


def summarize(rows, label):
    """The PhD's behaviour table: gives B's / A's answer, says left, direct-A, direct-B, flip split."""
    n = len(rows)
    m = lambda f: sum(f(r) for r in rows) / max(n, 1)
    flip = [r for r in rows if r["flip"]]
    same = [r for r in rows if not r["flip"]]
    acc = lambda rs: sum(r["answer"] == r["relB"] for r in rs) / max(len(rs), 1)
    print(f"\n{label}: {n} questions | gives B's answer {m(lambda r: r['answer'] == r['relB']):.0%} | "
          f"gives A's answer {m(lambda r: r['answer'] == r['relA']):.0%} | says left {m(lambda r: r['answer'] == 'left'):.0%} | "
          f"invalid {m(lambda r: r['answer'] == 'invalid'):.0%} | direct-A {m(lambda r: r['direct_A'] == r['relA']):.0%} | "
          f"direct-B {m(lambda r: r['direct_B'] == r['relB']):.0%} | flip questions {acc(flip):.0%} | "
          f"no-flip questions {acc(same):.0%}", flush=True)
    for key, vals in (("depth_rank", ("nearest", "middle", "farthest")), ("image_side", ("left", "right"))):
        parts = []
        for v in vals:
            rs = [r for r in rows if r[key] == v]
            if rs:
                fr = sum(r["answer"] != r["relA"] for r in rs) / len(rs)
                parts.append(f"{v}: B's answer {acc(rs):.0%}, answers differently from A {fr:.0%} (n={len(rs)})")
        print(f"   by {key}: " + "; ".join(parts), flush=True)


ROW_KEYS = ("qid", "photo", "room", "split", "B", "X", "Y", "relA", "relB", "flip", "depth_rank", "image_side",
            "angle_A", "angle_B")
