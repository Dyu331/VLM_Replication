"""Check 0 + contact sheet: load every candidate HSSD model into an emptied room and photograph it.

    python repro/v4/thumbs.py [--scene apt_0]

Writes $V4_DATA/thumbs/{sheet.png, <id>.png, room_raw.png, room_composite.png} and
thumbs/report.json (load failures, sizes as placed, room facts).
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.config_v4 import CATEGORIES, V4_DATA
from v4.scene import Room, mask_stats, yaw_towards

TILE = 160


def shoot(room, asset, rng):
    """Place the asset somewhere it fits and photograph it from a spot with a clear view."""
    for _ in range(60):
        p = np.array(room.pf.get_random_navigable_point())
        o = room.add(asset, p[0], p[2], rng.uniform(0, 360), sid=1)
        if o is None:
            continue
        size = max(o["height"], np.ptp(o["foot"][:, 0]), np.ptp(o["foot"][:, 1]))
        for _ in range(60):
            q = np.array(room.pf.get_random_navigable_point())
            d = np.hypot(q[0] - o["centre"][0], q[2] - o["centre"][2])
            if not (1.2 * size + 0.8 <= d <= 1.2 * size + 2.5):
                continue
            xz = (q[0], q[2])
            if not room.line_of_sight(xz, o, [o]):
                continue
            yaw = yaw_towards([q[0], 0, q[2]], o["centre"])
            rgb, sem, depth = room.render(xz, yaw)
            st = mask_stats(sem, 1)
            if st["vis_pct"] < 0.5 or st["xmin"] <= 1 or st["xmax"] >= sem.shape[1] - 2 or st["ymax"] >= sem.shape[0] - 2:
                continue
            img = room.composite(rgb, sem, depth, xz, yaw, keep={1})
            info = {"height": round(o["height"], 2), "foot": np.ptp(o["foot"], axis=0).round(2).tolist(),
                    "vis_pct": st["vis_pct"], "dist": round(float(d), 2)}
            room.remove(o)
            return img, rgb, st, info
        room.remove(o)
    return None


def crop(img, st):
    cx, cy = (st["xmin"] + st["xmax"]) / 2, (st["ymin"] + st["ymax"]) / 2
    half = max(st["xmax"] - st["xmin"], st["ymax"] - st["ymin"]) * 0.65 + 8
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, img.shape[1])), int(min(cy + half, img.shape[0]))
    return cv2.resize(img[y0:y1, x0:x1], (TILE, TILE), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="apt_0")
    args = ap.parse_args()
    out = V4_DATA / "thumbs"
    out.mkdir(parents=True, exist_ok=True)
    cands = json.loads((V4_DATA / "assets_candidates.json").read_text())
    rng = np.random.default_rng(0)
    room = Room(args.scene)
    print(f"{args.scene}: removed {room.n_removed} objects, floor y {room.floor_y:.3f}, nav y {room.nav_y:.3f}, "
          f"navigable area {room.pf.navigable_area:.1f} m2", flush=True)
    report = {"scene": args.scene, "removed": room.n_removed, "floor_y": room.floor_y,
              "navigable_area": room.pf.navigable_area, "models": {}}
    tiles = {}
    first = True
    for a in cands:
        try:
            r = shoot(room, a, rng)
        except Exception as e:  # load failure: report and go on
            traceback.print_exc()
            report["models"][a["id"]] = {"category": a["category"], "error": repr(e)}
            continue
        if r is None:
            report["models"][a["id"]] = {"category": a["category"], "error": "no placement / view found"}
            print(f"  {a['category']:13s} {a['id'][:8]} NO VIEW", flush=True)
            continue
        img, rgb, st, info = r
        if first:
            cv2.imwrite(str(out / "room_raw.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "room_composite.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            first = False
        t = crop(img, st)
        cv2.imwrite(str(out / f"{a['id']}.png"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
        tiles[a["id"]] = t
        report["models"][a["id"]] = {"category": a["category"], **info}
        print(f"  {a['category']:13s} {a['id'][:8]} h {info['height']} foot {info['foot']} "
              f"(listed {np.round(a['dims'], 2).tolist()})", flush=True)
    room.close()

    # sheet: one row per category, tiles labelled with the first 6 characters of the id
    rows = []
    for cat in CATEGORIES:
        ids = [a["id"] for a in cands if a["category"] == cat]
        row = [np.full((TILE, 150, 3), 255, np.uint8)]
        cv2.putText(row[0], cat, (5, TILE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for i in ids:
            t = tiles.get(i, np.full((TILE, TILE, 3), 60, np.uint8)).copy()
            cv2.putText(t, i[:6], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
            row.append(t)
        row += [np.full((TILE, TILE, 3), 255, np.uint8)] * (12 - len(ids))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out / "sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    bad = [k for k, v in report["models"].items() if "error" in v]
    print(f"{len(tiles)} models photographed, {len(bad)} failed -> {out / 'sheet.png'}")


if __name__ == "__main__":
    main()
