"""Outline X (orange) and Y (blue) on a photo, from the saved instance mask.

Used for the model's marked condition and for the gallery, so both show the same marks.
"""
import cv2
import numpy as np

X_BGR, Y_BGR = (40, 120, 255), (255, 150, 40)  # orange, blue


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


def outline(rgb_path, sem_path, view):
    """BGR image with X and Y outlined (about 3 px wide)."""
    img = cv2.imread(str(rgb_path))
    sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
    k = np.ones((5, 5), np.uint8)
    for key, color in (("X", X_BGR), ("Y", Y_BGR)):
        sid = find_sid(sem, view[key])
        if sid is None:
            continue
        m = (sem == sid).astype(np.uint8)
        img[(cv2.dilate(m, k) > 0) & ~(cv2.erode(m, k) > 0)] = color
    return img
