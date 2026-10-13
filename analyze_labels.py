#!/usr/bin/env python3
"""Compare score labels of several files with the same positions (same file name, same line order).

Usage: python analyze_labels.py <name>=<dir> [<name>=<dir> ...] [--file all-01-filt.csv] [--limit N]

For every label set: best sigmoid scale k and Texel error against the game result (result/2), score
clipped to +-2000; mean absolute difference, correlation and share of |difference| > 150 cp against the
first set. Then the result error of a mix (1-t)*first + t*other (both with their best k).
Pure Python (no numpy).
"""

from __future__ import annotations

import argparse
import math
import os


def load(path: str, limit: int):
    fen, sc, res = [], [], []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for i, ln in enumerate(fh):
            if limit and i >= limit:
                break
            a = ln.rstrip("\r\n").rsplit(",", 2)
            fen.append(a[0]); sc.append(max(-2000.0, min(2000.0, float(a[1])))); res.append(float(a[2]) / 2.0)
    return fen, sc, res


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def err(sc, tg, k):
    return sum((t - sig(s / k)) ** 2 for s, t in zip(sc, tg)) / len(tg)


def best_k(sc, tg):
    best = (1e9, 0)
    for k in range(60, 701, 10):
        e = err(sc, tg, k)
        if e < best[0]:
            best = (e, k)
    return best[1], best[0]


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--file", default="all-01-filt.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    names, scores, tg, fens0 = [], [], None, None
    for s in a.sets:
        name, d = s.split("=", 1)
        fen, sc, res = load(os.path.join(d, a.file), a.limit)
        if fens0 is None:
            fens0, tg = fen, res
        elif fen != fens0:
            raise SystemExit(f"{name}: positions differ from the first set")
        names.append(name); scores.append(sc)
    n = len(tg)
    base = scores[0]
    print(f"{n} positions, mean result {sum(tg) / n:.3f}")
    print(f"{'label':<12}{'best k':>7}{'error':>10}{'mean|d|':>9}{'corr':>7}{'|d|>150':>9}{'|s|>=2000':>10}")
    ks = []
    for nm, sc in zip(names, scores):
        k, e = best_k(sc, tg)
        ks.append(k)
        d = [abs(x - y) for x, y in zip(sc, base)]
        print(f"{nm:<12}{k:>7}{e:>10.5f}{sum(d) / n:>9.1f}{corr(sc, base):>7.3f}"
              f"{sum(1 for x in d if x > 150) / n:>9.3f}{sum(1 for x in sc if abs(x) >= 2000) / n:>10.3f}")
    print()
    print("result error of (1-t)*first + t*other, both with best k:")
    x1 = [sig(s / ks[0]) for s in base]
    for nm, sc, k in zip(names[1:], scores[1:], ks[1:]):
        x2 = [sig(s / k) for s in sc]
        res = []
        for i in range(11):
            t = i / 10
            e = sum((y - ((1 - t) * p + t * q)) ** 2 for y, p, q in zip(tg, x1, x2)) / n
            res.append((e, t))
        e, t = min(res)
        print(f"  {nm:<12} t=0 {res[0][0]:.5f}  t=1 {res[-1][0]:.5f}  best t={t:.1f} {e:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
