"""Search margins: how far does the search value of depth d get above the static eval?

Input: files of SelfPlay -P --perf-scores (fen, static eval, search score of depth 1 .. D, side to move),
one per eval variant, on the same positions. Futility pruning at a node of depth d skips the quiet moves
when static + margin(d) <= alpha, margin(d) = 24 * 2^(d-1) cp (Search/Core.hs), so the margin has to cover
how much the search of depth d gains over the static eval: the upper quantiles of score(d) - static.
Mate scores (|score| > --mate) are left out.

Example:
    python hv_margins.py --margin-base 24 old=scores-old.csv new=scores-new.csv
"""

from __future__ import annotations

import argparse
import sys

import env_dlls  # noqa: F401  (before numpy)
import numpy as np


MISSING = 1 << 30   # depth not reached (search ended early, e.g. mate): treated like a mate score


def load(path: str) -> np.ndarray:
    """Rows: static, score(1), score(2), ...; incomplete lines (file still written) are skipped."""
    rows = []
    with open(path, encoding="utf-8") as fi:
        for line in fi:
            if not line.endswith("\n"):
                continue
            parts = line.rstrip("\n").split(",")
            try:
                rows.append([int(x) for x in parts[1:]])
            except ValueError:
                continue
    width = max(len(r) for r in rows)
    return np.array([r + [MISSING] * (width - len(r)) for r in rows], dtype=np.int64)


def error_curves(data: dict[str, np.ndarray], args) -> int:
    """Files of --perf-quiet: static, quiet(1 .. D), rest(1 .. D). A futility cut at a node of depth d
    happens when static + margin <= alpha and is harmful when the best skipped quiet move beats alpha and
    the best searched move: with the smallest alpha that prunes (static + margin, the worst case) that is
    quiet(d) > max(static + margin, rest(d)). Reported per depth: the harmful rate and mean size at the
    current margin, and the margin needed for the target rates."""
    n = min(a.shape[0] for a in data.values())
    width = min(a.shape[1] for a in data.values())
    depth = (width - 1) // 2
    targets = [float(t) for t in args.targets.split(",")]
    margins = np.arange(0, 2049, 4)
    print(f"positions {n}, depths 1 .. {depth}; harmful cut: quiet(d) > max(static + margin, rest(d))")
    print("needed margin (cp) for harmful rates " + ", ".join(f"{t}%" for t in targets))
    for d in range(1, depth + 1):
        cur = args.margin_base << (d - 1)
        print(f"depth {d} (current margin {cur}):")
        for name, a in data.items():
            a = a[:n]
            st, q, r = a[:, 0], a[:, d], a[:, depth + d]
            ok = (np.abs(st) < args.mate) & (np.abs(q) < args.mate) & ((np.abs(r) < args.mate) | (r < -args.mate))
            st, q, r = st[ok], q[ok], r[ok]
            exc = q[None, :] - np.maximum(st[None, :] + margins[:, None], r[None, :])   # margins x positions
            rate = (exc > 0).mean(axis=1) * 100
            size = np.where(exc > 0, exc, 0).sum(axis=1) / np.maximum((exc > 0).sum(axis=1), 1)
            i = int(np.searchsorted(margins, cur))
            need = []
            for t in targets:
                j = np.nonzero(rate <= t)[0]
                need.append(f"{margins[j[0]]:5d}" if len(j) else "  >2k")
            print(f"  {name:6s} n {ok.sum():5d}  at {cur}: {rate[i]:5.2f}% (mean {size[i]:4.0f} cp)   needed: "
                  + "  ".join(need))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", help="name=file pairs")
    ap.add_argument("--margin-base", type=int, default=24, help="futility margin of depth 1 (cp)")
    ap.add_argument("--mate", type=int, default=10000, help="scores above are mate scores")
    ap.add_argument("--quiet-rest", action="store_true",
                    help="files of --perf-quiet with quiet and other moves: harmful cut rates per margin")
    ap.add_argument("--targets", default="2,1,0.5,0.2",
                    help="--quiet-rest: harmful cut rates (%%) for which the needed margin is reported")
    args = ap.parse_args()

    data = {}
    for v in args.variants:
        name, _, path = v.partition("=")
        if not path:
            print(f"Expected name=file, got {v}", file=sys.stderr)
            return 2
        data[name] = load(path)
    if args.quiet_rest:
        return error_curves(data, args)
    depth = max(a.shape[1] for a in data.values()) - 1
    n = min(a.shape[0] for a in data.values())
    data = {k: a[:n] for k, a in data.items()}     # the same positions in all variants
    print(f"positions: " + ", ".join(f"{n} {a.shape[0]}" for n, a in data.items()) + f"; depths 1 .. {depth}")
    print("score(d) - static (cp): quantiles; 'over' = part above the current futility margin of depth d")
    for d in range(1, depth + 1):
        margin = args.margin_base << (d - 1)
        print(f"depth {d} (margin {margin}):")
        for name, a in data.items():
            st, sc = a[:, 0], a[:, d]
            ok = (np.abs(sc) < args.mate) & (np.abs(st) < args.mate)
            diff = (sc - st)[ok]
            q = np.percentile(diff, [1, 50, 90, 95, 99, 99.9])
            over = np.mean(diff > margin)
            print(f"  {name:6s} n {ok.sum():6d}  1%: {q[0]:6.0f}  median {q[1]:5.0f}  90%: {q[2]:5.0f}  95%: {q[3]:5.0f}"
                  f"  99%: {q[4]:5.0f}  99.9%: {q[5]:6.0f}  over: {100 * over:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
