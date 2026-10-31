"""Evaluation of a node scan (node_scan.ps1): total nodes per config, compared with a base config (paired
bootstrap over the positions), the typical change per position (geometric mean of the node ratios against
the reference config, e.g. pruning off) and the blow-ups (positions with more than twice the nodes of the
reference); optionally by game phase (non pawn material from the sample csv of sample_roots.py).

The total is what a fixed-depth search costs; it is dominated by a few positions whose tree blows up, so the
bootstrap range shows whether a difference is more than the chance of which positions were sampled.

Example:
    python node_scan_report.py --dir C:\\Learn\\Tune\\hvar\\nodescan\\out --ref off --base m50
        --sample C:\\Learn\\Tune\\hvar\\nodescan\\roots-2000.csv
"""
import argparse
import csv
import glob
import math
import os
import random


def load(d):
    res = {}
    for fn in glob.glob(os.path.join(d, "*-s*.csv")):
        name = os.path.basename(fn).rsplit("-s", 1)[0]
        with open(fn) as f:
            for l in f:
                if l.strip():
                    i, n = l.split(",")
                    res.setdefault(name, {})[int(i)] = int(n)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="off", help="reference config for ratios and blow-ups")
    ap.add_argument("--base", required=True, help="config the totals are compared with (bootstrap)")
    ap.add_argument("--sample", help="sample csv (index, non_pawn_material) for the phase table")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    res = load(a.dir)
    idx = sorted(set.intersection(*(set(v) for v in res.values())))
    # no ratio for a position skipped by the perf test (0 nodes) or over the time limit (-1)
    none = {i for i in idx if any(v[i] <= 0 for v in res.values())}
    idx = [i for i in idx if i not in none]
    print(f"{len(res)} configs, {len(idx)} positions done by all, {len(none)} without node count left out: {sorted(none)}")
    ref, base = res[a.ref], res[a.base]
    rng = random.Random(1)
    boots = [[rng.choice(idx) for _ in idx] for _ in range(a.boot)]
    tref = sum(ref[i] for i in idx)
    tbase = sum(base[i] for i in idx)
    print(f"{'config':12s} {'total':>13s} {'vs ' + a.ref:>9s} {'vs ' + a.base:>9s} {'90% range':>17s} {'below':>6s}"
          f" {'geomean':>8s} {'>2x':>5s} {'their share':>11s}")
    for name in sorted(res, key=lambda k: sum(res[k][i] for i in idx)):
        r = res[name]
        tot = sum(r[i] for i in idx)
        bs = sorted(sum(r[i] for i in s) / sum(base[i] for i in s) - 1 for s in boots)
        below = sum(x < 0 for x in bs) / len(bs)
        geo = math.exp(sum(math.log(r[i] / ref[i]) for i in idx) / len(idx)) - 1
        ex = [i for i in idx if r[i] > 2 * ref[i]]
        print(f"{name:12s} {tot:13d} {tot / tref - 1:+9.1%} {tot / tbase - 1:+9.1%}"
              f" {bs[len(bs) // 20]:+8.1%}..{bs[-len(bs) // 20 - 1]:+7.1%} {below:6.0%} {geo:+8.1%} {len(ex):5d}"
              f" {sum(r[i] for i in ex) / tot:11.1%}")

    if a.sample:
        npm = {}
        with open(a.sample) as f:
            for row in csv.DictReader(f):
                npm[int(row["index"])] = int(row["non_pawn_material"])
        phases = [("opening/middle (npm >= 50)", 50, 99), ("middle (30-49)", 30, 49), ("late middle (14-29)", 14, 29),
                  ("endgame (<= 13)", 0, 13)]
        print(f"\nby phase: total vs {a.base} (geomean vs {a.ref})")
        print(f"{'config':12s}" + "".join(f" {p[0]:>27s}" for p in phases))
        for name in sorted(res):
            r = res[name]
            cells = []
            for _, lo, hi in phases:
                ii = [i for i in idx if lo <= npm.get(i, -1) <= hi]
                if not ii:
                    cells.append(f" {'-':>27s}")
                    continue
                t = sum(r[i] for i in ii) / sum(base[i] for i in ii) - 1
                g = math.exp(sum(math.log(r[i] / ref[i]) for i in ii) / len(ii)) - 1
                cells.append(f" {t:+10.1%} ({g:+6.1%}, n={len(ii):4d})")
            print(f"{name:12s}" + "".join(cells))


if __name__ == "__main__":
    main()
