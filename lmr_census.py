"""Tables of the LMR census (Barbarossa SelfPlay built with the flag lmrcensus, branch lmcs): reads the "census,key,n"
lines of one or more `SelfPlay -P` outputs and prints, per output (name = file name without extension):

1. by level: reduced searches, share with d' = 1 and d' <= 2, share pulled into the static range (d1 >= 7, d' <= 6),
   mean reduction;
2. by level and d1 class: share of the zero window searches of zeroWithLMR that are reduced, mean reduction;
3. by level, d1 class and w class: reduced searches, fail high / reduced, confirmed / reduced, confirmed / fail high.

Key = kind * 10000 + level * 100 + a * 10 + b (Search.Core census). Kind 1: (d1 class, d' class) reduced; 2, 3, 4:
(d1 class, w class) reduced, failed high, confirmed; 5: (d1 class) not reduced; 6: (d1 class) sum of reductions.
Level 0 = below 8. Usage: lmr_census.py <out.txt> ...
"""
import os
import re
import sys
from collections import defaultdict

D1 = ["2-3", "4-6", "7-10", "11+"]
DR = ["1", "2", "3-6", "7+"]
WC = ["<=2", "3-5", "6-10", "11+"]


def read(fn):
    c = defaultdict(int)
    with open(fn, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"census,(\d+),(\d+)", line.strip())
            if m:
                k, n = int(m.group(1)), int(m.group(2))
                c[(k // 10000, k // 100 % 100, k // 10 % 10, k % 10)] += n
    return c


def pct(a, b):
    return f"{100 * a / b:6.1f}" if b else "     -"


def report(name, c):
    levels = sorted({lv for (_, lv, _, _) in c})
    print(f"== {name}")
    print("1. by level: reduced, d'=1 %, d'<=2 %, d1>=7 & d'<=6 %, mean reduction")
    for lv in levels:
        red = sum(n for (k, l, a, b), n in c.items() if k == 1 and l == lv)
        r1 = sum(n for (k, l, a, b), n in c.items() if k == 1 and l == lv and b == 0)
        r2 = sum(n for (k, l, a, b), n in c.items() if k == 1 and l == lv and b <= 1)
        st = sum(n for (k, l, a, b), n in c.items() if k == 1 and l == lv and a >= 2 and b <= 2)
        sr = sum(n for (k, l, a, b), n in c.items() if k == 6 and l == lv)
        mr = f"{sr / red:5.2f}" if red else "    -"
        print(f"  lv {lv:2d}: {red:10d} {pct(r1, red)} {pct(r2, red)} {pct(st, red)}  {mr}")
    print("2. by level and d1: reduced % of zero window searches, mean reduction")
    print("         " + "".join(f"{d:>16s}" for d in D1))
    for lv in levels:
        row = []
        for a in range(4):
            red = sum(n for (k, l, x, b), n in c.items() if k == 1 and l == lv and x == a)
            full = c.get((5, lv, a, 0), 0)
            sr = c.get((6, lv, a, 0), 0)
            mr = f"{sr / red:4.1f}" if red else "   -"
            row.append(f"{pct(red, red + full)}% {mr:>6s} ")
        print(f"  lv {lv:2d}: " + "".join(f"{x:>16s}" for x in row))
    print("3. by level, d1, w: reduced, fh/red %, conf/red %, conf/fh %")
    for a in range(4):
        for w in range(4):
            cells = []
            for lv in levels:
                red = c.get((2, lv, a, w), 0)
                fh = c.get((3, lv, a, w), 0)
                cf = c.get((4, lv, a, w), 0)
                if red:
                    cells.append(f"lv{lv}: {red:9d} {pct(fh, red)} {pct(cf, red)} {pct(cf, fh)}")
            if cells:
                print(f"  d1 {D1[a]:>4s} w {WC[w]:>4s} | " + " | ".join(cells))
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for fn in sys.argv[1:]:
        report(os.path.splitext(os.path.basename(fn))[0], read(fn))


if __name__ == "__main__":
    main()
