"""LMR level of the adaptive LMR by game phase, from the trace in the engine logs (Barbarossa lmrt 38658b1 or later, info
level, -l 2). Phase by the pieces other than pawns and kings in the root position (logged once per search, NoEval line
at draft 1). Node weighted mean level per phase, overall and at equal draft (the phases reach different drafts).
lmr_phase.py "<glob of barbarossa-*.log>" ...
"""
import glob
import re
import sys
from collections import defaultdict

chg = re.compile(r"LMR change: draft (\d+) nodes (\d+) ply \d+ pen \d+ lv (\d+) (\d+)")
start = re.compile(r"searchTheTree starts draft (\d+)")
noeval = re.compile(r"NoEval \|([^ |]+) ")
nodes = re.compile(r"\[Info\]: Nodes: (\d+),")

PHASES = [("13-14", 13, 14), ("9-12", 9, 12), ("5-8", 5, 8), ("0-4", 0, 4)]


def phase_of(board):
    n = sum(1 for c in board if c in "nbrqNBRQ")
    for name, lo, hi in PHASES:
        if lo <= n <= hi:
            return name
    return "?"


iters = []      # (phase, draft, [(node, level)], end nodes)
for pat in sys.argv[1:]:
    for fn in glob.glob(pat):
        ph = "?"
        cur = None
        for line in open(fn, encoding="utf-8", errors="replace"):
            m = start.search(line)
            if m:
                if cur:
                    iters.append((ph, cur[0], cur[1], max(n for n, _ in cur[1])))
                cur = (int(m.group(1)), [(0, 8)])
                continue
            m = noeval.search(line)
            if m and cur and cur[0] == 1:
                ph = phase_of(m.group(1))
                continue
            if cur is None:
                continue
            m = chg.search(line)
            if m:
                cur[1].append((int(m.group(2)), int(m.group(4))))
                continue
            m = nodes.search(line)
            if m:
                iters.append((ph, cur[0], cur[1], int(m.group(1))))
                cur = None


def dgroup(d):
    return "d<=8" if d <= 8 else ("d9-12" if d <= 12 else "d13+")


acc = defaultdict(lambda: [0, 0])          # (phase, group) -> [level*nodes, nodes]
searches = defaultdict(int)
for ph, d, seg, end in iters:
    for (n0, lv), (n1, _) in zip(seg, seg[1:] + [(end, None)]):
        w = max(0, n1 - n0)
        for g in (dgroup(d), "all"):
            acc[(ph, g)][0] += lv * w
            acc[(ph, g)][1] += w
    if d == 1:
        searches[ph] += 1
print("Pieces (no pawns, kings) | searches | node weighted mean level: all, d<=8, d9-12, d13+ | nodes (M)")
for name, _, _ in PHASES:
    row = []
    for g in ("all", "d<=8", "d9-12", "d13+"):
        s, w = acc[(name, g)]
        row.append("%5.2f" % (s / w) if w else "    -")
    print("  %-6s %6d   %s   %8.1f" % (name, searches[name], "  ".join(row), acc[(name, "all")][1] / 1e6))
