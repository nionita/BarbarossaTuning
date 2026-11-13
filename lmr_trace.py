"""Parse the LMR level trace (Barbarossa lmrt 38658b1 or later, engine logs at info level, -l 2): node weighted level
distribution by draft, steps up and down, penalty depth of the drops, recovery.
lmr_trace.py "<glob of barbarossa-*.log>"
"""
import glob
import re
import sys
from collections import Counter, defaultdict

chg = re.compile(r"LMR change: draft (\d+) nodes (\d+) ply (\d+) pen (\d+) lv (\d+) (\d+) hi (\d+) (\d+)")
start = re.compile(r"searchTheTree starts draft (\d+)")
nodes = re.compile(r"\[Info\]: Nodes: (\d+),")

iters = []          # (draft, [(node, level)...], end nodes, complete)
cur = None


def close(end, complete):
    global cur
    if cur is not None:
        iters.append((cur[0], cur[1], end, complete))
    cur = None


for fn in glob.glob(sys.argv[1]):
    for line in open(fn, encoding="utf-8", errors="replace"):
        m = start.search(line)
        if m:
            if cur is not None:     # previous iteration had no statistics: aborted
                close(max(n for n, _ in cur[1]), False)
            cur = (int(m.group(1)), [(0, 8)], [])
            continue
        if cur is None:
            continue
        m = chg.search(line)
        if m:
            d, n, ply, pen, l0, l1, h0, h1 = map(int, m.groups())
            cur[1].append((n, l1))
            cur[2].append((n, ply, pen, l0, l1, h0, h1))
            continue
        m = nodes.search(line)
        if m:
            close(int(m.group(1)), True)
    close(max(n for n, _ in cur[1]) if cur else 0, False)

# node weighted level distribution, by draft group
wt = defaultdict(Counter)
endlv = defaultdict(Counter)
for d, seg, end, complete in iters:
    if not complete:
        continue
    g = "d<=8" if d <= 8 else ("d9-12" if d <= 12 else "d13+")
    for (n0, lv), (n1, _) in zip(seg, seg[1:] + [(end, None)]):
        wt[g][lv] += max(0, n1 - n0)
        wt["all"][lv] += max(0, n1 - n0)
    endlv[g][seg[-1][1]] += 1

print("Node weighted level distribution (complete iterations), % of nodes:")
for g in ("d<=8", "d9-12", "d13+", "all"):
    tot = sum(wt[g].values())
    if not tot:
        continue
    mean = sum(l * w for l, w in wt[g].items()) / tot
    row = " ".join("%2d:%4.1f" % (l, 100 * wt[g][l] / tot) for l in range(16) if wt[g][l])
    print("  %-6s nodes %11d mean %.2f | %s" % (g, tot, mean, row))
print("Level at the end of complete iterations:")
for g in ("d<=8", "d9-12", "d13+"):
    print("  %-6s %s" % (g, dict(sorted(endlv[g].items()))))

# events: ups, downs, penalty depth of downs, drops in a row
ups = downs = 0
pens = Counter()
runs = Counter()
recov = []          # nodes from a drop until the level is back where it was before
for d, seg, end, complete in iters:
    run = 0
    lvls = seg
    for i, (n, lv) in enumerate(seg[1:], 1):
        if lv < lvls[i - 1][1]:
            downs += 1
            run += 1
            # recovery
            target = lvls[i - 1][1]
            rec = None
            for n2, lv2 in seg[i + 1:]:
                if lv2 >= target:
                    rec = n2 - n
                    break
            recov.append(rec)
        elif lv > lvls[i - 1][1]:
            ups += 1
            if run:
                runs[run] += 1
            run = 0
    if run:
        runs[run] += 1
# penalty depths from the raw events
for fn in glob.glob(sys.argv[1]):
    for line in open(fn, encoding="utf-8", errors="replace"):
        m = chg.search(line)
        if m:
            d, n, ply, pen, l0, l1, h0, h1 = map(int, m.groups())
            if l1 < l0:
                pens[pen] += 1

ndone = sum(1 for r in recov if r is not None)
print("Iterations: %d (complete %d); level steps up %d, down %d" % (len(iters), sum(1 for i in iters if i[3]), ups, downs))
print("Drops in a row (without a step up between):", dict(sorted(runs.items())))
print("Penalty depth of the drops:", dict(sorted(pens.items())))
rs = sorted(r for r in recov if r is not None)
if rs:
    print("Recovery to the level before a drop: %d of %d within the iteration; nodes median %d, 75%% %d, 90%% %d"
          % (ndone, len(recov), rs[len(rs) // 2], rs[3 * len(rs) // 4], rs[9 * len(rs) // 10]))
