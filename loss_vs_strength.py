#!/usr/bin/env python3
"""How well does each loss variant predict the measured strength of the candidate configs?

For every measured candidate (result.txt of a group / sharded run with its candidate.cfg) the loss of
several loss variants is computed with TuneSGD -e on a position subset; the change against the compiled
weights (in percent) is regressed against the measured Elo (weighted by 1/se^2, through the origin).
Reports slope (Elo per +1% loss), uncentered correlation r and the same for loss decreases only.
"""
import glob, math, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

GROUPS = "C:/Learn/Tune/cxig/groups"
TUNE = "C:/astra/TuneSGD/TuneSGD-tsub.exe"
DATA = "C:/Learn/Tune/cxig/data-all-600k"
WORKERS = 4
EXCLUDE = {"val-n2k"}               # other node count
POOL = {"run4all-h": ["run4all-h", "val-open"]}   # same config measured on two opening sets
SKIP = {"val-open"}

VARIANTS = [  # label, loss function, q, k
    ("L5 q0   k190", 5, 0.0, 190), ("L5 q0.2 k205", 5, 0.2, 205), ("L5 q0.4 k235", 5, 0.4, 235),
    ("L5 q1   k200", 5, 1.0, 200),
    ("L1 k190", 1, 0, 190), ("L1 k300", 1, 0, 300), ("L2 k190", 2, 0, 190), ("L2 k300", 2, 0, 300),
    ("L3 k190", 3, 0, 190), ("L3 k300", 3, 0, 300), ("L4 k190", 4, 0, 190), ("L4 k300", 4, 0, 300),
]

def wdl(path):
    for line in open(path):
        m = re.match(r"WDL:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", line)
        if m: return tuple(int(x) for x in m.groups())
    return None

def elo_se(w, d, l):
    n = w + d + l; s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s * s) / n
    se = 400 / math.log(10) / (s * (1 - s)) * math.sqrt(var / n)
    return -400 * math.log10(1 / s - 1), se

cands = []   # name, cfg, wdl
for d in sorted(os.listdir(GROUPS)):
    res = os.path.join(GROUPS, d, "result.txt")
    if not os.path.isfile(res) or d in EXCLUDE or d in SKIP or re.search(r"-s\d$", d): continue
    cfg = os.path.join(GROUPS, d, "candidate.cfg")
    if not os.path.isfile(cfg): cfg = os.path.join(GROUPS, d + "-s1", "candidate.cfg")
    if not os.path.isfile(cfg): continue
    w = wdl(res)
    if w is None: continue
    for extra in POOL.get(d, [])[1:]:
        w2 = wdl(os.path.join(GROUPS, extra, "result.txt"))
        w = tuple(a + b for a, b in zip(w, w2))
    cands.append((d, cfg, w))
print(len(cands), "candidates", flush=True)

def loss(cfg, var):
    _, lf, q, k = var
    cmd = [TUNE, "-e", "-i", DATA, "-L", str(lf), "-k", str(k), "-q", str(q)]
    if cfg: cmd += ["-c", cfg]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"^Loss:\s*(\S+)", out, re.M)
    return float(m.group(1))

jobs = [(None, v) for v in VARIANTS] + [(c[1], v) for c in cands for v in VARIANTS]
with ThreadPoolExecutor(WORKERS) as ex:
    vals = list(ex.map(lambda j: loss(*j), jobs))
base = {v[0]: vals[i] for i, v in enumerate(VARIANTS)}
table = {}
for i, (cfg, v) in enumerate(jobs[len(VARIANTS):]):
    table[(cfg, v[0])] = vals[len(VARIANTS) + i]

print()
print("%-14s %7s %6s | %s" % ("variant", "slope", "r", "loss-decrease points: n, mean Elo, mean predicted"))
for v in VARIANTS:
    xs, ys, ws = [], [], []
    for name, cfg, w in cands:
        e, se = elo_se(*w)
        xs.append(100 * (table[(cfg, v[0])] - base[v[0]]) / base[v[0]]); ys.append(e); ws.append(1 / se ** 2)
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys)); sxx = sum(w * x * x for w, x in zip(ws, xs))
    syy = sum(w * y * y for w, y in zip(ws, ys))
    slope = sxy / sxx; r = sxy / math.sqrt(sxx * syy)
    dec = [(x, y, w) for x, y, w in zip(xs, ys, ws) if x < -0.02]
    if dec:
        sw = sum(w for _, _, w in dec)
        my = sum(w * y for _, y, w in dec) / sw
        mp = sum(w * slope * x for x, _, w in dec) / sw
        extra = "n=%d  Elo %+.1f  predicted %+.1f" % (len(dec), my, mp)
    else:
        extra = "none"
    print("%-14s %7.1f %6.2f | %s" % (v[0], slope, r, extra))

print()
print("points (loss change % for the first 4 variants, measured Elo):")
for name, cfg, w in cands:
    e, se = elo_se(*w)
    xs = ["%+.2f" % (100 * (table[(cfg, v[0])] - base[v[0]]) / base[v[0]]) for v in VARIANTS[:4]]
    print("  %-34s %s   Elo %+6.1f +- %.1f" % (name, " ".join("%7s" % x for x in xs), e, 1.96 * se))
