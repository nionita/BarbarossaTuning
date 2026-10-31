"""Paired beta cut statistics of parameter variants against a base: chunks of positions, one process per chunk (S2).
hist_scan.py <exe> <fen> <depth> <outdir> <base params> <params> ...   ("-" = no parameters)"""
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import sys

exe, fen, depth, out = sys.argv[1:5]
variants = sys.argv[5:]
CHUNK, WORKERS = 25, 4
os.makedirs(out, exist_ok=True)
# SelfPlay -P ignores -s and -f: one file per chunk
fens = [l for l in open(fen) if l.strip()]
chunks = list(range(0, len(fens), CHUNK))
for s in chunks:
    with open(os.path.join(out, "chunk-%d.fen" % s), "w") as f:
        f.writelines(fens[s:s + CHUNK])
RE = re.compile(r"Beta cuts:\s*(\d+) at move 1: ([\d.]+)%, 2: ([\d.]+)%, 3: ([\d.]+)%, 4\+: ([\d.]+)%")


def run(v, s):
    fn = os.path.join(out, "%s-%d.json" % (re.sub(r"[^\w]", "_", v), s))
    if os.path.exists(fn):
        return v, s, json.load(open(fn))
    cmd = [exe, "-P", "-i", os.path.abspath(os.path.join(out, "chunk-%d.fen" % s)), "-d", depth]
    if v != "-":
        cmd += ["-p", v]
    t = subprocess.run(cmd, cwd=out, text=True, capture_output=True).stdout
    m = RE.search(t)
    r = {"nodes": int(re.search(r"Total nodes:\s*(\d+)", t).group(1)), "cuts": int(m.group(1)),
         "m1": float(m.group(2)), "m4": float(m.group(5))}
    json.dump(r, open(fn, "w"))
    return v, s, r


res = {}
with cf.ThreadPoolExecutor(WORKERS) as ex:
    for v, s, r in ex.map(lambda a: run(*a), [(v, s) for s in chunks for v in variants]):
        res[(v, s)] = r


def stat(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0
    return m, sd / math.sqrt(len(xs))


base = variants[0]
print("%-26s %11s %7s %7s  %15s %15s %15s" % ("variant", "nodes", "move1", "move4+", "late d +-se", "4+ d +-se", "geo nodes% +-se"))
for v in variants:
    tot = sum(res[(v, s)]["nodes"] for s in chunks)
    cuts = sum(res[(v, s)]["cuts"] for s in chunks)
    m1 = sum(res[(v, s)]["m1"] * res[(v, s)]["cuts"] for s in chunks) / cuts
    m4 = sum(res[(v, s)]["m4"] * res[(v, s)]["cuts"] for s in chunks) / cuts
    dl = stat([(100 - res[(v, s)]["m1"]) - (100 - res[(base, s)]["m1"]) for s in chunks])
    d4 = stat([res[(v, s)]["m4"] - res[(base, s)]["m4"] for s in chunks])
    dn = stat([100 * math.log(res[(v, s)]["nodes"] / res[(base, s)]["nodes"]) for s in chunks])
    print("%-26s %11d %7.2f %7.2f  %+7.3f %6.3f %+7.3f %6.3f %+7.1f %6.1f" % (v, tot, m1, m4, dl[0], dl[1], d4[0], d4[1], dn[0], dn[1]))
