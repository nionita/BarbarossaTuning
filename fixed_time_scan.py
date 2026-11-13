"""Fixed time searches of one engine with several parameter settings on the same positions, all settings in parallel
(so a throttling CPU slows them alike); the search is stopped after the move time (the engine ignores movetime).
Per setting: completed depth, nodes and time of the last full-depth info line; the engine logs (-l 2) go to <out>/<label>/ for further parsing (e.g. lmr_trace.py).
fixed_time_scan.py <engine> <fen file> <n positions> <movetime ms> <out dir> <label=params|-> ...
"""
import os
import re
import subprocess
import sys
import threading
import time

eng, fenfile, npos, mt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
settings = [a.split("=", 1) for a in sys.argv[6:]]
fens = [l.split(" ")[:4] for l in open(fenfile) if l.strip()][:npos]
fens = [" ".join(f) + " 0 1" for f in fens]
info = re.compile(r"\bdepth (\d+) .*\btime (\d+) nodes (\d+)")
res = {}


def run(label, params):
    d = os.path.join(out, label)
    os.makedirs(d, exist_ok=True)
    cmd = [eng, "-l", "2"] + ([] if params == "-" else ["-p", params])
    p = subprocess.Popen(cmd, cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def wait_for(prefix):
        last = None
        for line in p.stdout:
            m = info.search(line)
            if m:
                last = tuple(map(int, m.groups()))
            if line.startswith(prefix):
                return last

    send("uci")
    wait_for("uciok")
    rows = []
    for f in fens:
        send("ucinewgame")
        send("isready")
        wait_for("readyok")
        send("position fen " + f)
        # Barbarossa ignores "go movetime": search without limit, stop after the move time
        send("go infinite")
        time.sleep(mt / 1000)
        send("stop")
        rows.append(wait_for("bestmove"))
    send("quit")
    p.wait()
    res[label] = rows


ts = [threading.Thread(target=run, args=(lab, par)) for lab, par in settings]
for t in ts:
    t.start()
for t in ts:
    t.join()

base = settings[0][0]
print("movetime %d ms, %d positions; depth = last completed depth, d-base paired against %s" % (mt, len(fens), base))
print("%-8s %7s %8s %9s %10s %9s" % ("setting", "depth", "d-base", "+-se", "nodes", "time ms"))
for lab, _ in settings:
    rows = res[lab]
    ok = [i for i in range(len(fens)) if rows[i] and res[base][i]]
    dep = sum(rows[i][0] for i in ok) / len(ok)
    diffs = [rows[i][0] - res[base][i][0] for i in ok]
    md = sum(diffs) / len(diffs)
    se = (sum((x - md) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5 / len(diffs) ** 0.5
    nodes = sum(rows[i][2] for i in ok) / len(ok)
    tm = sum(rows[i][1] for i in ok) / len(ok)
    print("%-8s %7.2f %+8.3f %9.3f %10.0f %9.0f" % (lab, dep, md, se, nodes, tm))
