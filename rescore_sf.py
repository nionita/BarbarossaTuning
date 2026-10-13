#!/usr/bin/env python3
"""Re-score TuneSGD training files with Stockfish (search to a depth, scores of several depths in one pass).

Input lines:  fen,score,result          (result 0/1/2)
Output:       one directory per depth, <out-root>/<prefix>-d<D>/, same file names, lines fen,sfscore,result
              (score from the side to move, cp as printed by Stockfish; mate = +-3000)

One search to the largest depth gives the scores of all smaller depths as well (iterative deepening),
so --depths 1,6,10 costs only the depth 10 search.
Every position starts from a clean state (ucinewgame, Threads 1, small hash): the labels do not depend on
the order of the positions or on the number of workers.

Example:  python rescore_sf.py --src C:\\Learn\\Tune\\cxig\\data-all --out-root C:\\Learn\\Tune\\cxig\\sf \\
              --prefix all-sf18 --depths 1,6,10 --limit 20000 --files 1 --workers 4
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

SF_DEFAULT = r"C:\Users\Nicu\Documents\Engines\stockfish-18\stockfish-windows-x86-64-avx2.exe"
MATE_CP = 3000
INFO_RX = re.compile(r"^info depth (\d+) .*?score (cp|mate) (-?\d+)(?: (lowerbound|upperbound))?")

_sf = None
_depths: list[int] = []


def _start(sf_path: str, depths: list[int], hash_mb: int) -> None:
    global _sf, _depths
    _depths = depths
    _sf = subprocess.Popen([sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, bufsize=1)
    _send(f"setoption name Threads value 1\nsetoption name Hash value {hash_mb}\nisready")
    _wait("readyok")


def _send(text: str) -> None:
    _sf.stdin.write(text + "\n")
    _sf.stdin.flush()


def _wait(token: str) -> None:
    for line in _sf.stdout:
        if line.startswith(token):
            return
    raise RuntimeError("Stockfish ended")


def score_fen(fen: str) -> list[int]:
    """Scores for the configured depths (side to move). A missing depth takes the last smaller one."""
    _send(f"ucinewgame\nposition fen {fen}\ngo depth {_depths[-1]}")
    got: dict[int, int] = {}
    for line in _sf.stdout:
        if line.startswith("bestmove"):
            break
        m = INFO_RX.match(line)
        if m and m.group(4) is None:
            d = int(m.group(1))
            v = int(m.group(3))
            if m.group(2) == "mate":
                v = MATE_CP if v > 0 else -MATE_CP
            got[d] = v
    out = []
    last = 0
    for d in _depths:
        if d in got:
            last = got[d]
        else:
            # use the deepest smaller depth reported, otherwise 0 (terminal position)
            smaller = [x for x in got if x <= d]
            if smaller:
                last = got[max(smaller)]
        out.append(last)
    return out


def _init(sf_path: str, depths: list[int], hash_mb: int) -> None:
    _start(sf_path, depths, hash_mb)


def _job(lines: list[str]) -> list[list[int]]:
    return [score_fen(ln.split(",")[0]) for ln in lines]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="input directory with *-filt.csv files")
    ap.add_argument("--glob", default="*-filt.csv")
    ap.add_argument("--out-root", required=True, help="the directories <prefix>-d<D> are created here")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--depths", default="1,6,10")
    ap.add_argument("--files", type=int, default=0, help="only the first N files (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N lines of every file (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hash", type=int, default=4, help="hash MB per worker")
    ap.add_argument("--sf", default=SF_DEFAULT)
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    depths = sorted({int(x) for x in args.depths.split(",") if x})
    files = sorted(glob.glob(os.path.join(args.src, args.glob)))
    if args.files:
        files = files[: args.files]
    if not files:
        print("no input files", file=sys.stderr)
        return 2
    outs = {d: os.path.join(args.out_root, f"{args.prefix}-d{d}") for d in depths}
    for p in outs.values():
        os.makedirs(p, exist_ok=True)

    t0 = time.time()
    total = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.sf, depths, args.hash)) as pool:
        for path in files:
            with open(path, "r", encoding="ascii", errors="replace", newline="") as fh:
                lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
            if args.limit:
                lines = lines[: args.limit]
            chunks = [lines[i: i + args.chunk] for i in range(0, len(lines), args.chunk)]
            name = os.path.basename(path)
            handles = {d: open(os.path.join(outs[d], name), "w", encoding="ascii", newline="\n") for d in depths}
            try:
                done = 0
                for chunk, res in zip(chunks, pool.imap(_job, chunks)):
                    for ln, sc in zip(chunk, res):
                        fen, _old, result = ln.rsplit(",", 2)
                        for d, v in zip(depths, sc):
                            handles[d].write(f"{fen},{v},{result}\n")
                    done += len(chunk)
                    total += len(chunk)
                    if done % (args.chunk * 25) == 0:
                        el = time.time() - t0
                        print(f"{name}: {done}/{len(lines)}  {total / el:.0f} pos/s", flush=True)
            finally:
                for h in handles.values():
                    h.close()
    el = time.time() - t0
    print(f"done: {total} positions, depths {depths}, {el:.0f} s, {total / el:.0f} pos/s, workers {args.workers}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
