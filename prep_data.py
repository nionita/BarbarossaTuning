#!/usr/bin/env python3
"""Preprocess chilo-g4 CSV files into TuneSGD training files.

Input lines:  fen,score,result      (result -1/0/1, side-to-move POV, one header line per file)
Output lines: fen,score,result+1    (result 0/1/2, as expected by TuneSGD input type 2)

Steps (single streaming pass, stdlib only):
  1. skip header lines and malformed lines
  2. deterministic hash sampling on the position key (first 4 FEN fields): a position is
     either selected with all its duplicates or not at all, so the dedup only needs to
     look at the sampled part and no global sort is necessary
  3. keep positions with gamePhase >= --phase-min (default 120 = mid game, same rule as TuneSGD -g;
     0 keeps all phases, so the end values of the weights are determined by the data)
  4. dedup on the position key (first occurrence wins)
  5. random subsample to --records, shuffle, write --files files named *-filt.csv

With --natural there is no dedup and no hash sampling by position: every valid line is kept with
probability --rate (a uniform sample of the lines, duplicates stay), so the output has the distribution
in which the positions occurred in the games (repeated positions are repeated records).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import sys
import time

PHASE_MIN_DEFAULT = 120
PIECE_WEIGHT = {"q": 39, "r": 20, "b": 12, "n": 12}


def game_phase(placement: str) -> int:
    """Same formula as Eval.Core.gamePhase: both colours counted, kings and pawns ignored."""
    low = placement.lower()
    return sum(low.count(p) * w for p, w in PIECE_WEIGHT.items())


def key_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("ascii", "replace"), digest_size=8).digest(), "little")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="Z:/nicu/extract/2026/chilo/chilo-g4", help="input directory")
    ap.add_argument("--glob", default="chilo-*.csv", help="input file pattern in --src")
    ap.add_argument("--out", default="C:/Learn/Tune/cxig/data", help="output directory")
    ap.add_argument("--records", type=int, default=2_000_000, help="total records to write")
    ap.add_argument("--files", type=int, default=10, help="number of output files")
    ap.add_argument("--rate", type=float, default=0.06,
                    help="fraction of distinct positions kept in the first stage (must be well above "
                         "records / distinct mid game positions)")
    ap.add_argument("--seed", type=int, default=1, help="seed of the final subsample and shuffle")
    ap.add_argument("--phase-min", type=int, default=PHASE_MIN_DEFAULT,
                    help="minimum game phase of a kept position (0 = all phases; start position is 254)")
    ap.add_argument("--prefix", default="cxig", help="output files are <prefix>-NN-filt.csv")
    ap.add_argument("--natural", action="store_true",
                    help="no dedup: uniform sample of the lines, --rate = probability per valid line")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.rate <= 1:
        print("ERROR: --rate must be in (0, 1]", file=sys.stderr)
        return 2
    files = sorted(glob.glob(os.path.join(args.src, args.glob)))
    if not files:
        print(f"ERROR: no input files {args.glob} in {args.src}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    threshold = int(args.rate * (1 << 64))
    kept: dict[int, str] = {}      # key hash -> output line (without newline)
    nat: list[str] = []            # --natural: all sampled lines
    line_rng = random.Random(args.seed + 1000)
    lines = bad = sampled = early = conflicts = 0
    t0 = time.time()

    for fi, path in enumerate(files, 1):
        with open(path, "r", encoding="ascii", errors="replace", newline="") as fh:
            for raw in fh:
                lines += 1
                line = raw.rstrip("\r\n")
                parts = line.rsplit(",", 2)
                if len(parts) != 3:
                    bad += 1
                    continue
                fen, score, res = parts
                fields = fen.split(" ")
                if len(fields) < 4 or fields[1] not in ("w", "b") or res not in ("-1", "0", "1"):
                    bad += 1               # includes the header line
                    continue
                if args.natural:
                    try:
                        int(score)
                    except ValueError:
                        bad += 1
                        continue
                    if game_phase(fields[0]) < args.phase_min:
                        early += 1
                        continue
                    if line_rng.random() < args.rate:
                        sampled += 1
                        nat.append(f"{fen},{score},{int(res) + 1}")
                    continue
                key = " ".join(fields[:4])
                h = key_hash(key)
                if h >= threshold:
                    continue
                sampled += 1
                try:
                    int(score)
                except ValueError:
                    bad += 1
                    continue
                if game_phase(fields[0]) < args.phase_min:
                    early += 1
                    continue
                new = f"{fen},{score},{int(res) + 1}"
                old = kept.get(h)
                if old is not None:
                    if old.split(",", 1)[1] != new.split(",", 1)[1]:
                        conflicts += 1
                    continue
                kept[h] = new
        print(f"[{fi}/{len(files)}] {os.path.basename(path)}: lines={lines:,} sampled={sampled:,} "
              f"unique kept={len(kept):,} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nlines read          : {lines:,}")
    print(f"malformed/header    : {bad:,}")
    print(f"hash sampled        : {sampled:,}")
    print(f"dropped, game phase : {early:,}")
    print(f"unique kept          : {len(kept):,}")
    print(f"dups with other s/r : {conflicts:,}")

    pool = nat if args.natural else list(kept.values())
    if len(pool) < args.records:
        print(f"ERROR: only {len(pool):,} candidate records, need {args.records:,}. "
              f"Increase --rate (now {args.rate}).", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    out_lines = pool
    del kept
    if not args.natural:
        out_lines.sort()                   # make the result independent of file/dict order
    rng.shuffle(out_lines)
    out_lines = out_lines[:args.records]

    per = -(-len(out_lines) // args.files)
    for i in range(args.files):
        chunk = out_lines[i * per:(i + 1) * per]
        name = os.path.join(args.out, f"{args.prefix}-{i + 1:02d}-filt.csv")
        with open(name, "w", encoding="ascii", newline="\n") as fo:
            fo.write("\n".join(chunk))
            fo.write("\n")
        print(f"wrote {name}: {len(chunk):,} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
