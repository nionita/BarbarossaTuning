"""SelfPlay --data files -> training and validation files for TuneSGD -X / hv_fit.py.

Input: <src>/<run>-s<i>-filt.csv (fen,score,result 0/1/2 from the leaf's side to move) with the
.meta files (game number per row). Whole shards given by --val are the validation data (game
disjoint from the rest); their games are split by game number parity into two files: va-01
(selection) and va-02 (test). The other shards are distributed by game over --files training files.
Positions are kept as they occurred (no dedup: the distribution of the engine's own games).
Pure pawn endings and positions with |score| > --max-score are dropped (like the chilo data).

Example:
    python hv_prep_sp.py --src C:\\Learn\\Tune\\hvar\\sp-data --val gen10k-b-s4 --out C:\\Learn\\Tune\\hvar\\sp10k
    -> sp10k\\data-train\\tr-01-filt.csv .. and sp10k\\data-val\\va-01-filt.csv, va-02-filt.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

PIECE_WEIGHT = {"q": 39, "r": 20, "b": 12, "n": 12}


def game_phase(placement: str) -> int:
    low = placement.lower()
    return sum(low.count(p) * w for p, w in PIECE_WEIGHT.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory with the SelfPlay --data files")
    ap.add_argument("--val", required=True, help="comma separated shard names (e.g. gen10k-b-s4) for validation")
    ap.add_argument("--out", required=True, help="output directory (data-train, data-val are created in it)")
    ap.add_argument("--files", type=int, default=8, help="number of training files")
    ap.add_argument("--phase-min", type=int, default=1, help="minimum game phase (1 = drop pure pawn endings)")
    ap.add_argument("--max-score", type=int, default=2000, help="drop positions with |score| above")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*-filt.csv")))
    if not files:
        print(f"No *-filt.csv in {args.src}", file=sys.stderr)
        return 2
    val_names = {v.strip() for v in args.val.split(",") if v.strip()}
    shard = {f: os.path.basename(f)[:-len("-filt.csv")] for f in files}
    missing = val_names - set(shard.values())
    if missing:
        print(f"Validation shards not found: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    dtrain = os.path.join(args.out, "data-train")
    dval = os.path.join(args.out, "data-val")
    os.makedirs(dtrain, exist_ok=True)
    os.makedirs(dval, exist_ok=True)
    tr = [open(os.path.join(dtrain, f"tr-{i + 1:02d}-filt.csv"), "w", newline="\n") for i in range(args.files)]
    va = [open(os.path.join(dval, f"va-{i + 1:02d}-filt.csv"), "w", newline="\n") for i in range(2)]
    ntr = [0] * args.files
    nva = [0, 0]
    dropped = 0
    gtrain = 0
    for f in files:
        is_val = shard[f] in val_names
        meta = f + ".meta"
        if not os.path.exists(meta):
            print(f"Missing {meta}", file=sys.stderr)
            return 2
        with open(f) as fd, open(meta) as fm:
            fm.readline()                         # header
            last_game = None
            out_i = 0
            for line, mline in zip(fd, fm):
                line = line.rstrip("\n")
                fen, score, res = line.rsplit(",", 2)
                game = int(mline.split(",", 1)[0])
                if game != last_game:
                    last_game = game
                    if is_val:
                        out_i = game % 2
                    else:
                        out_i = gtrain % args.files
                        gtrain += 1
                if game_phase(fen.split(" ", 1)[0]) < args.phase_min or abs(int(score)) > args.max_score:
                    dropped += 1
                    continue
                if is_val:
                    va[out_i].write(line + "\n")
                    nva[out_i] += 1
                else:
                    tr[out_i].write(line + "\n")
                    ntr[out_i] += 1
        print(f"{os.path.basename(f)}: {'validation' if is_val else 'training'}", flush=True)
    for h in tr + va:
        h.close()
    print(f"training {sum(ntr):,} positions in {args.files} files ({gtrain:,} games), "
          f"validation {nva[0]:,} + {nva[1]:,}, dropped {dropped:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
