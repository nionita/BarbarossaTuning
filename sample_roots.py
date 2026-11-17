"""Random sample of root search positions from Barbarossa's games, for search measurements (node scans) with
the position distribution of real games (all game phases).

Input: data files of Barbarossa -d (see uci_data.py), lines
    S|ply|start|root fen|root score|leaf score|leaf fen|best move
Every search of every game is one candidate; the sample is uniform over all of them (so over games and game
phases as they occur), without duplicate positions (same placement, side, castling and en passant) and
without positions whose root score is a mate score (|score| >= --max-score): they are decided, the pruning
conditions (not near mate) do not apply, and their search can still be expensive.

Output: <out>.fen (one root fen per line, standard FEN) and <out>.csv (index,source,ply,root_score,
non_pawn_material,fen) for the analysis by game phase (non pawn material of both sides, N = B = 3, R = 5,
Q = 9: 62 at the start).

Example:
    python sample_roots.py --data C:\\Learn\\Data\\raw\\games\\ug-a\\eng C:\\Learn\\Data\\raw\\games\\ug-d\\eng
        --count 2000 --seed 1 --out C:\\Learn\\Tune\\hvar\\nodescan\\roots-2000
"""
import argparse
import glob
import os
import random

NPM = {"n": 3, "b": 3, "r": 5, "q": 9}


def std_fen(f):
    """Barbarossa before sfen writes a '/' after the placement: make it standard."""
    return f.replace("/ ", " ", 1)


def key(fen):
    """Position identity without the move counters."""
    return " ".join(fen.split()[:4])


def non_pawn_material(fen):
    return sum(NPM.get(c.lower(), 0) for c in fen.split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="folders with *.data files")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-score", type=int, default=10000, help="drop roots with |root score| >= this (mates)")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()

    cands = []
    for d in a.data:
        for fn in sorted(glob.glob(os.path.join(d, "*.data"))):
            src = os.path.join(os.path.basename(os.path.dirname(os.path.normpath(d))), os.path.basename(d),
                               os.path.basename(fn))
            with open(fn) as f:
                for l in f:
                    w = l.rstrip("\n").split("|")
                    if w[0] == "S" and len(w) >= 5 and w[3] and w[4] and abs(int(w[4])) < a.max_score:
                        cands.append((src, int(w[1]), int(w[4]), std_fen(w[3])))
    print(f"{len(cands)} searches in {len(a.data)} folders")
    rng = random.Random(a.seed)
    rng.shuffle(cands)
    seen, out = set(), []
    for c in cands:
        k = key(c[3])
        if k not in seen:
            seen.add(k)
            out.append(c)
            if len(out) == a.count:
                break
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".fen", "w") as f:
        f.writelines(c[3] + "\n" for c in out)
    with open(a.out + ".csv", "w") as f:
        f.write("index,source,ply,root_score,non_pawn_material,fen\n")
        for i, (src, ply, sc, fen) in enumerate(out):
            f.write(f"{i},{src},{ply},{sc},{non_pawn_material(fen)},{fen}\n")
    npm = sorted(non_pawn_material(c[3]) for c in out)
    print(f"{len(out)} positions -> {a.out}.fen / .csv; non pawn material quartiles "
          f"{npm[len(npm) // 4]} / {npm[len(npm) // 2]} / {npm[3 * len(npm) // 4]} (start 62)")


if __name__ == "__main__":
    main()
