"""Evaluation of the reverse futility pruning shadow diagnostic (Barbarossa branch rfpdiag: a config with
epRFutD < 0 searches every node as without RFP and records the RFP decisions; SelfPlay -P prints them).

Input: the outputs of one or more perf runs (e.g. shards of a position file), lines
    RFPH|depth|excess per depth|correct|wrong     histogram of all decisions (excess = static score - beta)
    RFPW|fen|depth|beta|static                    a wrong decision (the search failed low)
    RFPC|fen|depth|beta|static                    a correct decision of the control sample (1 of 256 positions)
As the shadow mode does not change the search, a run with a small margin (e.g. epRFutM = 10) gives the
decisions of every larger margin M: those with excess per depth >= M.

A) wrong rate by margin and depth (histogram, exact)
B) at one margin (--margin): features of the nodes of the wrong decisions against the control sample, as
   wrong rate with and without the feature (control weighted by the sample rate), overall and by game phase

Example:
    python rfp_shadow_report.py C:\\Learn\\Tune\\hvar\\nodescan\\diag\\shadow-a.txt ...\\shadow-b.txt --margin 50
"""
import argparse

import chess

SAMPLE = 256
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def hanging(b, color):
    """Pieces of color (no king) attacked and undefended or attacked by a cheaper piece."""
    n = 0
    for sq, pc in b.piece_map().items():
        if pc.color != color or pc.piece_type == chess.KING:
            continue
        att = b.attackers(not color, sq)
        if att and (not b.attackers(color, sq) or min(VAL[b.piece_at(a).piece_type] for a in att) < VAL[pc.piece_type]):
            n += 1
    return n


def forks(b, color):
    """Pieces of color attacking 2+ enemy pieces (no pawns) which are more valuable or undefended."""
    n = 0
    for sq, pc in b.piece_map().items():
        if pc.color != color:
            continue
        t = 0
        for s in b.attacks(sq):
            q = b.piece_at(s)
            if q and q.color != color and q.piece_type != chess.PAWN and \
               (VAL[q.piece_type] > VAL[pc.piece_type] or not b.attackers(not color, s)):
                t += 1
        n += t >= 2
    return n


def ring(b, color):
    k = b.king(color)
    sqs = chess.SquareSet(chess.BB_KING_ATTACKS[k]) | chess.SquareSet.from_square(k)
    return sum(len(b.attackers(not color, s)) for s in sqs)


def passers(b, color, minrank):
    n = 0
    for sq in b.pieces(chess.PAWN, color):
        r = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
        if r < minrank:
            continue
        f = chess.square_file(sq)
        rr = range(chess.square_rank(sq) + 1, 8) if color == chess.WHITE else range(0, chess.square_rank(sq))
        if not any(b.piece_at(chess.square(ff, x)) == chess.Piece(chess.PAWN, not color)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for x in rr):
            n += 1
    return n


def opp_checks(b):
    c = b.copy()
    c.push(chess.Move.null())
    return any(c.gives_check(m) for m in c.legal_moves)


def npm(b):
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))


FEATS = {
    "own hanging >= 1": lambda b: hanging(b, b.turn) >= 1,
    "own hanging >= 2": lambda b: hanging(b, b.turn) >= 2,
    "opp hanging >= 1": lambda b: hanging(b, not b.turn) >= 1,
    "opp fork on own pieces": lambda b: forks(b, not b.turn) >= 1,
    "own fork on opp pieces": lambda b: forks(b, b.turn) >= 1,
    "own pinned piece": lambda b: any(b.is_pinned(b.turn, s) for s, p in b.piece_map().items()
                                      if p.color == b.turn and p.piece_type != chess.KING),
    "opp attacks own king ring >= 3": lambda b: ring(b, b.turn) >= 3,
    "own attacks opp king ring >= 3": lambda b: ring(b, not b.turn) >= 3,
    "opp passer on rank 5-7": lambda b: passers(b, not b.turn, 4) >= 1,
    "opp passer on rank 6/7": lambda b: passers(b, not b.turn, 5) >= 1,
    "own passer on rank 6/7": lambda b: passers(b, b.turn, 5) >= 1,
    "opp has checking moves": opp_checks,
    "own has checking moves": lambda b: any(b.gives_check(m) for m in b.legal_moves),
    "queens on board": lambda b: len(b.pieces(chess.QUEEN, True)) + len(b.pieces(chess.QUEEN, False)) > 0,
}
PHASES = [("npm >= 30", 30, 99), ("npm 14-29", 14, 29), ("npm <= 13", 0, 13)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--margin", type=int, default=50, help="margin (cp per depth) of part B")
    a = ap.parse_args()

    hist, wrong, ctrl = {}, [], []
    for fn in a.files:
        with open(fn) as f:
            for l in f:
                w = l.rstrip("\n").split("|")
                if w[0] == "RFPH":
                    k = (int(w[1]), int(w[2]))
                    c, x = hist.get(k, (0, 0))
                    hist[k] = (c + int(w[3]), x + int(w[4]))
                elif w[0] in ("RFPW", "RFPC"):
                    rec = (w[1].replace("/ ", " "), int(w[2]), int(w[3]), int(w[4]))
                    (wrong if w[0] == "RFPW" else ctrl).append(rec)
    depths = sorted({d for d, _ in hist})

    print("A) wrong decisions (search failed low) in % of the decisions, by margin M (cp per depth) and depth")
    print("   M     all " + "".join(f"     d{d}" for d in depths) + "    decisions")
    for M in (10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150, 200):
        row, tc, tw = [], 0, 0
        for d in depths:
            c = sum(v[0] for (dd, q), v in hist.items() if dd == d and q >= M)
            w = sum(v[1] for (dd, q), v in hist.items() if dd == d and q >= M)
            row.append(f"{100 * w / (c + w):7.3f}" if c + w else "      -")
            tc += c
            tw += w
        print(f"{M:4d} {100 * tw / max(1, tc + tw):7.3f} " + "".join(row) + f" {tc + tw:12d}")

    def sel(recs):
        # features once per node: {feature: bool}, plus the non pawn material
        out = []
        for r in recs:
            if (r[3] - r[2]) // r[1] >= a.margin:
                b = chess.Board(r[0])
                fs = {k: f(b) for k, f in FEATS.items()}
                fs["_npm"] = npm(b)
                out.append(fs)
        return out

    W, C = sel(wrong), sel(ctrl)
    groups = [("all", 0, 99)] + PHASES
    for gname, lo, hi in groups:
        Wg = [fs for fs in W if lo <= fs["_npm"] <= hi]
        Cg = [fs for fs in C if lo <= fs["_npm"] <= hi]
        if not Wg or not Cg:
            continue
        print(f"\nB) margin M{a.margin}, {gname}: {len(Wg)} wrong decisions, {len(Cg)} sampled correct ones "
              f"(x{SAMPLE}); wrong rate {100 * len(Wg) / (len(Wg) + SAMPLE * len(Cg)):.3f}%")
        print(f"   {'feature':32s} {'wrong: share':>12s} {'correct: share':>15s} {'rate with':>10s} {'without':>8s}"
              f" {'ratio':>6s}")
        for k in FEATS:
            w1 = sum(fs[k] for fs in Wg)
            c1 = sum(fs[k] for fs in Cg)
            w0, c0 = len(Wg) - w1, len(Cg) - c1
            r1 = w1 / (w1 + SAMPLE * c1) if w1 + c1 else float("nan")
            r0 = w0 / (w0 + SAMPLE * c0) if w0 + c0 else float("nan")
            print(f"   {k:32s} {100 * w1 / len(Wg):11.1f}% {100 * c1 / len(Cg):14.1f}% "
                  f"{100 * r1:9.3f}% {100 * r0:7.3f}% {r1 / r0 if r0 else float('nan'):6.1f}")


if __name__ == "__main__":
    main()
