"""Bias tables: where does the engine's eval mispredict the results of its games?

Input: TuneSGD -X exports (training and validation directories, e.g. of one on-policy round). The eval
column is calibrated once on the training data (global offset c and scale k, as in hv_fit.py). Then the
positions are grouped by descriptors computed from the FEN (material signature, imbalances, opposite
bishops, pawn structure, phase, halfmove clock, side to move, eval level). For every group an own offset
and scale are fitted on the training data and applied to the validation data: the loss they save there
is an honest bound of what a feature distinguishing these groups could gain (as a constant or a scale in
the group, not as a full new term). Reported per descriptor: the gain in % of the validation loss, and
per group: share, mean residual (result / 2 - prediction, in % points), the offset in cp and k.

Example:
    python hv_bias.py --train C:\\Learn\\Tune\\hvar\\sp3\\exp-train --val C:\\Learn\\Tune\\hvar\\sp3\\exp-val
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import env_dlls  # noqa: F401  (before numpy)
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

COLS = ["fen", "result", "eval", "phase", "m50"]
VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}


def read(path: str, max_score: float) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*-exp.csv"))) if os.path.isdir(path) else [path]
    df = pd.concat([pd.read_csv(f, engine="pyarrow", usecols=COLS + ["score"]) for f in files],
                   ignore_index=True)
    df = df[np.abs(df["score"]) <= max_score].reset_index(drop=True)
    return df


def describe(fen: str) -> dict[str, str]:
    """Descriptors of a position from the side to move's view (own pieces upper case in the keys)."""
    board, stm = fen.split(" ", 2)[:2]
    white_moves = stm == "w"
    cnt = defaultdict(int)
    bishops = {True: [], False: []}         # square colours of the bishops per side (True = white)
    pawn_files = {True: set(), False: set()}
    pawn_sq = {True: [], False: []}
    kings = {}
    rank, file = 7, 0
    for ch in board:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch <= "9":
            file += ord(ch) - 48
        else:
            white = ch < "a"
            p = ch.lower()
            cnt[(white, p)] += 1
            if p == "b":
                bishops[white].append((rank + file) % 2)
            elif p == "p":
                pawn_files[white].add(file)
                pawn_sq[white].append((file, rank))
            elif p == "k":
                kings[white] = (file, rank)
            file += 1
    me, op = white_moves, not white_moves

    def sig(side: bool) -> str:
        return "".join(p.upper() * cnt[(side, p)] for p in "qrbn")

    pieces_me, pieces_op = sig(me), sig(op)
    pd_ = cnt[(me, "p")] - cnt[(op, "p")]
    mat_me = sum(VALUES[p] * cnt[(me, p)] for p in "qrbn")
    mat_op = sum(VALUES[p] * cnt[(op, p)] for p in "qrbn")
    d = {}
    # material signature of the pieces (without pawns) and the pawn difference
    d["pieces"] = f"{pieces_me or '-'} v {pieces_op or '-'}"
    d["pieces+pawns"] = f"{d['pieces']} p{pd_:+d}" if abs(pd_) <= 3 else f"{d['pieces']} p{'+' if pd_ > 0 else '-'}4+"
    # imbalance type: differences of queens, rooks, minors, pawns
    dq = cnt[(me, "q")] - cnt[(op, "q")]
    dr = cnt[(me, "r")] - cnt[(op, "r")]
    dm = cnt[(me, "b")] + cnt[(me, "n")] - cnt[(op, "b")] - cnt[(op, "n")]
    d["imbalance"] = "balanced" if dq == dr == dm == 0 else f"Q{dq:+d} R{dr:+d} m{dm:+d}"
    # opposite coloured bishops (each side exactly one bishop, other pieces listed)
    if cnt[(me, "b")] == 1 and cnt[(op, "b")] == 1:
        ocb = bishops[me][0] != bishops[op][0]
        others = mat_me + mat_op - 6
        d["bishops"] = ("opposite" if ocb else "same") + (" only" if others == 0 else " +pieces")
    else:
        d["bishops"] = "other"
    # pawn structure: both wings or one wing, total pawns
    allf = pawn_files[True] | pawn_files[False]
    wings = ("q" if allf & {0, 1, 2} else "") + ("c" if allf & {3, 4} else "") + ("k" if allf & {5, 6, 7} else "")
    d["pawn wings"] = wings or "none"
    npawns = cnt[(True, "p")] + cnt[(False, "p")]
    d["pawns total"] = str(min(npawns // 2 * 2, 14))
    # passed pawns (own minus opponent), simple definition (no pawn ahead on the same or adjacent files)

    def passed(side: bool) -> int:
        n = 0
        for f, r in pawn_sq[side]:
            ahead = [(ff, rr) for ff, rr in pawn_sq[not side] if abs(ff - f) <= 1 and (rr > r if side else rr < r)]
            n += not ahead
        return n

    pp = passed(me) - passed(op)
    d["passed diff"] = f"{max(min(pp, 3), -3):+d}"
    # doubled pawns (own minus opponent)
    dbl = (cnt[(me, "p")] - len(pawn_files[me])) - (cnt[(op, "p")] - len(pawn_files[op]))
    d["doubled diff"] = f"{max(min(dbl, 2), -2):+d}"
    d["side to move"] = "white" if white_moves else "black"
    return d


def fit_group(e: np.ndarray, t: np.ndarray, c0: float, u0: float) -> tuple[float, float]:
    """Offset (cp) and u = 1 / k of the group, least squares on the sigmoid (start c0, u0)."""
    def f(x):
        c, lu = x[0] * 10.0, x[1]
        u = u0 * np.exp(lu)
        s = expit(u * (e + c))
        r = t - s
        g = -2.0 * r * s * (1 - s)
        return float(np.mean(r * r)) * 1e3, np.array([np.mean(g * u) * 10.0, np.mean(g * u * (e + c))]) * 1e3
    res = minimize(f, np.array([c0 / 10.0, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(-50, 50), (-1.5, 1.5)])
    return res.x[0] * 10.0, u0 * float(np.exp(res.x[1]))


def pool(g_tr: np.ndarray, g_va: np.ndarray, min_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Groups below min_share (%) of the training rows (or not in the training data) become '(other)'."""
    keys, counts = np.unique(g_tr, return_counts=True)
    big = set(keys[counts >= min_share / 100 * len(g_tr)])
    return (np.where(np.isin(g_tr, list(big)), g_tr, "(other)"),
            np.where(np.isin(g_va, list(big)), g_va, "(other)"))


def one_hot(names: list[str], groups: dict) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str]]:
    """One column per group of each descriptor (training and validation rows), and the column labels."""
    mats_tr, mats_va, labels = [], [], []
    for name in names:
        g_tr, g_va = groups[name]
        keys = list(np.unique(g_tr))
        idx = {k: i for i, k in enumerate(keys)}
        for g, mats in ((g_tr, mats_tr), (g_va, mats_va)):
            col = np.array([idx[x] for x in g], dtype=np.int32)
            mats.append(sp.csr_matrix((np.ones(len(g), np.float32), (np.arange(len(g)), col)),
                                      shape=(len(g), len(keys))))
        labels += [f"{name}: {k}" for k in keys]
    return sp.hstack(mats_tr, format="csr"), sp.hstack(mats_va, format="csr"), labels


class Joint:
    """e' = e + c + X g (offsets, optional), u' = u exp(l + X b) (scales); weak ridge on g and b (the
    one-hot columns of a descriptor sum to 1, collinear with c and l)."""

    def __init__(self, X: sp.csr_matrix, e: np.ndarray, t: np.ndarray, u0: float, offsets: bool,
                 lam: float = 1e-6):
        self.X, self.e, self.t, self.u0, self.offsets, self.lam = X, e, t, u0, offsets, lam
        self.p = X.shape[1]

    def unpack(self, x: np.ndarray):
        c, l = 10.0 * x[0], x[1]
        b = x[2:2 + self.p]
        g = 10.0 * x[2 + self.p:] if self.offsets else np.zeros(self.p)
        return c, l, b, g

    def size(self) -> int:
        return 2 + self.p * (2 if self.offsets else 1)

    def predict(self, x: np.ndarray, X: sp.csr_matrix, e: np.ndarray) -> np.ndarray:
        c, l, b, g = self.unpack(x)
        ee = e + c + X @ g
        uu = self.u0 * np.exp(l + X @ b)
        return expit(uu * ee)

    def fun(self, x: np.ndarray):
        c, l, b, g = self.unpack(x)
        ee = self.e + c + self.X @ g
        uu = self.u0 * np.exp(l + self.X @ b)
        s = expit(uu * ee)
        r = self.t - s
        n = len(r)
        gz = -2.0 * r * s * (1 - s) / n
        ge = gz * uu                      # dL / de'
        gl = gz * uu * ee                 # dL / d log u'
        grad = np.zeros_like(x)
        grad[0] = 10.0 * ge.sum()
        grad[1] = gl.sum()
        grad[2:2 + self.p] = self.X.T @ gl + 2 * self.lam * b
        f = float(np.mean(r * r)) + self.lam * float(b @ b)
        if self.offsets:
            gg = g / 10.0
            grad[2 + self.p:] = 10.0 * (self.X.T @ ge) + 2 * self.lam * gg
            f += self.lam * float(gg @ gg)
        return f * 1e3, grad * 1e3

    def fit(self, x0: np.ndarray | None = None) -> np.ndarray:
        x0 = np.zeros(self.size()) if x0 is None else x0
        res = minimize(self.fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
        return res.x


def joint_models(args, groups: dict, e_tr, t_tr, e_va, t_va, c: float, u: float, base: float) -> None:
    names = [n.strip() for n in args.joint.split(",") if n.strip()]
    for n in names:
        if n not in groups:
            raise SystemExit(f"unknown descriptor {n}; known: {', '.join(groups)}")
    e_tr, e_va = e_tr + c, e_va + c         # the global calibration as the start

    def run(sel: list[str], offsets: bool):
        X_tr, X_va, labels = one_hot(sel, groups)
        m = Joint(X_tr, e_tr, t_tr, u, offsets)
        x = m.fit()
        lv = float(np.mean((t_va - m.predict(x, X_va, e_va)) ** 2))
        return 100 * (base - lv) / base, m, x, labels

    print(f"\nJoint models of {', '.join(names)} (validation gain, % of the loss):")
    for offsets in (False, True):
        gain, m, x, labels = run(names, offsets)
        print(f"  all, {'scale and offset' if offsets else 'scale only'}: {gain:+.3f}%", flush=True)
    # the coefficients of the full model: k factor and offset per group
    print("\nFull model (scale and offset): k = k0 * factor, offset in cp, per group (training share):")
    X_tr, _, _ = one_hot(names, groups)
    share = np.asarray(X_tr.sum(axis=0)).ravel() / X_tr.shape[0] * 100
    c0, l, b, g = m.unpack(x)
    for j, lab in enumerate(labels):
        print(f"  {lab:40s} {share[j]:5.1f}%  k x {np.exp(-(l + b[j])):5.2f}  offset {c0 + g[j]:+6.1f}")
    # greedy forward selection (scale and offset)
    print("\nForward selection (scale and offset): validation gain after adding each descriptor")
    chosen: list[str] = []
    rest = list(names)
    while rest:
        best = max(((run(chosen + [n], True)[0], n) for n in rest))
        chosen.append(best[1])
        rest.remove(best[1])
        print(f"  + {best[1]:16s} {best[0]:+.3f}%", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="export directory (fit of the group offsets)")
    ap.add_argument("--val", required=True, help="export directory (measurement)")
    ap.add_argument("--max-score", type=float, default=2000)
    ap.add_argument("--min-share", type=float, default=0.3, help="groups below this share (%%) are pooled")
    ap.add_argument("--top", type=int, default=12, help="groups shown per descriptor (largest effect first)")
    ap.add_argument("--max-train", type=int, default=0, help="use only so many training rows (0 = all)")
    ap.add_argument("--joint", default="",
                    help="comma separated descriptors: joint model (scale and offset per group of each), the "
                         "coefficients and a forward selection, instead of the tables")
    args = ap.parse_args()

    tr, va = read(args.train, args.max_score), read(args.val, args.max_score)
    if 0 < args.max_train < len(tr):
        tr = tr.iloc[:args.max_train].reset_index(drop=True)
    print(f"train {len(tr):,} rows, validation {len(va):,} rows", flush=True)
    t_tr, t_va = tr["result"].to_numpy() / 2.0, va["result"].to_numpy() / 2.0
    e_tr, e_va = tr["eval"].to_numpy(np.float64), va["eval"].to_numpy(np.float64)
    c, u = fit_group(e_tr, t_tr, 0.0, 1 / 160.0)
    p_va = expit(u * (e_va + c))
    base = float(np.mean((t_va - p_va) ** 2))
    print(f"global calibration: c {c:.1f} cp, k {1 / u:.1f}; validation loss {base:.6f}", flush=True)

    desc_tr = pd.DataFrame([describe(f) for f in tr["fen"]])
    desc_va = pd.DataFrame([describe(f) for f in va["fen"]])
    for df_, src in ((desc_tr, tr), (desc_va, va)):
        df_["phase"] = (src["phase"].to_numpy() // 25 * 25).astype(str)
        df_["m50"] = np.minimum(src["m50"].to_numpy() // 10 * 10, 60).astype(str)
        ev = src["eval"].to_numpy()
        df_["eval level"] = pd.cut(ev, [-1e9, -400, -200, -100, -50, -20, 20, 50, 100, 200, 400, 1e9]).astype(str)
    print("descriptors computed", flush=True)

    groups = {name: pool(desc_tr[name].to_numpy(), desc_va[name].to_numpy(), args.min_share)
              for name in desc_tr.columns}
    if args.joint:
        joint_models(args, groups, e_tr, t_tr, e_va, t_va, c, u, base)
        return 0

    summary = []
    for name in desc_tr.columns:
        g_tr, g_va = groups[name]
        p_new = p_va.copy()
        rows = []
        for key in np.unique(g_tr):
            mt, mv = g_tr == key, g_va == key
            cg, ug = fit_group(e_tr[mt], t_tr[mt], c, u)
            if mv.any():
                p_new[mv] = expit(ug * (e_va[mv] + cg))
                lv_old = float(np.sum((t_va[mv] - p_va[mv]) ** 2))
                lv_new = float(np.sum((t_va[mv] - p_new[mv]) ** 2))
                resid = float(np.mean(t_va[mv] - p_va[mv]))
            else:
                lv_old = lv_new = resid = 0.0
            rows.append((key, mt.mean() * 100, mv.sum(), resid * 100, cg - c, 1 / ug, (lv_old - lv_new) / len(t_va)))
        gain = 100 * (base - float(np.mean((t_va - p_new) ** 2))) / base
        summary.append((gain, name, len(rows)))
        print(f"\n== {name}: {len(rows)} groups, validation gain {gain:+.3f}% of the loss")
        print(f"   {'group':28s} {'share':>6s} {'val n':>8s} {'resid':>7s} {'offset':>7s} {'k':>6s} {'gain':>8s}")
        for key, sh, nv, rs, oc, kg, gl in sorted(rows, key=lambda r: -abs(r[6]))[:args.top]:
            print(f"   {key:28s} {sh:5.1f}% {nv:8d} {rs:+6.2f} {oc:+7.1f} {kg:6.0f} {100 * gl / base:+7.3f}%")
    print("\nSummary (validation gain of a per group offset and scale, % of the loss):")
    for gain, name, n in sorted(summary, reverse=True):
        print(f"  {name:16s} {gain:+.3f}%  ({n} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
