"""Fit a drawishness scale of the eval (replacement of the heuristic part of Eval.Core.scaleFactor and of
the linear 50 moves scale) on exports of TuneSGD -X with the columns eflat, eend, sknow, special.

Models (each with a fitted offset c and sigmoid scale k):
    old    the engine's eval: (eflat + eend * scale / 64) * (128 - m50) / 128
    end    (eflat + eend * s) * g(m50): s scales the end game part only (like scaleFactor)
    all    (eflat + eend * s0) * s * g(m50): s scales the whole eval
where s = sknow / 64 where the knowledge rule applies (fixed), else the product of table factors over
descriptors of the position (stronger side = sign of the end game part for "end", of the eval for
"all"): pawns of the stronger side (0 .. 7, 8 or more = 1), opposite coloured bishops (only bishops /
with other pieces), the stronger side's pawns on one wing only, rook ending (R v R), queen ending
(Q v Q), each a log factor fitted from 0; g(m50) is a table over buckets of 10 half moves (bucket 0 = 1),
replacing (128 - m50) / 128. s0 = sknow / 64 or 1. Rows with a special eval (exact score in play) are
left out. Fitted on the training data, the loss reported on the validation data (first file
selection, second test, as hv_fit.py). The tables are printed as factors and in units of 1/64.

Example:
    python hv_scale.py --train C:\\Learn\\Tune\\hvar\\sp3\\exp-train-scale --val C:\\Learn\\Tune\\hvar\\sp3\\exp-val-scale
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import env_dlls  # noqa: F401  (before numpy)
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

COLS = ["fen", "score", "result", "eval", "scale", "m50", "eflat", "eend", "special", "sknow"]
M50_BUCKETS = 10


def read(files: list[str], max_score: float) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, engine="pyarrow", usecols=COLS) for f in files], ignore_index=True)
    keep = (np.abs(df["score"]) <= max_score) & (df["special"] == 0)
    return df[keep].reset_index(drop=True)


def board_info(fen: str) -> tuple:
    """Per side (white, black): pawns, pawn files mask, bishops on light / dark squares, knights, rooks,
    queens; and the side to move."""
    board, stm = fen.split(" ", 2)[:2]
    pw = pb = fw = fb = 0
    bl = [0, 0]
    bd = [0, 0]
    n = [0, 0]
    r = [0, 0]
    q = [0, 0]
    rank, file = 7, 0
    for ch in board:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch <= "9":
            file += ord(ch) - 48
        else:
            s = 0 if ch < "a" else 1
            c = ch.lower()
            if c == "p":
                if s == 0:
                    pw += 1
                    fw |= 1 << file
                else:
                    pb += 1
                    fb |= 1 << file
            elif c == "b":
                if (rank + file) % 2:
                    bl[s] += 1
                else:
                    bd[s] += 1
            elif c == "n":
                n[s] += 1
            elif c == "r":
                r[s] += 1
            elif c == "q":
                q[s] += 1
            file += 1
    return pw, pb, fw, fb, bl[0], bl[1], bd[0], bd[1], n[0], n[1], r[0], r[1], q[0], q[1], stm == "w"


class Design:
    """Descriptor columns (one-hot, log factors) of the scale s, given the stronger side of every row."""

    def __init__(self, info: np.ndarray, strong_is_mover: np.ndarray):
        (pw, pb, fw, fb, blw, blb, bdw, bdb, nw, nb, rw, rb, qw, qb, wmove) = info.T
        strong_white = np.where(strong_is_mover, wmove, 1 - wmove).astype(bool)
        sp_ = np.where(strong_white, pw, pb)
        sf_ = np.where(strong_white, fw, fb)
        bw, bb = blw + bdw, blb + bdb
        minors_rooks_queens = nw + nb + rw + rb + qw + qb
        ocb = (bw == 1) & (bb == 1) & (((blw == 1) & (bdb == 1)) | ((bdw == 1) & (blb == 1)))
        cols, self.labels = [], []
        for k in range(8):
            cols.append(sp_ == k)
            self.labels.append(f"stronger side pawns {k}")
        cols.append(ocb & (minors_rooks_queens == 0))
        self.labels.append("opposite bishops only")
        cols.append(ocb & (minors_rooks_queens > 0))
        self.labels.append("opposite bishops + pieces")
        qside, kside = 0x0F, 0xF0
        cols.append((sf_ != 0) & (((sf_ & qside) == 0) | ((sf_ & kside) == 0)))
        self.labels.append("stronger pawns on one wing")
        cols.append((rw == 1) & (rb == 1) & (bw + bb + nw + nb + qw + qb == 0))
        self.labels.append("rook ending (R v R)")
        cols.append((qw == 1) & (qb == 1) & (bw + bb + nw + nb + rw + rb == 0))
        self.labels.append("queen ending (Q v Q)")
        self.X = sp.csr_matrix(np.column_stack(cols).astype(np.float32))


def m50_design(m50: np.ndarray) -> sp.csr_matrix:
    b = np.minimum(m50 // 10, M50_BUCKETS - 1)
    rows = np.nonzero(b > 0)[0]
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, b[rows] - 1)), shape=(len(m50), M50_BUCKETS - 1))


class Model:
    """Loss and gradient of the "end" or "all" model on one data set."""

    def __init__(self, kind: str, df: pd.DataFrame, info: np.ndarray):
        self.kind = kind
        self.t = df["result"].to_numpy() / 2.0
        self.flat = df["eflat"].to_numpy(np.float64)
        self.end = df["eend"].to_numpy(np.float64)
        sk = df["sknow"].to_numpy()
        self.know = sk >= 0
        self.s0 = np.where(self.know, sk / 64.0, 1.0)
        self.m50 = df["m50"].to_numpy()
        if kind == "end":
            strong = self.end >= 0
        else:
            strong = (self.flat + self.end * self.s0) >= 0
        d = Design(info, strong)
        self.labels = d.labels
        # the knowledge rows have a fixed s (no table factors) in the "end" model
        self.Xs = d.X if kind == "all" else sp.diags((~self.know).astype(np.float32)) @ d.X
        self.Xg = m50_design(self.m50)
        self.ps, self.pg = self.Xs.shape[1], self.Xg.shape[1]

    def size(self) -> int:
        return 2 + self.ps + self.pg

    def parts(self, x: np.ndarray):
        c, lu = 10.0 * x[0], x[1]
        a, b = x[2:2 + self.ps], x[2 + self.ps:]
        return c, lu, a, b

    def evals(self, x: np.ndarray):
        _, _, a, b = self.parts(x)
        s = np.exp(self.Xs @ a)
        g = np.exp(self.Xg @ b)
        if self.kind == "end":
            s = np.where(self.know, self.s0, s)
            inner = self.flat + self.end * s
            return inner * g, s, g, inner
        inner = self.flat + self.end * self.s0
        return inner * s * g, s, g, inner

    def fun(self, x: np.ndarray):
        c, lu, a, b = self.parts(x)
        u = np.exp(lu) / 200.0
        e, s, g, inner = self.evals(x)
        z = u * (e + c)
        p = expit(z)
        r = self.t - p
        n = len(r)
        gz = -2.0 * r * p * (1 - p) / n
        ge = gz * u
        grad = np.zeros_like(x)
        grad[0] = 10.0 * ge.sum()
        grad[1] = float(np.dot(gz, z))
        if self.kind == "end":
            de_da = ge * self.end * s * g          # d e / d a_j = end * s * g * X_j (s fixed on knowledge rows)
        else:
            de_da = ge * e
        grad[2:2 + self.ps] = self.Xs.T @ de_da
        grad[2 + self.ps:] = self.Xg.T @ (ge * e)
        return float(np.mean(r * r)) * 1e3, grad * 1e3

    def loss(self, x: np.ndarray) -> float:
        c, lu, _, _ = self.parts(x)
        e = self.evals(x)[0]
        r = self.t - expit(np.exp(lu) / 200.0 * (e + c))
        return float(np.mean(r * r))


def calib_loss(e_tr, t_tr, e_va, t_va) -> tuple[float, float, float]:
    """Fit c, k of a fixed eval on the training data, loss on the validation data."""
    def f(x):
        u = np.exp(x[1]) / 200.0
        z = u * (e_tr + 10 * x[0])
        p = expit(z)
        r = t_tr - p
        gz = -2 * r * p * (1 - p) / len(r)
        return float(np.mean(r * r)) * 1e3, np.array([10 * u * gz.sum(), float(np.dot(gz, z))]) * 1e3
    x = minimize(f, np.zeros(2), jac=True, method="L-BFGS-B").x
    u = np.exp(x[1]) / 200.0
    return float(np.mean((t_va - expit(u * (e_va + 10 * x[0]))) ** 2)), 10 * x[0], 1 / u


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True, help="directory with two exports: selection and test")
    ap.add_argument("--max-score", type=float, default=2000)
    ap.add_argument("--models", default="end,all")
    args = ap.parse_args()

    trf = sorted(glob.glob(os.path.join(args.train, "*-exp.csv")))
    vaf = sorted(glob.glob(os.path.join(args.val, "*-exp.csv")))
    tr, va = read(trf, args.max_score), read(vaf, args.max_score)
    info_tr = np.array([board_info(f) for f in tr["fen"]], dtype=np.int64)
    info_va = np.array([board_info(f) for f in va["fen"]], dtype=np.int64)
    print(f"train {len(tr):,} rows ({len(trf)} files), validation {len(va):,} (special evals left out)", flush=True)
    t_tr, t_va = tr["result"].to_numpy() / 2.0, va["result"].to_numpy() / 2.0
    base, c0, k0 = calib_loss(tr["eval"].to_numpy(float), t_tr, va["eval"].to_numpy(float), t_va)
    print(f"old    validation loss {base:.6f}  (c {c0:.1f}, k {k0:.1f})", flush=True)
    for kind in [m for m in args.models.split(",") if m]:
        mt, mv = Model(kind, tr, info_tr), Model(kind, va, info_va)
        x0 = np.zeros(mt.size())
        x0[0], x0[1] = c0 / 10.0, np.log(200.0 / k0)
        # start g at the old linear 50 moves scale
        for j in range(mt.pg):
            x0[2 + mt.ps + j] = np.log((128 - 10 * (j + 1) - 5) / 128)
        res = minimize(mt.fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-13})
        x = res.x
        lv = mv.loss(x)
        c, lu, a, b = mt.parts(x)
        print(f"\n{kind:6s} validation loss {lv:.6f}  ({100 * (lv - base) / base:+.3f}% against old; c {c:.1f}, "
              f"k {200 / np.exp(lu):.1f}, {res.nit} it)")
        share = np.asarray(mt.Xs.sum(axis=0)).ravel() / len(tr) * 100
        for lab, aj, sh in zip(mt.labels, a, share):
            print(f"   {lab:30s} {sh:5.1f}%  factor {np.exp(aj):5.2f}  ({64 * np.exp(aj):4.0f} / 64)")
        print("   50 moves counter (factor, old (128 - m50) / 128 at the bucket middle):")
        print("     0-9: 1.00 (0.96)")
        for j, bj in enumerate(b):
            lo = 10 * (j + 1)
            hi = "+" if j == mt.pg - 1 else f"-{lo + 9}"
            print(f"     {lo}{hi}: {np.exp(bj):4.2f} ({(128 - lo - 5) / 128:4.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
