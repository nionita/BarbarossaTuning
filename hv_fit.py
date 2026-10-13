"""How much does the HCE miss? Fit additive models on top of the exported HCE eval (TuneSGD -X).

Models (each with a free constant offset c and a free sigmoid scale k, fitted on the training data;
with free HCE weights k stays at the baseline value, as eval scale and k are degenerate there):
    hce          the HCE eval as it is                                    (baseline)
    gen3         the score column of the data (gen3 static eval)          (ceiling)
    hce+pst      HCE + antisymmetric PST, 2 x 384 weights, ridge          (piece placement only)
    hce-tuned    HCE with its 92 weights refitted (exact: eval is linear in the weights)
    hce-tuned+pst    both together
Loss as TuneSGD -L 5: mean (t - sigmoid(e / k))^2 with t = (1 - q) * result / 2 + q * sigmoid(score / kq),
records with |score| > --max-score dropped. The ridge prior of the PST weights is given as a standard
deviation tau in cp, lambda = sigma^2 / (N * (8 * tau)^2) with sigma^2 = baseline loss; tau is chosen on
the selection part of the validation data, the test part is only used for the report.
Validation: the files of --val are split, first half selection, second half test (one file: by rows).

Output in --out: report.txt (also printed), results.json, pst-<model>.csv, hce-tuned.cfg (TuneSGD
config with the refitted HCE weights).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import env_dlls  # noqa: F401  (before numpy)
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import hv_data as hd

W_SCALE = 64.0      # optimizer units of the weights: 64 / 8 = 8 cp
C_SCALE = 10.0      # optimizer units of the offset: 10 cp
K_REF = 200.0       # u = exp(rho) / K_REF
F_SCALE = 1000.0    # objective scale for the optimizer
ALL_MODELS = ["hce", "gen3", "hce+pst", "hce-tuned", "hce-tuned+pst"]


@dataclass(frozen=True)
class Spec:
    name: str
    base: str           # "eval" or "score"
    hce: bool           # free HCE weight deltas (2 x 46)
    pst: bool           # PST weights (2 x 384)


SPECS = {
    "hce": Spec("hce", "eval", False, False),
    "gen3": Spec("gen3", "score", False, False),
    "hce+pst": Spec("hce+pst", "eval", False, True),
    "hce-tuned": Spec("hce-tuned", "eval", True, False),
    "hce-tuned+pst": Spec("hce-tuned+pst", "eval", True, True),
}


class Problem:
    """Loss and gradient of one model on one data set."""

    def __init__(self, spec: Spec, d: hd.Data, q: float, kq: float, hscale: np.ndarray | None = None):
        self.spec = spec
        self.d = d
        self.q = q
        self.kq = kq
        self.base = d.eval if spec.base == "eval" else d.score
        self.target = d.result / 2.0
        if q > 0:
            self.target = (1 - q) * self.target + q * expit(d.score / kq)
        self.nh = len(d.names) if spec.hce else 0
        self.np_ = hd.N_PST if spec.pst else 0
        self.size = 2 + 2 * self.nh + 2 * self.np_
        self.lam = 0.0
        # Optimizer units of the HCE weights: one unit changes the eval by 8 cp RMS (the features have
        # very different magnitudes, a common unit makes the problem badly conditioned)
        self.hscale = hscale
        if self.nh and hscale is None:
            f2 = d.feats * d.feats
            rms = np.sqrt(np.column_stack([f2.T @ (d.dmid * d.dmid), f2.T @ (d.dend * d.dend)]) / d.n)
            self.hscale = np.where(rms > 1e-9, 8.0 / np.maximum(rms, 1e-9), W_SCALE)

    def on(self, d: hd.Data) -> "Problem":
        """The same model (and parameter units) on other data."""
        return Problem(self.spec, d, self.q, self.kq, self.hscale)

    def split(self, x: np.ndarray):
        c = C_SCALE * x[0]
        u = math.exp(x[1]) / K_REF
        i = 2
        wh = x[i:i + 2 * self.nh].reshape(2, self.nh).T * self.hscale if self.nh else None
        i += 2 * self.nh
        wp = x[i:i + 2 * self.np_].reshape(2, self.np_).T * W_SCALE if self.np_ else None
        return c, u, wh, wp

    def evals(self, x: np.ndarray) -> np.ndarray:
        d = self.d
        c, u, wh, wp = self.split(x)
        add = np.zeros((d.n, 2), dtype=np.float32)
        if wh is not None:
            add += d.feats @ wh.astype(np.float32)
        if wp is not None:
            add += d.pst @ wp.astype(np.float32)
        return self.base + c + d.dmid * add[:, 0] + d.dend * add[:, 1]

    def loss(self, x: np.ndarray) -> float:
        _, u, _, _ = self.split(x)
        r = self.target - expit(u * self.evals(x))
        return float(np.mean(r * r))

    def fun(self, x: np.ndarray):
        d = self.d
        c, u, wh, wp = self.split(x)
        e = self.evals(x)
        s = expit(u * e)
        r = self.target - s
        n = d.n
        f = float(np.mean(r * r))
        gz = -2.0 * r * s * (1 - s) / n          # dL / d(u e)
        ge = gz * u                              # dL / de
        grad = np.zeros_like(x)
        grad[0] = C_SCALE * ge.sum()
        grad[1] = u * float(np.dot(gz, e))
        G = np.column_stack([ge * d.dmid, ge * d.dend]).astype(np.float32)
        i = 2
        if wh is not None:
            gh = (d.feats.T @ G).astype(np.float64)        # nh x 2
            grad[i:i + 2 * self.nh] = (gh * self.hscale).T.ravel()
            i += 2 * self.nh
        if wp is not None:
            gp = np.asarray(d.pst.T @ G, dtype=np.float64)  # 384 x 2
            gp += 2 * self.lam * wp
            f += self.lam * float(np.sum(wp * wp))
            grad[i:i + 2 * self.np_] = W_SCALE * gp.T.ravel()
        return F_SCALE * f, F_SCALE * grad

    def x0(self) -> np.ndarray:
        x = np.zeros(self.size)
        return x


def fit(prob: Problem, x0: np.ndarray, maxiter: int) -> tuple[np.ndarray, int]:
    # With free HCE weights the scale of the eval and k are (almost) degenerate: k stays at x0
    bounds = None
    if prob.spec.hce:
        bounds = [(None, None)] * len(x0)
        bounds[1] = (x0[1], x0[1])
    res = minimize(prob.fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-11, "gtol": 1e-8})
    return res.x, res.nit


def take(d: hd.Data, idx: np.ndarray) -> hd.Data:
    return replace(d, score=d.score[idx], result=d.result[idx], eval=d.eval[idx], phase=d.phase[idx],
                   scale=d.scale[idx], m50=d.m50[idx], dmid=d.dmid[idx], dend=d.dend[idx],
                   const=d.const[idx], feats=d.feats[idx], pst=d.pst[idx])


def sq_name(s: int) -> str:
    return "abcdefgh"[s % 8] + str(s // 8 + 1)


def pst_boards(wp: np.ndarray, used: np.ndarray) -> list[str]:
    """PST as 8 x 8 boards in cp (owner's view, rank 8 on top), mid and end side by side."""
    out = []
    for t, pn in enumerate(hd.PIECE_NAMES):
        out.append(f"  {pn}  mid (cp){' ' * 34}end (cp)")
        for rank in range(7, -1, -1):
            cells = []
            for ph in (0, 1):
                row = []
                for f in range(8):
                    j = t * 64 + rank * 8 + f
                    row.append(f"{wp[j, ph] / 8:5.0f}" if used[j] else "    .")
                cells.append("".join(row))
            out.append(f"  {rank + 1} {cells[0]}    {cells[1]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="directory (or file) with the training export")
    ap.add_argument("--val", required=True, help="directory (or file) with the validation export")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--models", default=",".join(ALL_MODELS), help="comma separated, of: " + ",".join(ALL_MODELS))
    ap.add_argument("--tau", default="2.5,5,10,20,40",
                    help="PST prior standard deviations in cp to try (0 = no ridge)")
    ap.add_argument("-q", type=float, default=0.0, help="weight of the score in the target (default 0)")
    ap.add_argument("--kq", type=float, default=200.0, help="sigmoid scale of the score in the target (q > 0)")
    ap.add_argument("--max-score", type=float, default=2000, help="drop records with |score| above (0 = keep all)")
    ap.add_argument("--max-train", type=int, default=0, help="use only so many training records (0 = all)")
    ap.add_argument("--maxiter", type=int, default=3000, help="L-BFGS iterations per fit")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    for m in models:
        if m not in SPECS:
            print(f"Unknown model {m}", file=sys.stderr)
            return 2
    # the baseline comes first: it gives the start (and for free HCE weights the fixed) c and k
    models = ["hce"] + [m for m in models if m != "hce"]
    taus = [float(t) for t in args.tau.split(",")]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    train = hd.load(hd.export_files(args.train), args.max_score)
    if 0 < args.max_train < train.n:
        train = take(train, np.arange(args.max_train))
    vfiles = hd.export_files(args.val)
    if len(vfiles) > 1:
        h = len(vfiles) // 2
        sel, test = hd.load(vfiles[:h], args.max_score), hd.load(vfiles[h:], args.max_score)
    else:
        v = hd.load(vfiles, args.max_score)
        sel, test = take(v, np.arange(0, v.n, 2)), take(v, np.arange(1, v.n, 2))
    used = hd.pst_used(train)
    say(f"train {train.n:,} records ({len(train.files)} files), selection {sel.n:,}, test {test.n:,}; "
        f"q = {args.q}, max score {args.max_score}; PST columns used {int(used.sum())} / {hd.N_PST} "
        f"({time.time() - t0:.0f}s)")

    results: dict[str, dict] = {}
    xs: dict[str, np.ndarray] = {}

    def evaluate(name: str, prob: Problem, x: np.ndarray, extra: dict) -> None:
        _, u, _, _ = prob.split(x)
        r = {"train": prob.loss(x),
             "sel": prob.on(sel).loss(x),
             "test": prob.on(test).loss(x),
             "k": 1 / u, "c": C_SCALE * x[0]}
        r.update(extra)
        results[name] = r
        xs[name] = x
        say(f"{name:14s} train {r['train']:.6f}  sel {r['sel']:.6f}  test {r['test']:.6f}  "
            f"k {r['k']:6.1f}  c {r['c']:6.1f}  " + "  ".join(f"{k} {v}" for k, v in extra.items())
            + f"  ({time.time() - t0:.0f}s)")

    sigma2 = None
    for m in models:
        spec = SPECS[m]
        prob = Problem(spec, train, args.q, args.kq)
        x0 = prob.x0()
        # warm start from the fitted c, k of the baseline
        if "hce" in xs and spec.base == "eval":
            x0[:2] = xs["hce"][:2]
        if not spec.pst:
            x, nit = fit(prob, x0, args.maxiter)
            evaluate(m, prob, x, {"iter": nit})
            if m == "hce":
                sigma2 = results[m]["train"]
            continue
        if sigma2 is None:
            sigma2 = Problem(SPECS["hce"], train, args.q, args.kq).loss(x0)
        # ridge path from strong to weak prior, warm started, tau chosen on the selection part
        best = None
        x = x0
        for tau in sorted(taus, key=lambda t: t if t > 0 else math.inf):
            prob.lam = sigma2 / (train.n * (8 * tau) ** 2) if tau > 0 else 0.0
            x, nit = fit(prob, x, args.maxiter)
            lsel = prob.on(sel).loss(x)
            say(f"  {m} tau {tau:5.1f} cp: train {prob.loss(x):.6f}  sel {lsel:.6f}  ({nit} it, "
                f"{time.time() - t0:.0f}s)")
            if best is None or lsel < best[0]:
                best = (lsel, tau, x.copy(), nit)
        prob.lam = 0.0
        evaluate(m, prob, best[2], {"tau": best[1], "iter": best[3]})

    # summary relative to the baseline, gap to the ceiling
    if "hce" in results:
        say()
        say("Test loss relative to hce (negative = better); gap closed = part of the hce -> gen3 gap")
        base = results["hce"]["test"]
        gap = base - results["gen3"]["test"] if "gen3" in results else None
        for m, r in results.items():
            rel = 100 * (r["test"] - base) / base
            closed = f"  gap closed {100 * (base - r['test']) / gap:5.1f}%" if gap and m != "hce" else ""
            over = 100 * (r["test"] - r["train"]) / r["train"]
            say(f"  {m:14s} {rel:+7.3f}%{closed}   (test - train {over:+.3f}%)")

    # PST and refitted HCE weights
    counts = np.asarray(abs(train.pst).sum(axis=0)).ravel()
    for m in models:
        if m not in xs:
            continue
        prob = Problem(SPECS[m], train, args.q, args.kq)
        _, _, wh, wp = prob.split(xs[m])
        if wp is not None:
            say()
            say(f"PST of {m} (cp, owner's view), occurrence weighted mean per piece (material shift):")
            for t, pn in enumerate(hd.PIECE_NAMES):
                sl = slice(t * 64, t * 64 + 64)
                w = counts[sl]
                means = [float(np.dot(w, wp[sl, ph]) / max(w.sum(), 1) / 8) for ph in (0, 1)]
                say(f"  {pn}: mid {means[0]:7.1f}  end {means[1]:7.1f}")
            for s in pst_boards(wp, used):
                say(s)
            with open(os.path.join(args.out, f"pst-{m}.csv"), "w", newline="\n") as fo:
                fo.write("piece,square,name,count,mid,end\n")
                for j in range(hd.N_PST):
                    fo.write(f"{hd.PIECE_NAMES[j // 64]},{j % 64},{sq_name(j % 64)},{int(counts[j])},"
                             f"{wp[j, 0]:.2f},{wp[j, 1]:.2f}\n")
        if wh is not None:
            w0 = hd.read_weights(os.path.dirname(train.files[0]))
            say()
            say(f"HCE weights of {m} (compiled -> refitted):")
            with open(os.path.join(args.out, f"{m}.cfg"), "w", newline="\n") as fo:
                fo.write(f"-- HCE weights refitted by hv_fit.py ({m}) on {args.train}\n")
                for ph, pname in ((0, "mid"), (1, "end")):
                    for i, n in enumerate(train.names):
                        old = w0[n][ph]
                        new = old + wh[i, ph]
                        fo.write(f"{pname}.{n} = {round(new)}\n")
                for i, n in enumerate(train.names):
                    say(f"  {n:18s} mid {w0[n][0]:5d} -> {w0[n][0] + wh[i, 0]:7.1f}   "
                        f"end {w0[n][1]:5d} -> {w0[n][1] + wh[i, 1]:7.1f}")

    with open(os.path.join(args.out, "results.json"), "w", newline="\n") as fo:
        json.dump({"args": vars(args), "results": results}, fo, indent=2)
    with open(os.path.join(args.out, "report.txt"), "w", newline="\n") as fo:
        fo.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
