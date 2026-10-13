"""Yardstick: loss of candidate evals on held-out on-policy positions with real game results.

A yardstick directory (made with hv_prep_sp.py and TuneSGD -X) contains exp-val (the export with the
compiled weights) and one cval-<name> per candidate (the same positions exported with -c <name>.cfg, so
the eval column is exactly the engine's eval of that candidate, HCE changes and PST included):

    TuneSGD-hvar.exe -X -i <yard>\\data-val -o <yard>\\exp-val
    TuneSGD-hvar.exe -X -c cand\\<name>.cfg -i <yard>\\data-val -o <yard>\\cval-<name>

For every candidate the offset c and the sigmoid scale k are fitted on va-01, the loss (q = 0, result
only) is measured on va-02 and reported relative to the compiled weights. The 95% interval comes from a
bootstrap over blocks of --block consecutive rows (about one game: the rows of a game are contiguous and
share the result, so positions are not independent).

Example:
    python hv_cand_eval.py --yard C:\\Learn\\Tune\\hvar\\yard-cd --cands sp-pst,sp-tuned,sp-both
"""

from __future__ import annotations

import argparse
import os

import env_dlls  # noqa: F401  (before numpy)
import numpy as np
from scipy.special import expit

import hv_data as hd
import hv_fit as hf


def row_losses(directory: str) -> tuple[np.ndarray, float]:
    """Per row loss on va-02 with c, k fitted on va-01; also the fitted k."""
    files = hd.export_files(directory)
    if len(files) != 2:
        raise ValueError(f"{directory}: expected 2 export files (va-01, va-02), found {len(files)}")
    sel, test = hd.load(files[:1]), hd.load(files[1:])
    prob = hf.Problem(hf.SPECS["hce"], sel, 0.0, 200.0)
    x, _ = hf.fit(prob, np.zeros(2), 200)
    c, u, _, _ = prob.split(x)
    r = test.result / 2.0 - expit(u * (test.eval + c))
    return r * r, 1 / u


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yard", required=True, help="yardstick directory with exp-val and cval-<name>")
    ap.add_argument("--cands", required=True, help="comma separated candidate names")
    ap.add_argument("--block", type=int, default=90, help="rows per bootstrap block (about one game)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap samples")
    args = ap.parse_args()

    lb, kb = row_losses(os.path.join(args.yard, "exp-val"))
    nb = len(lb) // args.block
    rng = np.random.default_rng(1)
    idx = rng.integers(0, nb, size=(args.boot, nb))
    print(f"test rows {len(lb):,} ({nb} blocks of {args.block}), compiled loss {lb.mean():.6f} (k {kb:.0f})")
    for name in [c for c in args.cands.split(",") if c]:
        lc, k = row_losses(os.path.join(args.yard, f"cval-{name}"))
        d = (lc - lb)[:nb * args.block].reshape(nb, args.block).mean(axis=1)
        boot = d[idx].mean(axis=1)
        print(f"  {name:16s} {100 * d.mean() / lb.mean():+7.3f}% +- {196 * boot.std() / lb.mean():.3f}% (95%)"
              f"   k {k:5.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
