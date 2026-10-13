"""Split a fitted PST (pst-<model>.csv of hv_fit.py) into its material and placement parts.

A constant over all squares of a piece type is a change of that piece's value (material part: the
occurrence weighted mean per piece and phase); the rest is placement. On the validation export (va-01
for c and k, va-02 for the loss) the HCE eval plus each part is compared with the HCE eval alone:
full PST, material part, placement part, and per piece: its placement alone, its material alone and
all placement except it.

Example:
    python hv_decomp.py --pst C:\\Learn\\Tune\\hvar\\fit1\\pst-hce+pst.csv --val C:\\Learn\\Tune\\hvar\\exp-val
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace

import env_dlls  # noqa: F401  (before numpy)
import numpy as np

import hv_data as hd
import hv_fit as hf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pst", required=True, help="pst-<model>.csv of hv_fit.py")
    ap.add_argument("--val", required=True, help="validation export directory (va-01, va-02)")
    args = ap.parse_args()

    files = hd.export_files(args.val)
    sel, test = hd.load(files[:1]), hd.load(files[1:])
    w = np.zeros((hd.N_PST, 2))
    cnt = np.zeros(hd.N_PST)
    with open(args.pst, encoding="utf-8") as fi:
        for j, r in enumerate(csv.DictReader(fi)):
            w[j] = float(r["mid"]), float(r["end"])
            cnt[j] = int(r["count"])
    mat = np.zeros_like(w)
    for t in range(6):
        sl = slice(t * 64, t * 64 + 64)
        if cnt[sl].sum() > 0:
            mat[sl] = (cnt[sl] @ w[sl]) / cnt[sl].sum()
    mat[cnt == 0] = 0
    pos = np.where(cnt[:, None] > 0, w - mat, 0)

    def loss_with(table: np.ndarray) -> tuple[float, float]:
        out = []
        for d in (sel, test):
            add = d.pst @ table.astype(np.float32)
            out.append(replace(d, eval=d.eval + d.dmid * add[:, 0] + d.dend * add[:, 1]))
        prob = hf.Problem(hf.SPECS["hce"], out[0], 0.0, 200.0)
        x, _ = hf.fit(prob, np.zeros(2), 200)
        return prob.on(out[1]).loss(x), 1 / prob.split(x)[1]

    base, kb = loss_with(np.zeros_like(w))
    print(f"test loss HCE {base:.6f} (k {kb:.0f}); change in % (c, k fitted on va-01)")

    def show(name: str, table: np.ndarray) -> None:
        loss, k = loss_with(table)
        print(f"  {name:28s} {100 * (loss - base) / base:+7.3f}%   k {k:5.0f}")

    show("full PST", w)
    show("material part (piece means)", mat)
    show("placement part", pos)
    for t, pn in enumerate(hd.PIECE_NAMES):
        piece = np.zeros(hd.N_PST, bool)
        piece[t * 64:(t + 1) * 64] = True
        show(f"{pn} placement only", np.where(piece[:, None], pos, 0))
        show(f"{pn} material only", np.where(piece[:, None], mat, 0))
        show(f"all placement except {pn}", np.where(piece[:, None], 0, pos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
