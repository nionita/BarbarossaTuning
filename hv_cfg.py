"""Make a candidate config for SelfPlay / TuneSGD from the output of hv_fit.py.

The config has optional HCE weights (the lines of a refitted <model>.cfg, scaled towards the compiled
values with --hce-scale) and the PST of a pst-<model>.csv as lines pst.mid.<piece><square> = value
(1/8 cp, rounded, multiplied by --pst-scale; zero and unused entries are left out). The engine adds
the PST to the eval only when such lines are present (branch hvar).

Examples:
    python hv_cfg.py --pst fit1\\pst-hce+pst.csv --out cand\\pst40.cfg
    python hv_cfg.py --pst fit1\\pst-hce+pst.csv --pst-scale 0.5 --out cand\\pst40-h.cfg
    python hv_cfg.py --hce fit1\\hce-tuned.cfg --out cand\\hce-tuned.cfg
    python hv_cfg.py --hce fit1\\hce-tuned+pst.cfg --pst fit1\\pst-hce-tuned+pst.csv --out cand\\both.cfg
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

WEIGHTS_HS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Barbarossa", "Struct", "Weights.hs")


def compiled_weights(path: str) -> dict[str, float]:
    text = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'\("(\w+)",\s*\(\s*(-?\d+),\s*(-?\d+)\)\)', text):
        out["mid." + m.group(1)] = float(m.group(2))
        out["end." + m.group(1)] = float(m.group(3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pst", help="pst-<model>.csv of hv_fit.py")
    ap.add_argument("--pst-scale", type=float, default=1.0, help="factor for the PST values")
    ap.add_argument("--hce", help="<model>.cfg of hv_fit.py with refitted HCE weights")
    ap.add_argument("--hce-scale", type=float, default=1.0,
                    help="value = compiled + scale * (refitted - compiled)")
    ap.add_argument("--weights", default=WEIGHTS_HS, help="Weights.hs with the compiled values (for --hce-scale)")
    ap.add_argument("--out", required=True, help="candidate config to write")
    args = ap.parse_args()
    if not args.pst and not args.hce:
        print("Give --pst and / or --hce", file=sys.stderr)
        return 2

    lines = [f"-- candidate from hv_cfg.py: hce {args.hce} (scale {args.hce_scale}), "
             f"pst {args.pst} (scale {args.pst_scale})"]
    if args.hce:
        comp = compiled_weights(args.weights) if args.hce_scale != 1 else {}
        for line in open(args.hce, encoding="utf-8"):
            m = re.match(r"^\s*((?:mid|end)\.\w+)\s*=\s*(-?[0-9.eE+-]+)\s*$", line)
            if not m:
                continue
            v = float(m.group(2))
            if args.hce_scale != 1:
                c = comp[m.group(1)]
                v = c + args.hce_scale * (v - c)
            lines.append(f"{m.group(1)} = {round(v)}")
    npst = 0
    if args.pst:
        with open(args.pst, encoding="utf-8") as fi:
            for r in csv.DictReader(fi):
                if int(r["count"]) == 0:
                    continue
                for ph in ("mid", "end"):
                    v = round(args.pst_scale * float(r[ph]))
                    if v != 0:
                        lines.append(f"pst.{ph}.{r['piece']}{r['name']} = {v}")
                        npst += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("\n".join(lines) + "\n")
    print(f"{args.out}: {len(lines) - 1 - npst} HCE values, {npst} PST values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
