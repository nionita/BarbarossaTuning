# BarbarossaTuning

Scripts for tuning and measuring the Barbarossa HCE evaluation (TuneSGD + SelfPlay), written during the
October 2026 tuning study. See `TUNING.md` in the Barbarossa repository for findings and the reasoning.

Working folder of the study: `C:\Learn\Tune\cxig` (the default paths in the scripts point there).
Binaries: `C:\astra\TuneSGD\TuneSGD-<branch>.exe`, `C:\astra\SelfPlay\SelfPlay-<branch>.exe`.

| Script | Purpose |
|---|---|
| `prep_data.py` | chilo CSV files -> TuneSGD training files: skip headers, hash sample, phase >= 120, dedup, result + 1, shuffle, split into files named `*-filt.csv` (run with the miniconda python) |
| `run_tune.ps1` | one TuneSGD run (`-p` subset, `-c` start config, `-K`, `-Q`, `-Fix`); writes `tuned.cfg`, `tune.log`, `run.txt` |
| `run_group.ps1` | one candidate (a group of weights from a config, optionally `-Scale`d) against the compiled weights, fixed number of pairs, Elo +- CI |
| `run_sharded.ps1` | the same as `run_group.ps1` in 3 parallel shards over disjoint openings, merged |
| `merge_groups.ps1` | adds up the WDL of several group runs |
| `run_sweep.ps1` | sweep of weights over factors of the compiled value: Elo and loss (several loss variants) per point |

Typical use:

    .\run_tune.ps1 -RunDir C:\Learn\Tune\cxig\runN -K 220 -Q 0 -Fix -1
    .\run_sharded.ps1 -Name test -Source C:\Learn\Tune\cxig\runN\tuned.cfg -All -Scale 0.5
    .\run_sweep.ps1 -Name matdiff -Params ewMaterialDiff -Factors 0.5,0.75,1.25,1.5

Notes:
- SelfPlay prints the SPRT / match verdict only to stdout; the scripts keep it in `match.stdout.txt`.
- `-Scale` and `run_sweep.ps1` read the compiled values from `Struct\Weights.hs` of the Barbarossa checkout.
- Existing group / run directories are never overwritten without `-Force`.
