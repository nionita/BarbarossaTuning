# BarbarossaTuning

Tools for tuning and measuring the hand-crafted evaluation (HCE) of the chess engine
[Barbarossa](https://github.com/nionita/Barbarossa): data preparation, offline fits of the evaluation
weights (and of a piece square table) in Python, and node-limited self-play matches to measure the
strength of a candidate against the compiled weights.

The main result so far: fitted on positions from the engine's own games with the real game results
(on-policy data), the refitted weights together with a piece square table gained about +28 Elo in
node-limited self-play (see `results/hvar` and the procedure below). Findings and reasoning are in
`TUNING.md` (state, lessons, ideas) and `TUNING-HISTORY.md` (all experiments) of the Barbarossa repository.

Requirements:
- Barbarossa executables built from its `hvar` branch: `TuneSGD` (export, `-X`) and `SelfPlay`
  (matches, data collection with `--data`).
- Python 3 with numpy, scipy and pandas (pyarrow for fast CSV reading); `prep_data.py` and
  `hv_prep_sp.py` need only the standard library, `uci_data.py` needs python-chess.
- Windows PowerShell for the match scripts (`*.ps1`).

The default paths in the scripts are those of the author's machines (working folders under
`C:\Learn\Tune`, binaries under `C:\astra`); pass your own paths as arguments.

License: MIT, see `LICENSE`.

## Scripts

| Script | Purpose |
|---|---|
| `prep_data.py` | chilo CSV files -> TuneSGD training files: skip headers, hash sample, phase >= 120, dedup, result + 1, shuffle, split into files named `*-filt.csv` (run with the miniconda python) |
| `run_tune.ps1` | one TuneSGD run (`-p` subset, `-c` start config, `-K`, `-Q`, `-Fix`); writes `tuned.cfg`, `tune.log`, `run.txt` |
| `run_group.ps1` | one candidate (a group of weights from a config, optionally `-Scale`d) against the compiled weights, fixed number of pairs, Elo +- CI |
| `run_sharded.ps1` | the same as `run_group.ps1` in 3 parallel shards over disjoint openings, merged |
| `merge_groups.ps1` | adds up the WDL of several group runs |
| `run_sweep.ps1` | sweep of weights over factors of the compiled value: Elo and loss (several loss variants) per point |
| `hv_data.py` | loads the `TuneSGD -X` export (`*-exp.csv`, cached as `.npz`): HCE eval, features, pseudo derivatives, antisymmetric PST matrix |
| `hv_fit.py` | how much the HCE misses: baseline, gen3 ceiling, HCE + PST, refitted HCE (+ PST) on held-out data |
| `env_dlls.py` | lets the conda env's `python.exe` run without activation (DLL path for numpy) |
| `hv_prep_sp.py` | SelfPlay `--data` files -> game disjoint training / validation files (whole shards held out) |
| `hv_cfg.py` | candidate config from `hv_fit.py` output: refitted HCE weights and / or PST (`pst.*` lines), `--placement` |
| `hv_cand_eval.py` | yardstick: loss of candidate evals on held-out on-policy games with real results, game block bootstrap |
| `hv_decomp.py` | split a fitted PST into material part (per piece means) and placement, loss of each part |
| `run_uci_games.ps1` | fastchess gauntlet of Barbarossa against other engines (PGN + Barbarossa's working directory) |
| `uci_data.py` | Barbarossa's `-d` data files (or Info logs) + fastchess PGN -> files in the format of SelfPlay `--data` |

## On-policy tuning of the HCE (branch `hvar`, 2026-10-13)

Results and reasoning: `TUNING-HISTORY.md` of the Barbarossa repository (findings 24-36). Working folder
`C:\Learn\Tune\hvar`, `python` = `C:\Users\Nicu\miniconda3a\envs\jacob\python.exe`, binaries of branch `hvar`
(`SelfPlay-hvar.exe`, `TuneSGD-hvar.exe`). The steps that gave `sp-both` (+27.9 Elo):

1. Data: the compiled engine plays itself with data collection (`cand\compiled.cfg` = one weight at its
   compiled value), 10k nodes per move, two runs on disjoint openings (40k games, ~4M positions):

       .\run_sharded.ps1 -Name gen10k-a -Config C:\Learn\Tune\hvar\cand\compiled.cfg -Shards 4 -Pairs 2500 -Nodes 10000 -DataDir C:\Learn\Tune\hvar\sp-data -Root C:\Learn\Tune\hvar\groups -Exe C:\astra\SelfPlay\SelfPlay-hvar.exe
       .\run_sharded.ps1 -Name gen10k-b ... -Offset 10000 ...

2. Training / validation files (whole shards held out = game disjoint), export with the compiled weights:

       python hv_prep_sp.py --src C:\Learn\Tune\hvar\sp-data-10k --val gen10k-b-s4 --out C:\Learn\Tune\hvar\sp10k
       TuneSGD-hvar.exe -X -i C:\Learn\Tune\hvar\sp10k\data-train -o C:\Learn\Tune\hvar\sp10k\exp-train
       TuneSGD-hvar.exe -X -i C:\Learn\Tune\hvar\sp10k\data-val -o C:\Learn\Tune\hvar\sp10k\exp-val

3. Fit (q = 0: real results only; ridge prior of the PST chosen on va-01, report on va-02):

       python hv_fit.py --train C:\Learn\Tune\hvar\sp10k\exp-train --val C:\Learn\Tune\hvar\sp10k\exp-val --out C:\Learn\Tune\hvar\fit-sp10k --models hce,hce+pst,hce-tuned,hce-tuned+pst --tau 5,10,20,40,80

4. Candidates:

       python hv_cfg.py --hce fit-sp10k\hce-tuned+pst.cfg --pst fit-sp10k\pst-hce-tuned+pst.csv --out cand\sp-both.cfg
       python hv_cfg.py --hce fit-sp10k\hce-tuned.cfg --out cand\sp-tuned.cfg

5. Yardstick before playing: on-policy games never used in a fit (here gen10k-c/d, other opening file),
   exported once with the compiled weights and once per candidate:

       python hv_prep_sp.py --src C:\Learn\Tune\hvar\sp-data2-10k --val gen10k-c-s1,...,gen10k-d-s4 --out C:\Learn\Tune\hvar\yard-cd --files 1
       TuneSGD-hvar.exe -X -i yard-cd\data-val -o yard-cd\exp-val
       TuneSGD-hvar.exe -X -c cand\sp-both.cfg -i yard-cd\data-val -o yard-cd\cval-sp-both
       python hv_cand_eval.py --yard C:\Learn\Tune\hvar\yard-cd --cands sp-both,sp-tuned

6. Match against the compiled weights (1000 nodes, 20k games):

       .\run_sharded.ps1 -Name sp-both -Config C:\Learn\Tune\hvar\cand\sp-both.cfg -Shards 4 -Pairs 2500 -Root C:\Learn\Tune\hvar\groups -Exe C:\astra\SelfPlay\SelfPlay-hvar.exe

Earlier steps of the experiment (chilo data, gen3 scores as ceiling; not used for the final candidates):

    python prep_data.py --out C:\Learn\Tune\hvar\data-train --records 4000000 --files 8 --rate 0.06 --phase-min 1 --prefix tr --holdout 8 --part train --seed 1
    python prep_data.py --out C:\Learn\Tune\hvar\data-val --records 1000000 --files 2 --rate 0.1 --phase-min 1 --prefix va --holdout 8 --part val --seed 2
    python hv_fit.py --train C:\Learn\Tune\hvar\exp-train --val C:\Learn\Tune\hvar\exp-val --out C:\Learn\Tune\hvar\fit1

Typical use:

    .\run_tune.ps1 -RunDir C:\Learn\Tune\cxig\runN -K 220 -Q 0 -Fix -1
    .\run_sharded.ps1 -Name test -Source C:\Learn\Tune\cxig\runN\tuned.cfg -All -Scale 0.5
    .\run_sweep.ps1 -Name matdiff -Params ewMaterialDiff -Factors 0.5,0.75,1.25,1.5

Notes:
- SelfPlay prints the SPRT / match verdict only to stdout; the scripts keep it in `match.stdout.txt`.
- `-Scale` and `run_sweep.ps1` read the compiled values from `Struct\Weights.hs` of the Barbarossa checkout.
- Existing group / run directories are never overwritten without `-Force`.

## Data from games against other engines (branch `ucidat`, 2026-10-15)

Barbarossa built from branch `ucidat` or later writes with `-d` per UCI search the leaf position that SelfPlay
`--data` writes (the evaluated position of the last root iteration, its static score, the root score, the
start position of the GUI's position command and the ply), into
`barbarossa-<version>-<branch>-<start second>-<pid>.data` in its working directory: one file per engine
process, about 16 KB per game. The game results come from the fastchess PGN; `uci_data.py` joins both.

1. Games: Barbarossa with `"args=-l 5 -d"` (`-l 5`: no log, `-d`: data file), its own working directory per run
   (all Barbarossa processes of a run may share it), a fresh PGN file per run (fastchess appends), no
   adjudication (real results), openings with full move number 1. Template (MEMO, concurrency 8):

       fastchess.exe -engine cmd=C:\Engines\Barbarossa\Barbarossa-0.8.0-ucidat.exe name=Barbarossa "args=-l 5 -d" dir=C:\Learn\Tune\uci\<run>\eng
           -engine cmd=<opponent exe> name=<opponent> dir=<opponent dir> [-engine ... more opponents]
           -each tc=10+0.1 proto=uci -tournament gauntlet -seeds 1 -rounds <r> -games 2 -repeat -concurrency 8
           -openings file=C:\Learn\Tune\hvar\open\open-moves-uniq-fm1.fen format=epd order=sequential start=<first>
           -pgnout file=C:\Learn\Tune\uci\<run>\games.pgn -recover -ratinginterval 50

   (`run_uci_games.ps1` builds this command and keeps the opening offset between runs.)

2. Join (name = the fastchess `name=` of Barbarossa), whole games round robin into `--shards` files:

       python uci_data.py --data C:\Learn\Tune\uci\<run>\eng --pgn C:\Learn\Tune\uci\<run>\games.pgn --engine Barbarossa --out C:\Learn\Tune\uci\data --name <run>

   It prints the games, the engine's moves, the rows written and what was dropped. The searches of a process
   are cut into games (ucinewgame, start position, ply) and every game segment is matched as a whole to the
   one PGN game with the same start position and the same root positions at the same plies.

3. Then as for self-play data: `hv_prep_sp.py --src C:\Learn\Tune\uci\data --val <run>-s4 --out ...`, export
   with `TuneSGD -X`, `hv_fit.py`.

Notes:
- Moves without a row: searches without a leaf (no root iteration found a new best path), mostly mate and
  draw scores; in the test (20 games against `boas`, 10+0.1) 15% of the moves, 2.4% with normal scores
  (SelfPlay: 6%, it resigns at -1200 and stops at mate in 1).
- The games are played out to mate (no resignation): about 10% of the rows have |score| > 2000, which
  `hv_prep_sp.py` drops (`--max-score`).
- Fallback without `-d`: `--logs <dir>` reads the Info logs of a branch build started without `-l 5` (last
  Origin line before each bestmove; 1.4 MB per game). Builds before `ucidat` name the log by the start second
  only, so engines started in the same second write interleaved lines into one file: such logs give wrong
  leaves (searches cut at another process's bestmove) and are not usable.
