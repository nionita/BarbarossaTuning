"""Load the files exported by TuneSGD -X (<name>-exp.csv) and build the piece-square features.

Export columns: fen, score, result, eval, phase, scale, m50, dmid, dend, const, then one feature
column per HCE weight. With the start weights (weights.csv in the same directory):
    eval ~ sum_i f_i * (wmid_i * dmid + wend_i * dend) + const
A term with feature z and weights zmid / zend (units of the engine: 1/8 cp) adds
    z * (zmid * dmid + zend * dend)
which is how the PST is added here.

PST (antisymmetric, side to move POV): every piece contributes +1 (piece of the side to move) or
-1 (opponent piece) to the column type * 64 + square, the square seen from the piece owner's side
(a1 = 0 .. h8 = 63, mirrored vertically for black pieces). So one table of 6 x 64 values per phase
serves both colours; pawns use only 48 squares.

Parsing a file is slow, so the arrays are cached next to it as <name>.npz (rebuilt when the CSV is
newer). Several files are parsed in parallel.
"""

from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import env_dlls  # noqa: F401  (before numpy)
import numpy as np
import pandas as pd
import scipy.sparse as sp

FIXED_COLS = ["fen", "score", "result", "eval", "phase", "scale", "m50", "dmid", "dend", "const"]
PIECE_INDEX = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
PIECE_NAMES = "PNBRQK"
N_PST = 6 * 64


def pst_entries(fens) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR parts (indptr, column, sign) of the PST feature matrix for a sequence of FENs."""
    indptr = np.zeros(len(fens) + 1, dtype=np.int64)
    cols: list[int] = []
    signs: list[int] = []
    for i, fen in enumerate(fens):
        board, stm = fen.split(" ", 2)[:2]
        mover_white = stm == "w"
        sq = 56                         # a8
        for ch in board:
            if ch == "/":
                sq -= 16                # end of a rank: a-file of the rank below
            elif ch <= "9":
                sq += ord(ch) - 48
            else:
                white = ch < "a"
                t = PIECE_INDEX[ch.lower()]
                cols.append(t * 64 + (sq if white else sq ^ 56))
                signs.append(1 if white == mover_white else -1)
                sq += 1
        indptr[i + 1] = len(cols)
    return indptr, np.array(cols, dtype=np.int16), np.array(signs, dtype=np.int8)


def _parse_file(path: str) -> str:
    """Parse one export CSV into its .npz cache, return the cache name."""
    cache = os.path.splitext(path)[0] + ".npz"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return cache
    df = pd.read_csv(path, engine="pyarrow")
    names = [c for c in df.columns if c not in FIXED_COLS]
    indptr, cols, signs = pst_entries(df["fen"].tolist())
    np.savez(cache,
             names=np.array(names),
             score=df["score"].to_numpy(np.float64),
             result=df["result"].to_numpy(np.int8),
             eval=df["eval"].to_numpy(np.float64),
             phase=df["phase"].to_numpy(np.int16),
             scale=df["scale"].to_numpy(np.int16),
             m50=df["m50"].to_numpy(np.int16),
             dmid=df["dmid"].to_numpy(np.float64),
             dend=df["dend"].to_numpy(np.float64),
             const=df["const"].to_numpy(np.float64),
             feats=df[names].to_numpy(np.float32),
             indptr=indptr, cols=cols, signs=signs)
    return cache


@dataclass
class Data:
    """Records of one or more export files (rows with |score| > max_score already removed)."""
    names: list[str]        # HCE weight names, the columns of feats
    score: np.ndarray       # target score of the data (gen3 static eval for chilo)
    result: np.ndarray      # 0 / 1 / 2 = loss / draw / win, side to move POV
    eval: np.ndarray        # HCE eval with the start weights (cp)
    phase: np.ndarray
    scale: np.ndarray
    m50: np.ndarray
    dmid: np.ndarray        # cp per unit of mid weight and unit of feature
    dend: np.ndarray        # cp per unit of end weight and unit of feature
    const: np.ndarray       # weight independent part of eval (cp)
    feats: np.ndarray       # N x 46 HCE features (float32)
    pst: sp.csr_matrix      # N x 384 PST features, +-1 (float32)
    files: list[str]
    src: np.ndarray         # source index of every record (into sources; 0 when not given)
    sources: list[str]      # source names (from the file names, see load)

    @property
    def n(self) -> int:
        return len(self.result)


def export_files(path: str) -> list[str]:
    """The export CSVs of a directory (sorted), or the file itself."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*-exp.csv")))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No *-exp.csv files in {path}")
    return files


def read_weights(directory: str) -> dict[str, tuple[int, int]]:
    """Start weights written by the export: name -> (mid, end)."""
    df = pd.read_csv(os.path.join(directory, "weights.csv"))
    return {r.name: (int(r.mid), int(r.end)) for r in df.itertuples(index=False)}


def source_of(path: str, source_re: str | None) -> str:
    """Source name of an export file: the first group of source_re matched on the file name ("all"
    without a pattern). Records of one source (e.g. one opponent, or self-play) share an offset."""
    if not source_re:
        return "all"
    m = re.search(source_re, os.path.basename(path))
    if not m:
        raise ValueError(f"{path}: file name does not match the source pattern {source_re}")
    return m.group(1)


def load(files: list[str], max_score: float = 2000, workers: int = 8, source_re: str | None = None,
         sources: list[str] | None = None) -> Data:
    """Load (and cache) the given export files and concatenate them. The source of each file comes
    from its name (source_of); with sources given (e.g. those of the training data) the indices refer
    to that list and an unknown source is an error."""
    if workers <= 1 or len(files) == 1:
        caches = [_parse_file(f) for f in files]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(files))) as ex:
            caches = list(ex.map(_parse_file, files))
    parts = [np.load(c) for c in caches]
    names = [str(x) for x in parts[0]["names"]]
    for c, p in zip(caches, parts):
        if [str(x) for x in p["names"]] != names:
            raise ValueError(f"Feature columns of {c} differ from {caches[0]}")
    cat = {k: np.concatenate([p[k] for p in parts])
           for k in ("score", "result", "eval", "phase", "scale", "m50", "dmid", "dend", "const", "feats")}
    mats = [sp.csr_matrix((p["signs"].astype(np.float32), p["cols"].astype(np.int32), p["indptr"]),
                          shape=(len(p["result"]), N_PST)) for p in parts]
    pst = sp.vstack(mats, format="csr")
    names_src = [source_of(f, source_re) for f in files]
    if sources is None:
        sources = sorted(set(names_src), key=names_src.index)     # in the order of the files
    for f, s in zip(files, names_src):
        if s not in sources:
            raise ValueError(f"{f}: source {s} is not among {sources}")
    cat["src"] = np.concatenate([np.full(len(p["result"]), sources.index(s), dtype=np.int16)
                                 for p, s in zip(parts, names_src)])
    keep = np.abs(cat["score"]) <= max_score if max_score > 0 else np.ones(len(cat["score"]), bool)
    if not keep.all():
        cat = {k: v[keep] for k, v in cat.items()}
        pst = pst[keep]
    return Data(names=names, pst=pst, files=list(files), sources=list(sources), **cat)


def pst_used(data: Data) -> np.ndarray:
    """Boolean mask of the PST columns which occur in the data."""
    return np.asarray((abs(data.pst)).sum(axis=0)).ravel() > 0
