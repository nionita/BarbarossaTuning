"""Barbarossa games against other engines (fastchess) -> training data in the format of SelfPlay --data.

Input: the PGN of the games (fastchess -pgnout, the real results) and what Barbarossa wrote about its searches:

- data files (preferred): Barbarossa started with -d (also with -l 5) writes in its working directory
  barbarossa-<version>-<branch>-<start second>-<pid>.data, per search a line
      S|ply|start|root fen|root score|leaf score|leaf fen|best move
  (start: the start position of the GUI's position command, "startpos" or a fen; ply: of this move counted
  from it, from 1; leaf: the evaluated position of the last root iteration, as SelfPlay --data takes it,
  empty when there is none) and a line N at every ucinewgame;
- or Info logs (fallback, large): branch builds without -l 5 log after every root iteration a line
  "Origin |<root fen>|<depth>|<leaf score>|<leaf fen>"; a search ends with its "bestmove" output, a game
  starts with "New game". The root fens of the log have no start position and ply.

Join: the searches of a file are cut into segments (ucinewgame, a new start position, a ply that does not
grow); every search has candidates (game, ply) from the PGN: the engine to move at that ply with the same
position (placement, side, castling, en passant when capturable; no move counters), with data files also
the same start position and ply. A segment goes to the one game that is a candidate of all its searches;
when there is no such game the searches are taken one by one (unique candidates only), when several
games fit all searches (identical games) the segment is dropped as ambiguous.

Output, as SelfPlay --data writes it (hv_prep_sp.py reads it): <out>/<name>-s<i>-filt.csv with rows
fen,score,result (leaf fen as Barbarossa wrote it, static score of the leaf and result 0 / 1 / 2 = loss /
draw / win, both from the leaf's side to move) and <name>-s<i>-filt.csv.meta with the header
game,ply,mover_white,root_score,leaf_white,mover (game number per file from 1, ply of the move in the game
from 1, root score of the last iteration from the mover's side, mate = +-20000). Games go round robin to
--shards files, so whole shards can be held out for validation. Games without a result (*) are not written.

Example:
    python uci_data.py --data C:\\Learn\\Tune\\uci\\run1\\eng --pgn C:\\Learn\\Tune\\uci\\run1\\games.pgn
        --engine Barbarossa --out C:\\Learn\\Tune\\uci\\data --name uci1
"""

from __future__ import annotations

import argparse
import functools
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import chess
import chess.pgn

MATE_SCORE = 20000
LOG_LINE = re.compile(r"^\d+ \[(\w*)\]: (.*)$")
DRAFT_SCORE = re.compile(r"^Draft \d+ Score (-?\d+) ")
INFO_SCORE = re.compile(r" score (cp|mate) (-?\d+)")
START = "startpos"


@dataclass
class Search:
    file: str
    ply: int | None         # ply of the move from the start position (data files)
    start: str | None       # start key of the position command (data files)
    root: str               # root fen (Barbarossa format)
    root_score: int | None  # score of the last iteration, mover's side
    leaf: str               # leaf fen, "" when there is none
    leaf_score: int | None  # static score of the leaf, leaf's side to move


def barbarossa_board(fen: str) -> chess.Board:
    """Board from a Barbarossa fen (placement may end with '/', counters may be missing)."""
    parts = fen.split()
    parts[0] = parts[0].rstrip("/")
    defaults = ["", "w", "-", "-", "0", "1"]
    parts += defaults[len(parts):]
    return chess.Board(" ".join(parts[:6]))


def board_key(board: chess.Board) -> str:
    """Placement, side to move, castling, en passant (only when capturable): no move counters."""
    return " ".join(board.fen(en_passant="legal").split()[:4])


@functools.lru_cache(maxsize=None)
def fen_key(fen: str) -> str | None:
    if fen == START:
        return board_key(chess.Board())
    try:
        return board_key(barbarossa_board(fen))
    except ValueError:
        return None


def to_int(s: str) -> int | None:
    return int(s) if s.strip() else None


def read_data_file(path: str, stats: Counter) -> list[list[Search]]:
    """Segments of the searches of one data file (-d); a segment is cut at N, a new start, a ply not growing."""
    segs: list[list[Search]] = [[]]
    with open(path, encoding="utf-8", errors="replace") as fd:
        for line in fd:
            line = line.rstrip("\r\n")
            if line == "N":
                segs.append([])
                continue
            f = line.split("|")
            if len(f) != 8 or f[0] != "S":
                stats["data lines not understood"] += 1
                continue
            stats["searches"] += 1
            try:
                s = Search(path, int(f[1]), fen_key(f[2]), f[3], to_int(f[4]), f[6], to_int(f[5]))
            except ValueError:
                stats["data lines not understood"] += 1
                continue
            cur = segs[-1]
            if cur and (s.start != cur[-1].start or s.ply <= cur[-1].ply):
                segs.append([])
            segs[-1].append(s)
    return [s for s in segs if s]


def read_log_file(path: str, stats: Counter) -> list[list[Search]]:
    """Segments of the searches of one Info log: one search per bestmove (last Origin line before it)."""
    segs: list[list[Search]] = [[]]
    origin = None
    draft_score = None
    info_score = None
    with open(path, encoding="utf-8", errors="replace") as fd:
        for line in fd:
            m = LOG_LINE.match(line.rstrip("\r\n"))
            if not m:
                continue
            kind, mes = m.groups()
            if kind == "Info":
                if mes.startswith("Origin |"):
                    origin = mes
                elif mes.startswith("Draft "):
                    d = DRAFT_SCORE.match(mes)
                    if d:
                        draft_score = int(d.group(1))
                elif mes == "New game":
                    segs.append([])
            elif kind == "Output":
                if mes.startswith("info ") and " score " in mes:
                    s = INFO_SCORE.search(mes)
                    if s:
                        v = int(s.group(2))
                        info_score = v if s.group(1) == "cp" else (MATE_SCORE if v > 0 else -MATE_SCORE)
                elif mes.startswith("bestmove"):
                    stats["searches"] += 1
                    root_score = draft_score if draft_score is not None else info_score
                    f = origin.split("|") if origin else []
                    if len(f) == 5:
                        try:
                            segs[-1].append(Search(path, None, None, f[1], root_score, f[4], int(f[3])))
                        except ValueError:
                            stats["torn Origin lines"] += 1
                    elif origin:
                        stats["torn Origin lines"] += 1
                    else:
                        stats["searches without leaf"] += 1
                    origin = draft_score = info_score = None
    return [s for s in segs if s]


@dataclass
class Game:
    number: int             # in the PGN files, from 1
    result: int             # engine's result: 0 / 1 / 2 = loss / draw / win
    moves: int              # moves of the engine


def read_pgn(paths: list[str], engine: str, stats: Counter, terminations: Counter):
    """Games of the engine with a result, and the maps position key -> [(game, ply, mover white)] and
    (start key, ply, position key) -> [(game, mover white)] of the positions where the engine moved."""
    games: dict[int, Game] = {}
    by_pos: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    by_ply: dict[tuple[str, int, str], list[tuple[int, bool]]] = defaultdict(list)
    n = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fd:
            while True:
                game = chess.pgn.read_game(fd)
                if game is None:
                    break
                n += 1
                h = game.headers
                white, black = h.get("White") == engine, h.get("Black") == engine
                if white and black:
                    stats["games of the engine against itself (skipped)"] += 1
                    continue
                if not (white or black):
                    stats["games without the engine"] += 1
                    continue
                res = h.get("Result", "*")
                if res not in ("1-0", "0-1", "1/2-1/2"):
                    stats["games without result (skipped)"] += 1
                    continue
                if game.errors:
                    stats["games with PGN errors (skipped)"] += 1
                    continue
                terminations[h.get("Termination", "normal")] += 1
                color = chess.WHITE if white else chess.BLACK
                board = game.board()
                start = board_key(board)
                moves = 0
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    if board.turn == color:
                        k = board_key(board)
                        by_pos[k].append((n, ply, white))
                        by_ply[(start, ply, k)].append((n, white))
                        moves += 1
                    board.push(move)
                white_score = {"1-0": 2, "0-1": 0, "1/2-1/2": 1}[res]
                games[n] = Game(n, white_score if white else 2 - white_score, moves)
    stats["PGN games"] = n
    return games, by_pos, by_ply


def candidates(s: Search, by_pos, by_ply) -> list[tuple[int, int, bool]]:
    k = fen_key(s.root)
    if k is None:
        return []
    if s.ply is not None:
        return [(g, s.ply, w) for g, w in by_ply.get((s.start, s.ply, k), [])]
    return by_pos.get(k, [])


def assign_segment(seg: list[Search], by_pos, by_ply, stats: Counter):
    """[(search, (game, ply, mover white))] of a segment."""
    cands = [candidates(s, by_pos, by_ply) for s in seg]
    nonempty = [c for c in cands if c]
    stats["searches without candidate"] += len(seg) - len(nonempty)
    if not nonempty:
        return []
    common = set(g for g, _, _ in nonempty[0])
    for c in nonempty[1:]:
        common &= set(g for g, _, _ in c)
    out = []
    if len(common) == 1:
        game = common.pop()
        last = 0
        for s, c in zip(seg, cands):
            plies = sorted(x for x in c if x[0] == game and x[1] > last)
            if plies:
                out.append((s, plies[0]))     # a repeated position: the next ply of the game
                last = plies[0][1]
        stats["segments matched to one game"] += 1
        return out
    if len(common) > 1:
        stats["segments fitting several games (dropped)"] += 1
        stats["searches in ambiguous segments (dropped)"] += len(nonempty)
        return []
    stats["segments without a common game (searches one by one)"] += 1
    for s, c in zip(seg, cands):
        if len(c) == 1:
            out.append((s, c[0]))
        elif c:
            stats["ambiguous searches (dropped)"] += 1
    return out


def find_files(paths: list[str], pattern: str) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", pattern), recursive=True))
        else:
            files += sorted(glob.glob(p)) or [p]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", nargs="+", help="data files (-d) or directories (searched recursively)")
    src.add_argument("--logs", nargs="+", help="Info logs or directories (searched recursively), fallback")
    ap.add_argument("--glob", default=None, help="file pattern in directories "
                    "(default barbarossa-*.data, with --logs barbarossa-*.log)")
    ap.add_argument("--pgn", required=True, nargs="+", help="PGN file(s) of the games (fastchess -pgnout)")
    ap.add_argument("--engine", required=True, help="name of Barbarossa in the PGN (fastchess name=)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", required=True, help="run name: files <name>-s<i>-filt.csv (+ .meta)")
    ap.add_argument("--shards", type=int, default=4, help="number of output files (games round robin)")
    ap.add_argument("--mover", default=None, help="mover column of the .meta (default: --engine)")
    args = ap.parse_args()
    stats: Counter = Counter()
    terminations: Counter = Counter()

    is_data = args.data is not None
    pattern = args.glob or ("barbarossa-*.data" if is_data else "barbarossa-*.log")
    files = find_files(args.data if is_data else args.logs, pattern)
    if not files:
        print("No data / log files found", file=sys.stderr)
        return 2
    in_bytes = sum(os.path.getsize(f) for f in files)

    games, by_pos, by_ply = read_pgn(args.pgn, args.engine, stats, terminations)
    if not games:
        print(f"No games of {args.engine} with a result in the PGN", file=sys.stderr)
        return 2

    rows: dict[tuple[int, int], tuple[Search, bool]] = {}
    for f in files:
        segs = read_data_file(f, stats) if is_data else read_log_file(f, stats)
        stats["segments"] += len(segs)
        for seg in segs:
            if is_data:
                for s in seg:
                    if not s.leaf:
                        stats["searches without leaf"] += 1
            for s, (g, ply, white) in assign_segment(seg, by_pos, by_ply, stats):
                if not s.leaf:
                    continue
                if (g, ply) in rows:
                    stats["searches of an already matched move (dropped)"] += 1
                    continue
                rows[(g, ply)] = (s, white)

    for s, _ in rows.values():
        try:
            if not barbarossa_board(s.leaf).is_valid():
                stats["leaf fens not valid (python-chess, written)"] += 1
        except ValueError:
            stats["leaf fens not parsable (python-chess, written)"] += 1

    by_game: dict[int, list[int]] = defaultdict(list)
    for g, ply in rows:
        by_game[g].append(ply)
    os.makedirs(args.out, exist_ok=True)
    mover = (args.mover or args.engine).replace(",", ";")
    nshards = max(1, args.shards)
    names = [os.path.join(args.out, f"{args.name}-s{i + 1}-filt.csv") for i in range(nshards)]
    for fn in names:
        if os.path.exists(fn) or os.path.exists(fn + ".meta"):
            print(f"{fn} exists, not overwritten", file=sys.stderr)
            return 2
    fds = [open(fn, "w", newline="\n") for fn in names]
    fms = [open(fn + ".meta", "w", newline="\n") for fn in names]
    for h in fms:
        h.write("game,ply,mover_white,root_score,leaf_white,mover\n")
    gcount, ncount = [0] * nshards, [0] * nshards
    for i, g in enumerate(sorted(by_game)):
        sh = i % nshards
        gcount[sh] += 1
        for ply in sorted(by_game[g]):
            s, mover_white = rows[(g, ply)]
            leaf_white = s.leaf.split()[1] == "w"
            res = games[g].result if leaf_white == mover_white else 2 - games[g].result
            fds[sh].write(f"{s.leaf},{s.leaf_score},{res}\n")
            root = s.root_score if s.root_score is not None else 0
            fms[sh].write(f"{gcount[sh]},{ply},{int(mover_white)},{root},{int(leaf_white)},{mover}\n")
            ncount[sh] += 1
            if s.root_score is None:
                stats["rows without root score (written as 0)"] += 1
    for h in fds + fms:
        h.close()

    moves = sum(g.moves for g in games.values())
    rcount = Counter(("loss", "draw", "win")[g.result] for g in games.values())
    gaps = Counter(g.moves - len(by_game.get(g.number, [])) for g in games.values())
    kind = "data" if is_data else "log"
    print(f"{kind} files: {len(files)}, {in_bytes / 2**20:.2f} MB, {in_bytes / 2**10 / len(games):.1f} KB per game")
    print(f"PGN games: {stats['PGN games']}, of {args.engine} with a result: {len(games)} "
          f"(+{rcount['win']} ={rcount['draw']} -{rcount['loss']})")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in terminations.most_common()))
    print(f"moves of {args.engine}: {moves}, with a row: {len(rows)} ({100 * len(rows) / max(1, moves):.2f}%), "
          f"games with a row: {len(by_game)}")
    print("games by number of moves without a row: " + ", ".join(f"{k}: {v}" for k, v in sorted(gaps.items())))
    always = ("searches", "segments", "segments matched to one game", "searches without candidate",
              "searches without leaf")
    for k in always + tuple(sorted(set(stats) - set(always) - {"PGN games"})):
        print(f"{k}: {stats[k]}")
    for i in range(nshards):
        print(f"{os.path.basename(names[i])}: {gcount[i]} games, {ncount[i]} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
