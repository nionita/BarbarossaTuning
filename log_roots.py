"""Root positions of the searches in a Barbarossa UCI log (Info level), for replays with SelfPlay -P (one process per
position). Uses the parser of mate_time.py (same folder): games with 10 moves or more, the root FEN of every search
(first NoEval line), with the clock, the final draft and score, the seldepth / draft ratio and the piece count.
The log writes the board with a trailing '/', removed here (standard FEN).

Writes into <outdir>:
- roots.csv: game,move,clock_ms,used_ms,atmax,draft,final,drop,np,nq,sd_over_d,fen
- roots-all.fen: every root
- roots-mate.fen: roots whose final score is a mate (exactly +-20000; the eval clamp -19744 is not a mate)
- roots-g<N>.fen: the roots of one game, with --game N (repeatable)

Usage: log_roots.py <log> <outdir> [--game N ...]
"""
import contextlib
import io
import os
import runpy
import sys


def main():
    args = sys.argv[1:]
    games = []
    while "--game" in args:
        i = args.index("--game")
        games.append(int(args[i + 1]))
        del args[i:i + 2]
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    log, out = args
    os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.argv = [os.path.join(here, "mate_time.py"), log]
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(sys.argv[0], run_name="x")
    allm = ns["allm"]
    rows = [x for x in allm if x.get("fen")]
    for x in rows:
        b, rest = x["fen"].split(" ", 1)
        x["fen"] = b.rstrip("/") + " " + rest
    with open(os.path.join(out, "roots.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("game,move,clock_ms,used_ms,atmax,draft,final,drop,np,nq,sd_over_d,fen\n")
        for x in rows:
            f.write(f"{x['g']},{x['m']},{x['clock']},{x['used']},{int(x['atmax'])},{x['dr'][-1]['d']},{x['fin']},"
                    f"{x['drop']},{x['np']},{x['nq']},{x['sdr']:.2f},{x['fen']}\n")

    def fens(name, sel):
        v = [x["fen"] for x in rows if sel(x)]
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(s + "\n" for s in v))
        print(f"{name}: {len(v)}")

    fens("roots-all.fen", lambda x: True)
    fens("roots-mate.fen", lambda x: abs(x["fin"]) == 20000)
    for g in games:
        fens(f"roots-g{g}.fen", lambda x, g=g: x["g"] == g)
    print(f"roots.csv: {len(rows)} rows ({len(allm) - len(rows)} moves without a root FEN)")


if __name__ == "__main__":
    main()
