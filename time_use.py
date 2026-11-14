"""Time use per game from Barbarossa logs (Info level): clock at moves 20 and 40 as share of the start, moves under
10% of the start, moves that ran to their maximum (mxr of compTime), mean target and used time, mean draft, LMR level at
the last draft (share at 15 or more). Usage: time_use.py <log> ..."""
import re, sys

def games_of(fn):
    games, cur, s, prev = [], None, None, ""
    for line in open(fn, encoding="utf-8", errors="replace"):
        if "[Info]: New game" in line:
            cur = []; games.append(cur); s = None
        m = re.search(r"\[Info\]: Time = (\d+) Timx1", line)
        if m and re.search(r"starts draft 1$", prev.rstrip()):
            s = {"clock": int(m.group(1)), "t0": int(line.split()[0]), "draft": 0, "lv": None, "ms": None, "mxr": None}
            if cur is None:
                cur = []; games.append(cur)
            cur.append(s)
        if s is not None:
            m = re.search(r"Draft (\d+) Score (-?\d+) ", line)
            if m: s["draft"] = int(m.group(1)); s["score"] = int(m.group(2))
            m = re.search(r"Variable LMR: hi = \d+, lv = (\d+)", line)
            if m: s["lv"] = int(m.group(1))
            m = re.search(r"compTime \(ms/mx/mxr\): (\d+) / (\d+) / (\d+)", line)
            if m: s["ms"], s["mxr"] = int(m.group(1)), int(m.group(3))
            m = re.match(r"(\d+) \[Output\]: bestmove", line)
            if m: s["used"] = int(m.group(1)) - s["t0"]
        prev = line
    return [g for g in games if len(g) >= 10]

for fn in sys.argv[1:]:
    gs = games_of(fn)
    print(f"{fn.split('/')[-1]}: {len(gs)} games (10 moves or more)")
    print("  moves  start  @20    @40    <10%  atmax  target  used  draft  lv  lv15")
    tot = {"n": 0, "low": 0, "max": 0, "c20": [], "c40": []}
    for g in gs:
        st = g[0]["clock"]
        c20 = g[19]["clock"] / st if len(g) >= 20 else None
        c40 = g[39]["clock"] / st if len(g) >= 40 else None
        low = sum(1 for x in g if x["clock"] < 0.1 * st)
        mv = [x for x in g if x.get("used") is not None and x["mxr"]]
        atmax = sum(1 for x in mv if x["used"] >= 0.9 * x["mxr"])
        tgt = sum(x["ms"] for x in mv) / max(1, len(mv)) / 1000
        used = sum(x["used"] for x in mv) / max(1, len(mv)) / 1000
        lvs = [x["lv"] for x in g if x["lv"] is not None]
        f = lambda v: "  -  " if v is None else f"{v:5.0%}"
        print(f"  {len(g):5d} {st/1000:5.0f}s {f(c20)}  {f(c40)}  {low:4d}  {atmax:5d}  {tgt:5.1f}s {used:5.1f}s"
              f"  {sum(x['draft'] for x in g)/len(g):5.1f} {sum(lvs)/max(1,len(lvs)):4.1f} {sum(v>=15 for v in lvs)/max(1,len(lvs)):4.0%}")
        tot["n"] += len(g); tot["low"] += low; tot["max"] += atmax
        if c20 is not None: tot["c20"].append(c20)
        if c40 is not None: tot["c40"].append(c40)
    if gs:
        print(f"  all: {tot['n']} moves, clock at 20 mean {sum(tot['c20'])/max(1,len(tot['c20'])):.0%}, at 40 "
              f"{sum(tot['c40'])/max(1,len(tot['c40'])):.0%}, under 10%: {tot['low']/tot['n']:.0%} of the moves, "
              f"at max: {tot['max']/tot['n']:.0%}")
