"""Time use, mate scores and score drops per move from a Barbarossa UCI log (Info level), for the review
`TIME-MATE-EXTENSION-REVIEW.md` (rel090, 2026-11-15): moves at the maximum (used >= 90% of the mxr of the draft before),
by final score class, score drop against the last move, mate scores (seen and gone, research at a mate bound), queen
endings and seldepth over 2 * draft, longest last iterations. Games with 10 moves or more. mate_time2.py and
mate_time3.py (same folder) build on it. Usage: mate_time.py <log>
"""
import re, sys, collections
fn = sys.argv[1]
games, cur, s, prev = [], None, None, ""
for line in open(fn, encoding="utf-8", errors="replace"):
    if "[Info]: New game" in line:
        cur = []; games.append(cur); s = None
    m = re.search(r"\[Info\]: Time = (\d+) Timx1", line)
    if m and prev.rstrip().endswith("starts draft 1"):
        s = {"clock": int(m.group(1)), "t0": int(line.split()[0]), "dr": [], "res": 0, "resm": 0, "fen": None,
             "flags": None, "used": None, "sd": None, "lv": None, "nodes": None, "exteline": True}
        if cur is None:
            cur = []; games.append(cur)
        cur.append(s)
    if s is not None:
        if "Research: a = " in line:
            s["res"] += 1
            m = re.search(r"a = (-?\d+), s1 = Path \{pathScore = (-?\d+)", line)
            if m and (abs(int(m.group(1))) >= 19000 or abs(int(m.group(2))) >= 19000): s["resm"] += 1
        m = re.search(r"NoEval \|([^|]+)\|", line)
        if m and s["fen"] is None: s["fen"] = m.group(1)
        m = re.search(r"\]: Nodes: (\d+)", line)
        if m: s["nodes"] = int(m.group(1))
        m = re.search(r"\]: seldepth (\d+) score", line)
        if m: s["sd"] = int(m.group(1))
        m = re.search(r"Variable LMR: hi = \d+, lv = (\d+)", line)
        if m: s["lv"] = int(m.group(1))
        m = re.search(r"\]: Draft (\d+) Score (-?\d+) path .* ms (\d+) used (\d+)", line)
        if m:
            s["dr"].append({"d": int(m.group(1)), "sc": int(m.group(2)), "ms": int(m.group(3)), "used": int(m.group(4)),
                            "res": s["res"], "resm": s["resm"], "sd": s["sd"], "lv": s["lv"], "nodes": s["nodes"],
                            "sbc": False})
            s["res"] = 0; s["resm"] = 0
        m = re.search(r"Time factors \(reds/redp\): ([\d.e+-]+)", line)
        if m and s["dr"]: s["dr"][-1]["reds"] = float(m.group(1))
        if "stopByChance: draft" in line and s["dr"]: s["dr"][-1]["sbc"] = True
        m = re.search(r"compTime \(ms/mx/mxr\): (\d+) / (\d+) / (\d+)", line)
        if m and s["dr"]: s["dr"][-1]["mx"], s["dr"][-1]["mxr"] = int(m.group(2)), int(m.group(3))
        m = re.search(r"terminated in first if: (\w+)/(\w+)/(\w+)/(\w+)/(\w+)", line)
        if m: s["flags"] = [x == "True" for x in m.groups()]
        m = re.match(r"(\d+) \[Output\]: bestmove", line)
        if m and s["used"] is None: s["used"] = int(m.group(1)) - s["t0"]
    prev = line

games = [g for g in games if len(g) >= 10]
MATE = 19000
def npieces(fen):
    b = fen.split()[0]
    return sum(c.isalpha() for c in b), sum(c in "qQ" for c in b)

allm = []
for gi, g in enumerate(games):
    gls = None
    for mi, x in enumerate(g):
        x["g"], x["m"] = gi + 1, mi + 1
        if not x["dr"] or x["used"] is None: continue
        d = x["dr"]
        x["fin"] = d[-1]["sc"]
        # the time limit valid for the last iteration is the mxr computed after the draft before
        x["lim"] = d[-2]["mxr"] if len(d) >= 2 else d[-1].get("mxr")
        x["atmax"] = x["lim"] is not None and x["used"] >= 0.9 * x["lim"]
        x["tgt"] = d[-1]["ms"]
        # exte per draft (los = draft before, gls = last move's score)
        ne = 0; jumps = 0; big = 0; mflip = 0
        for i, e in enumerate(d):
            los = d[i-1]["sc"] if i > 0 else None
            ex = (los is not None and gls is not None) and (e["sc"] < los - 24 or e["sc"] < gls - 24)
            e["exte"] = ex; ne += ex
            if los is not None:
                dl = abs(e["sc"] - los)
                if dl >= 100 and e["d"] >= 6: jumps += 1
                if (abs(e["sc"]) >= MATE) != (abs(los) >= MATE):
                    if abs(los) >= MATE: mflip += 1   # mate score seen, gone at the next draft
        x["nexte"], x["jumps"], x["mflip"] = ne, jumps, mflip
        x["extelast"] = len(d) >= 2 and d[-2]["exte"]
        x["restot"] = sum(e["res"] for e in d); x["resm"] = sum(e["resm"] for e in d)
        x["gls"] = gls
        x["drop"] = (gls - x["fin"]) if gls is not None else 0
        x["lastit"] = d[-1]["used"] - (d[-2]["used"] if len(d) >= 2 else 0)
        x["np"], x["nq"] = npieces(x["fen"]) if x["fen"] else (99, 0)
        x["sdr"] = max((e["sd"] or 0) / e["d"] for e in d if e["d"] >= 6) if any(e["d"] >= 6 for e in d) else 0
        gls = x["fin"]
        allm.append(x)

def cls(x):
    f = x["fin"]
    if f <= -MATE: return "1 mated"
    if f >= MATE: return "6 mating"
    if f <= -500: return "2 <= -500"
    if f <= -150: return "3 -150..-500"
    if f < 150: return "4 open"
    return "5 >= +150"

print(f"{len(games)} games, {len(allm)} moves, at max {sum(x['atmax'] for x in allm)}")
print("\nby final score class: moves, at max, used s, target s, used/target, timint, over, mean research, exte-last, mean draft")
by = collections.defaultdict(list)
for x in allm: by[cls(x)].append(x)
for k in sorted(by):
    v = by[k]; n = len(v)
    print(f"  {k:14s} {n:4d} {sum(x['atmax'] for x in v):4d} ({sum(x['atmax'] for x in v)/n:4.0%}) "
          f"{sum(x['used'] for x in v)/1000:7.1f} {sum(x['tgt'] for x in v)/1000:7.1f} {sum(x['used'] for x in v)/max(1,sum(x['tgt'] for x in v)):5.2f} "
          f" ti {sum(x['flags'][1] for x in v if x['flags']):4d} ov {sum(x['flags'][2] for x in v if x['flags']):4d} "
          f" res {sum(x['restot'] for x in v)/n:5.1f}  exl {sum(x['extelast'] for x in v):4d}  dr {sum(x['dr'][-1]['d'] for x in v)/n:5.1f}")

am = [x for x in allm if x["atmax"]]
print(f"\nmoves at max: {len(am)}; time {sum(x['used'] for x in am)/1000:.0f}s of {sum(x['used'] for x in allm)/1000:.0f}s")
print(f"  with exte before last iteration: {sum(x['extelast'] for x in am)}; any exte: {sum(x['nexte']>0 for x in am)}")
print(f"  with drop vs last move >= 24: {sum(x['drop']>=24 for x in am)}, >= 100: {sum(x['drop']>=100 for x in am)}")
print(f"  with score jump >= 100 between drafts (d>=6): {sum(x['jumps']>0 for x in am)} (all moves: {sum(x['jumps']>0 for x in allm)})")
print(f"  flags timint {sum(x['flags'][1] for x in am)}, over {sum(x['flags'][2] for x in am)}, juststop {sum(x['flags'][4] for x in am)}, draftmax {sum(x['flags'][0] for x in am)}")
print(f"  seldepth > 2*draft (d>=6): {sum(x['sdr']>2 for x in am)} (all: {sum(x['sdr']>2 for x in allm)}); > 3*draft: {sum(x['sdr']>3 for x in am)} (all: {sum(x['sdr']>3 for x in allm)})")
print(f"  research: mean {sum(x['restot'] for x in am)/len(am):.1f} at max, {sum(x['restot'] for x in allm if not x['atmax'])/max(1,len(allm)-len(am)):.1f} others")

print("\nmate scores:")
mm = [x for x in allm if any(abs(e['sc']) >= MATE for e in x['dr'])]
print(f"  moves with a mate score at some draft: {len(mm)} (against: {sum(any(e['sc']<=-MATE for e in x['dr']) for x in mm)}, for: {sum(any(e['sc']>=MATE for e in x['dr']) for x in mm)})")
print(f"  of them at max: {sum(x['atmax'] for x in mm)}; time {sum(x['used'] for x in mm)/1000:.1f}s; mean used {sum(x['used'] for x in mm)/len(mm):.0f} ms")
print(f"  mate score seen and gone at the next draft (within a search): {sum(x['mflip'] for x in allm)} times in {sum(x['mflip']>0 for x in allm)} moves")
fl = 0; first = []
for g in games:
    pm = None
    for x in g:
        if "fin" not in x: continue
        if pm is not None and abs(pm) >= MATE and abs(x["fin"]) < MATE: fl += 1
        if (pm is None or abs(pm) < MATE) and abs(x["fin"]) >= MATE: first.append(x)
        pm = x["fin"]
print(f"  mate at the end of a move, no mate at the end of the next: {fl}")
print(f"  first move with final mate score (per run of mate scores): {len(first)}")
for x in first:
    print(f"    g{x['g']} m{x['m']} fin {x['fin']} gls {x['gls']} clock {x['clock']} used {x['used']} lim {x['lim']} tgt {x['tgt']} drafts {x['dr'][-1]['d']} atmax {x['atmax']} flags {x['flags']} np {x['np']}")
print(f"  research lines with a mate bound (a or s1): {sum(x['resm'] for x in allm)} of {sum(x['restot'] for x in allm)}")
dm = [x for x in allm if x['flags'] and x['flags'][0]]
print(f"  moves ended by draftmax: {len(dm)}, drafts {sorted(set(x['dr'][-1]['d'] for x in dm))}, mean used {sum(x['used'] for x in dm)/max(1,len(dm)):.0f} ms")

print("\nnear mate / lost (final <= -500, no mate): per game")
for gi, g in enumerate(games):
    v = [x for x in g if "fin" in x]
    lost = [x for x in v if -MATE < x["fin"] <= -500]
    mt = [x for x in v if abs(x["fin"]) >= MATE]
    print(f"  g{gi+1}: {len(v)} moves, last {v[-1]['fin']}, lost-phase moves {len(lost)} atmax {sum(x['atmax'] for x in lost)} time {sum(x['used'] for x in lost)/1000:.1f}s clock at its start {lost[0]['clock'] if lost else '-'}; mate-phase {len(mt)} atmax {sum(x['atmax'] for x in mt)} time {sum(x['used'] for x in mt)/1000:.1f}s")

print("\nscore drop (final vs last move's final) and time:")
for lo, hi in [(-10**6, -24), (-24, 24), (24, 100), (100, 300), (300, MATE), (MATE, 10**6)]:
    v = [x for x in allm if x["gls"] is not None and lo <= x["drop"] < hi]
    if v: print(f"  drop [{lo},{hi}): {len(v):4d} moves, at max {sum(x['atmax'] for x in v):4d} ({sum(x['atmax'] for x in v)/len(v):4.0%}), used/target {sum(x['used'] for x in v)/max(1,sum(x['tgt'] for x in v)):.2f}, used {sum(x['used'] for x in v)/1000:.0f}s")

print("\nqueen endings and few pieces (root): pieces incl. kings and pawns")
for name, f in [("queens, <= 8 pieces", lambda x: x["nq"] > 0 and x["np"] <= 8), ("queens, 9-14 pieces", lambda x: x["nq"] > 0 and 9 <= x["np"] <= 14),
                ("no queens, <= 14", lambda x: x["nq"] == 0 and x["np"] <= 14), ("rest", lambda x: x["np"] > 14)]:
    v = [x for x in allm if f(x)]
    if not v: print(f"  {name}: none"); continue
    n = len(v)
    print(f"  {name}: {n} moves, at max {sum(x['atmax'] for x in v)} ({sum(x['atmax'] for x in v)/n:.0%}), used/target {sum(x['used'] for x in v)/max(1,sum(x['tgt'] for x in v)):.2f}, "
          f"mean draft {sum(x['dr'][-1]['d'] for x in v)/n:.1f}, sd>2d {sum(x['sdr']>2 for x in v)}, sd>3d {sum(x['sdr']>3 for x in v)}, "
          f"last it share {sum(x['lastit'] for x in v)/max(1,sum(x['used'] for x in v)):.0%}, lv>=15 {sum((x['dr'][-1]['lv'] or 0)>=15 for x in v)/n:.0%}, mate fin {sum(abs(x['fin'])>=MATE for x in v)}")

print("\nresearch per move, by drafts: mean per draft; share of moves with >= 5 in one draft")
print(f"  moves with >= 5 Research in one draft: {sum(any(e['res']>=5 for e in x['dr']) for x in allm)}; of them at max {sum(x['atmax'] for x in allm if any(e['res']>=5 for e in x['dr']))}")
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 1000)]:
    v = [x for x in allm if lo <= x["restot"] < hi]
    if v: print(f"  research [{lo},{hi}): {len(v):4d} moves, at max {sum(x['atmax'] for x in v)/len(v):4.0%}, mean draft {sum(x['dr'][-1]['d'] for x in v)/len(v):.1f}")
# slow last iterations
sl = sorted(allm, key=lambda x: -x["lastit"])[:15]
print("\nlongest last iterations:")
for x in sl:
    print(f"  g{x['g']} m{x['m']} lastit {x['lastit']} used {x['used']} lim {x['lim']} tgt {x['tgt']} fin {x['fin']} gls {x['gls']} d {x['dr'][-1]['d']} sd {x['dr'][-1]['sd']} res {x['dr'][-1]['res']} np {x['np']} q {x['nq']} flags {x['flags']}")
