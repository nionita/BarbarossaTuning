"""Score-drop path (exte, reds) in Barbarossa logs (Info level): what the extra time changed, and simulated rules.
Assumes an increment of 1 s (TPM = 1000; 4+1 and 60+1). For `TIME-SCORE-DROP-REVIEW.md` (rel090, 2026-11-15).
score_drop2.py and score_drop3.py (same folder) build on it. Usage: score_drop.py <log> ..."""
import re, sys, collections
TPM = 1000; MARG = 24; MATE = 17000

def parse(fn):
    games, cur, s, prev = [], None, None, ""
    for line in open(fn, encoding="utf-8", errors="replace"):
        if "[Info]: New game" in line:
            cur = []; games.append(cur); s = None
        m = re.search(r"\[Info\]: Time = (\d+) Timx1", line)
        if m and prev.rstrip().endswith("starts draft 1"):
            s = {"clock": int(m.group(1)), "t0": int(line.split()[0]), "dr": [], "flags": None, "used": None}
            if cur is None:
                cur = []; games.append(cur)
            cur.append(s)
        if s is not None:
            m = re.search(r"\]: Draft (\d+) Score (-?\d+) path \[([^,\]]*).* ms (\d+) used (\d+)", line)
            if m:
                s["dr"].append({"d": int(m.group(1)), "sc": int(m.group(2)), "mv": m.group(3), "ms": int(m.group(4)),
                                "used": int(m.group(5)), "reds": 1.0, "redp": 1.0, "mx": None, "mxr": None})
            m = re.search(r"Time factors \(reds/redp\): ([\d.e+-]+) / ([\d.e+-]+)", line)
            if m and s["dr"]: s["dr"][-1]["reds"], s["dr"][-1]["redp"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"compTime \(ms/mx/mxr\): (\d+) / (\d+) / (\d+)", line)
            if m and s["dr"]: s["dr"][-1]["mx"], s["dr"][-1]["mxr"] = int(m.group(2)), int(m.group(3))
            m = re.search(r"terminated in first if: (\w+)/(\w+)/(\w+)/(\w+)/(\w+)", line)
            if m: s["flags"] = [x == "True" for x in m.groups()]
            m = re.match(r"(\d+) \[Output\]: bestmove (\S+)", line)
            if m and s["used"] is None: s["used"] = int(m.group(1)) - s["t0"]; s["bm"] = m.group(2)
        prev = line
    return [g for g in games if len(g) >= 10]

def mx_A(tim, ctm, tpm=TPM, incs=60):
    timx = min(tim, incs * tpm); fr = max(0, timx - ctm)
    frac = 0.15 + 0.35 * ctm / max(fr, ctm)
    return max(1, min(round(ctm + frac * fr), tim - 100 if tim > 200 else tim // 2))

def prep(games):
    out = []
    for gi, g in enumerate(games):
        gls = None
        for mi, x in enumerate(g):
            d = x["dr"]
            if not d or x["used"] is None or d[-1]["mx"] is None: continue
            x["g"], x["m"], x["gls"] = gi + 1, mi + 1, gls
            for i, e in enumerate(d):
                los = d[i-1]["sc"] if i > 0 else None
                e["dl"] = los is not None and e["sc"] < los - MARG              # drop against the last draft
                e["dg"] = los is not None and gls is not None and e["sc"] < gls - MARG   # under the last move's score
                e["exte"] = e["dl"] or e["dg"] if gls is not None else False
                e["msr"] = round(e["reds"] * e["redp"] * e["ms"])
                e["ovr"] = e["exte"] and e["used"] >= e["msr"] and e["used"] < e["mx"] and i < len(d) - 1   # exte overrode a sure stop
            x["fin"], x["fmv"] = d[-1]["sc"], x["bm"]
            # settle: used at the end of the first draft from which the best move stays the final one
            k = len(d) - 1
            while k > 0 and d[k-1]["mv"] == d[-1]["mv"]: k -= 1
            x["settle"] = d[k]["used"]; x["settle_i"] = k
            x["d0"] = next((i for i, e in enumerate(d) if e["exte"]), None)
            x["e"] = next((i for i, e in enumerate(d) if e["ovr"]), None)
            gls = x["fin"]
            out.append(x)
    return out

def outcome(x, i):
    d = x["dr"]; s0, f = d[i]["sc"], x["fin"]
    ref = max(v for v in [x["gls"], d[i-1]["sc"] if i > 0 else None] if v is not None)
    mvch = d[i]["mv"] != d[-1]["mv"]
    if f >= ref - MARG: sc = "recovered fully"
    elif f >= s0 + MARG: sc = "recovered partly"
    elif f <= s0 - MARG: sc = "deeper"
    else: sc = "confirmed"
    return mvch, sc

def table(v, key, title):
    print(title)
    c = collections.Counter(); t = collections.Counter();
    for x in v:
        i = x[key]; mvch, sc = outcome(x, i)
        c[(sc, mvch)] += 1; t[(sc, mvch)] += x["used"] - x["dr"][i]["used"]
    n = len(v)
    for sc in ["recovered fully", "recovered partly", "confirmed", "deeper"]:
        a, b = c[(sc, False)], c[(sc, True)]
        print(f"  {sc:17s} same move {a:4d} ({t[(sc, False)]/1000:6.0f}s)   move changed {b:4d} ({t[(sc, True)]/1000:6.0f}s)")
    print(f"  total {n}: move changed {sum(c[k] for k in c if k[1])}, same {sum(c[k] for k in c if not k[1])}; time after that draft {sum(t.values())/1000:.0f}s")

def sim(x, rule, K=None):
    """returns (time, move) under the rule; conservative: only exte decisions that were sure stops are changed"""
    d = x["dr"]; n = len(d)
    if rule == "A":
        lim = mx_A(x["clock"], d[0]["ms"])
        if x["used"] <= lim: return x["used"], x["fmv"]
        j = max([i for i in range(n - 1) if d[i]["used"] <= lim] or [0])
        return max(lim, d[j]["used"]) if d[j]["used"] > lim else lim, d[j]["mv"]
    calm = 0
    for i in range(n - 1):
        e = d[i]
        calm = 0 if e["dl"] else calm + 1
        if not (e["exte"] and e["used"] >= e["msr"] and e["used"] < e["mx"]): continue
        if rule == "L" and not e["dl"]: return e["used"], e["mv"]
        if rule == "C2" and calm >= 2: return e["used"], e["mv"]
        if rule == "K":
            lim = min(e["mx"], e["msr"] + K * TPM)
            if e["used"] >= lim: return e["used"], e["mv"]
            if x["used"] > lim:
                j = max(k for k in range(i, n - 1) if d[k]["used"] <= lim)
                return lim, d[j]["mv"]
        if rule == "KG" and not e["dl"]:          # the cap only for the drop against the last move without a new drop
            lim = min(e["mx"], e["msr"] + K * TPM)
            if e["used"] >= lim: return e["used"], e["mv"]
            nd = next((k for k in range(i + 1, n - 1) if d[k]["dl"] and d[k]["used"] <= lim), None)
            if nd is None and x["used"] > lim:
                j = max(k for k in range(i, n - 1) if d[k]["used"] <= lim)
                return lim, d[j]["mv"]
    return x["used"], x["fmv"]

def report(name, allm):
    tot = sum(x["used"] for x in allm)
    print(f"\n===== {name}: {len(allm)} moves, {tot/1000:.0f}s")
    D = [x for x in allm if x["d0"] is not None]
    E = [x for x in allm if x["e"] is not None]
    print(f"moves with exte at some draft: {len(D)} ({sum(x['used'] for x in D)/1000:.0f}s); with exte overriding a sure stop (used >= reds*redp*ms): {len(E)} ({sum(x['used'] for x in E)/1000:.0f}s, after the override draft {sum(x['used']-x['dr'][x['e']]['used'] for x in E)/1000:.0f}s)")
    if not E: return
    print(f"  E: at max (>=90% mx) {sum(x['used']>=0.9*x['dr'][x['e']]['mx'] for x in E)}, ended timint {sum(x['flags'][1] for x in E)}, over {sum(x['flags'][2] for x in E)}; extra in increments mean {sum(x['used']-x['dr'][x['e']]['used'] for x in E)/len(E)/TPM:.1f}, max {max(x['used']-x['dr'][x['e']]['msr'] for x in E)/TPM:.1f}")
    print(f"  E: trigger at the override draft: new drop against the last draft {sum(x['dr'][x['e']]['dl'] for x in E)}, only under the last move's score {sum(not x['dr'][x['e']]['dl'] for x in E)}")
    print(f"  E by move number: 1-10 {sum(x['m']<=10 for x in E)}, 11-20 {sum(10<x['m']<=20 for x in E)}, 21-40 {sum(20<x['m']<=40 for x in E)}, 41+ {sum(x['m']>40 for x in E)}; time after override 1-10 {sum(x['used']-x['dr'][x['e']]['used'] for x in E if x['m']<=10)/1000:.0f}s, 11-20 {sum(x['used']-x['dr'][x['e']]['used'] for x in E if 10<x['m']<=20)/1000:.0f}s, 21+ {sum(x['used']-x['dr'][x['e']]['used'] for x in E if x['m']>20)/1000:.0f}s")
    print(f"  E by clock: >= 60 incs {sum(x['clock']>=60*TPM for x in E)} ({sum(x['used']-x['dr'][x['e']]['used'] for x in E if x['clock']>=60*TPM)/1000:.0f}s after override), below {sum(x['clock']<60*TPM for x in E)} ({sum(x['used']-x['dr'][x['e']]['used'] for x in E if x['clock']<60*TPM)/1000:.0f}s)")
    table(E, "e", "E: draft where the search would have stopped without exte, against the end:")
    table(D, "d0", "D: draft where the drop was first seen, against the end:")
    # when did the move change come, in increments after the override draft
    ch = [x for x in E if x["dr"][x["e"]]["mv"] != x["fmv"]]
    late = [(x["settle"] - x["dr"][x["e"]]["msr"]) / TPM for x in ch]
    print(f"  E move changes: {len(ch)}; final move settled within 2 / 3 / 5 / 8 increments above msr: " + " / ".join(str(sum(v <= k for v in late)) for k in (2, 3, 5, 8)))
    drafts_after = [len(x["dr"]) - 1 - x["e"] for x in E]
    print(f"  E: drafts completed or started after the override draft: mean {sum(drafts_after)/len(E):.1f}; none completed (only an aborted one): {sum(1 for x in E if len(x['dr'])-1-x['e']==1 and x['flags'][1])}")
    print("rules (simulated on E decisions only; A on all moves): moves changed in time, time saved, final move lost (other move played), of them recoveries lost")
    for rule, K in [("A", None), ("L", None), ("C2", None), ("K", 2), ("K", 3), ("K", 5), ("K", 8), ("KG", 2), ("KG", 3), ("KG", 5)]:
        aff = 0; sav = 0; lost = 0; lostrec = 0; s60 = 0
        for x in allm:
            t, mv = sim(x, rule, K)
            if t < x["used"] - 20:
                aff += 1; sav += x["used"] - t
                if mv != x["fmv"]:
                    lost += 1
        print(f"  {rule}{K or '':<3} affected {aff:4d}  saved {sav/1000:7.0f}s ({sav/tot:5.1%} of all time)  other move {lost:4d}")

for fn in sys.argv[1:]:
    pass
sets = collections.OrderedDict()
for fn in sys.argv[1:]:
    key = "60+1" if "gui-60p1" in fn else ("4+1 0.8.0" if fn.rstrip("/").count("ics-4p1/") else "4+1 chcm")
    sets.setdefault(key, []).extend(prep(parse(fn)))
for k, v in sets.items():
    report(k, v)
