"""Score-drop path, second pass: counterfactual without drop handling (exte off, reds = 1) and simulated rules.
Usage: score_drop2.py <log> ...  (needs score_drop.py beside it)"""
import sys, os, collections, importlib.util, io, contextlib
here = os.path.dirname(os.path.abspath(sys.argv[0]))
spec = importlib.util.spec_from_file_location("sd", os.path.join(here, "score_drop.py"))
sd = importlib.util.module_from_spec(spec)
argv = sys.argv; sys.argv = [argv[0]]
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(sd)
sys.argv = argv
TPM, MARG = sd.TPM, sd.MARG
P = 0.5   # unknown change probability in mxr: mxr = min mx (red * mx * (1 + p) / 2)

def simulate(x, capA=False, exte="now", reds="now", K=None, redcap=None, incs=60):
    d = x["dr"]; n = len(d)
    for i in range(n - 1):
        e = d[i]
        mx = min(e["mx"], sd.mx_A(x["clock"], e["ms"], incs=incs)) if capA else e["mx"]
        ex = {"now": e["exte"], "L": e["dl"], "off": False}[exte]
        red = e["reds"] if reds == "now" else 1.0
        if redcap: red = min(red, redcap)
        base = e["redp"] * e["ms"]
        msr = red * base
        drop = ex or red > 1
        if K is not None and drop:
            cap = base + K * TPM
            msr = min(msr, cap)
            if e["used"] >= cap: ex = False
        if e["used"] >= mx: return e["used"], e["mv"]
        if not ex and e["used"] >= msr: return e["used"], e["mv"]
        lim = mx if ex else min(mx, red * e["redp"] * mx * (1 + P) / 2)
        if K is not None and drop:
            lim = min(lim, max(base + K * TPM, min(mx, e["redp"] * mx * (1 + P) / 2)))
        nxt = d[i+1]["used"] if i + 1 < n - 1 else x["used"]
        if nxt > lim + 20: return max(lim, e["used"]), e["mv"]
    return x["used"], x["fmv"]

def report(name, allm):
    tot = sum(x["used"] for x in allm)
    print(f"\n===== {name}: {len(allm)} moves, {tot/1000:.0f}s; clock above 60 increments in {sum(x['clock']>60*TPM for x in allm)} moves")
    # sanity of mx_A without cap against the logged mx
    bad = sum(abs(sd.mx_A(x["clock"], x["dr"][0]["ms"], incs=10**6) - x["dr"][0]["mx"]) > 3 for x in allm)
    print(f"sanity: mx formula differs from the log in {bad} moves; simulation of 'now' differs in time by {sum(abs(simulate(x)[0]-x['used']) for x in allm)/1000:.0f}s, other move in {sum(simulate(x)[1]!=x['fmv'] for x in allm)}")
    S = [x for x in allm if any(e["exte"] or e["reds"] > 1 for e in x["dr"][:-1])]
    print(f"moves with a drop signal (exte or reds > 1) at some draft before the last: {len(S)} ({sum(x['used'] for x in S)/1000:.0f}s)")
    # counterfactual: no drop handling
    rows = []
    for x in S:
        t, mv = simulate(x, exte="off", reds="off")
        if t < x["used"] - 20:
            # the draft whose result would have been played
            i = max(k for k, e in enumerate(x["dr"][:-1]) if e["used"] <= t + 1)
            rows.append((x, t, mv, i))
    extra = sum(x["used"] - t for x, t, mv, i in rows)
    print(f"drop moves that got extra time from the drop handling: {len(rows)}, extra {extra/1000:.0f}s ({extra/tot:.1%} of all time), mean {extra/max(1,len(rows))/TPM:.1f} increments")
    c = collections.Counter(); tm = collections.Counter()
    for x, t, mv, i in rows:
        s0, f, gls = x["dr"][i]["sc"], x["fin"], x["gls"]
        if gls is not None and f >= gls - MARG: sc = "recovered fully"
        elif f >= s0 + MARG: sc = "recovered partly"
        elif f <= s0 - MARG: sc = "deeper"
        else: sc = "confirmed"
        c[(sc, mv != x["fmv"])] += 1; tm[(sc, mv != x["fmv"])] += x["used"] - t
    for sc in ["recovered fully", "recovered partly", "confirmed", "deeper"]:
        print(f"  {sc:17s} same move {c[(sc, False)]:4d} ({tm[(sc, False)]/1000:6.0f}s)   move changed {c[(sc, True)]:4d} ({tm[(sc, True)]/1000:6.0f}s)")
    print(f"  move changed {sum(v for k, v in c.items() if k[1])} of {len(rows)}; extra time in moves with the same move {sum(v for k, v in tm.items() if not k[1])/1000:.0f}s")
    # where the extra went: by move number, by size
    for lo, hi in [(1, 10), (11, 20), (21, 40), (41, 999)]:
        v = [(x, t, mv) for x, t, mv, i in rows if lo <= x["m"] <= hi]
        print(f"  moves {lo}-{hi}: {len(v)} moves, extra {sum(x['used']-t for x, t, mv in v)/1000:.0f}s, move changed {sum(mv != x['fmv'] for x, t, mv in v)}")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 100)]:
        v = [(x, t, mv) for x, t, mv, i in rows if lo <= (x["used"] - t) / TPM < hi]
        print(f"  extra {lo}-{hi} increments: {len(v)} moves, extra {sum(x['used']-t for x, t, mv in v)/1000:.0f}s, move changed {sum(mv != x['fmv'] for x, t, mv in v)}")
    # when the final move settled, for the changed ones, in increments above the no-drop stop
    ch = [(x, t) for x, t, mv, i in rows if mv != x["fmv"]]
    print("  changed moves: final move settled within 1 / 2 / 3 / 5 / 8 increments after the no-drop stop: " + " / ".join(str(sum((x["settle"] - t) / TPM <= k for x, t in ch)) for k in (1, 2, 3, 5, 8)))
    drop_sz = collections.Counter()
    for x, t, mv, i in rows:
        dd = max((x["gls"] - e["sc"]) if x["gls"] is not None else 0 for e in x["dr"])
        b = "24-50" if dd < 50 else "50-100" if dd < 100 else "100-300" if dd < 300 else "300+"
        if dd < 24: b = "<24"
        drop_sz[b] += 1; drop_sz[b + " ch"] += mv != x["fmv"]; drop_sz[b + " t"] += x["used"] - t
    print("  by largest drop against the last move: " + "; ".join(f"{b}: {drop_sz[b]} moves, {drop_sz[b+' ch']} changed, {drop_sz[b+' t']/1000:.0f}s" for b in ["<24", "24-50", "50-100", "100-300", "300+"]))
    print("rules: time, saved against now (simulated), moves with less time, other move played; of the lost changes: with a higher final score than at the stop")
    t0 = sum(simulate(x)[0] for x in allm)
    for label, kw in [("no drop handling", dict(exte="off", reds="off")), ("A", dict(capA=True)), ("A + no drop handling", dict(capA=True, exte="off", reds="off")),
                      ("L (last draft only)", dict(exte="L")), ("L, reds off", dict(exte="L", reds="off")), ("exte off, reds now", dict(exte="off")),
                      ("reds cap 2", dict(redcap=2.0)), ("K=2", dict(K=2)), ("K=3", dict(K=3)), ("K=5", dict(K=5)), ("K=8", dict(K=8)),
                      ("A + L", dict(capA=True, exte="L")), ("A + K=3", dict(capA=True, K=3)), ("A + K=5", dict(capA=True, K=5)), ("A + reds cap 2", dict(capA=True, redcap=2.0)),
                      ("A + L + reds cap 2", dict(capA=True, exte="L", redcap=2.0))]:
        tt = 0; aff = 0; oth = 0
        for x in allm:
            t, mv = simulate(x, **kw); tb = simulate(x)[0]
            tt += t
            if t < tb - 20:
                aff += 1; oth += mv != x["fmv"]
        print(f"  {label:22s} {tt/1000:6.0f}s  saved {(t0-tt)/1000:6.0f}s ({(t0-tt)/t0:5.1%})  moves {aff:4d}  other move {oth:4d}")

sets = collections.OrderedDict()
for fn in sys.argv[1:]:
    key = "60+1" if "gui-60p1" in fn else "4+1 chcm"
    sets.setdefault(key, []).extend(sd.prep(sd.parse(fn)))
for k, v in sets.items(): report(k, v)
