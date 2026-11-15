"""Rule D: drop extra capped at K increments above the target, only on the clock above 60 increments, on top of A."""
import sys, os, importlib.util, io, contextlib, collections
here = os.path.dirname(os.path.abspath(sys.argv[0]))
def load(n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(here, n + ".py")); m = importlib.util.module_from_spec(spec)
    a = sys.argv; sys.argv = [a[0]]
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
    sys.argv = a; return m
s2 = load("score_drop2"); sd = s2.sd
def simD(x, K, capA=True):
    return s2.simulate(x, capA=capA, K=K if x["clock"] > 60 * sd.TPM else None)
sets = collections.OrderedDict()
for fn in sys.argv[1:]:
    sets.setdefault("60+1" if "gui-60p1" in fn else "4+1 chcm", []).extend(sd.prep(sd.parse(fn)))
for name, allm in sets.items():
    t0 = sum(s2.simulate(x)[0] for x in allm); tA = sum(s2.simulate(x, capA=True)[0] for x in allm)
    print(name, f"now {t0/1000:.0f}s, A {tA/1000:.0f}s")
    for K in (2, 3, 4, 5, 6, 8):
        tt = aff = oth = affA = othA = 0
        for x in allm:
            t, mv = simD(x, K); ta, mva = s2.simulate(x, capA=True)
            tt += t
            if t < ta - 20: affA += 1; othA += mv != mva
        print(f"  A + D(K={K}): {tt/1000:.0f}s, saved against A {(tA-tt)/1000:.0f}s ({(tA-tt)/tA:.1%}), moves {affA}, other move than A {othA}")
    # after A: the remaining drop extra and what it buys
    rem = ch = n = 0
    for x in allm:
        ta, mva = s2.simulate(x, capA=True); tn, mvn = s2.simulate(x, capA=True, exte="off", reds="off")
        if tn < ta - 20: n += 1; rem += ta - tn; ch += mvn != mva
    print(f"  after A: drop extra {rem/1000:.0f}s in {n} moves, move changed in {ch}")
