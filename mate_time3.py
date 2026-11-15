"""More detail from mate_time.py (same folder, same argument): exact mate (20000) against the eval clamp range,
time spent after a mate score was first seen, moves 1 to 10 and moves at the maximum with |final| < 500.
Usage: mate_time3.py <log>
"""
import sys, runpy, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(sys.argv[0].replace("mate_time3", "mate_time"), run_name="x")
allm = ns["allm"]
def rep(name, v):
    n = len(v)
    if not n: print(name, "none"); return
    print(f"{name}: {n} moves, at max {sum(x['atmax'] for x in v)}, used {sum(x['used'] for x in v)/1000:.1f}s, target {sum(x['tgt'] for x in v)/1000:.1f}s, "
          f"mean used {sum(x['used'] for x in v)/n:.0f} ms, mean clock {sum(x['clock'] for x in v)/n:.0f} ms, draft25 {sum(x['dr'][-1]['d']==25 for x in v)}, mean draft {sum(x['dr'][-1]['d'] for x in v)/n:.1f}, onlyone {sum(x['flags'][3] for x in v)}")
rep("final -20000", [x for x in allm if x["fin"] == -20000])
rep("final -19999..-17000 (eval clamp range)", [x for x in allm if -20000 < x["fin"] <= -17000])
rep("final -16999..-500", [x for x in allm if -17000 < x["fin"] <= -500])
rep("final +20000", [x for x in allm if x["fin"] == 20000])
rep("final 17000..19999", [x for x in allm if 17000 <= x["fin"] < 20000])
# true mate flips within a search (exact 20000)
fl = 0; mv = 0
for x in allm:
    d = x["dr"]; c = sum(1 for i in range(1, len(d)) if abs(d[i-1]["sc"]) == 20000 and abs(d[i]["sc"]) != 20000)
    fl += c; mv += c > 0
print("true mate (20000) seen and gone at the next draft:", fl, "in", mv, "moves")
# drafts after a -20000 score was first seen in a search: time spent
t = 0; n = 0
for x in allm:
    d = x["dr"]
    i = next((i for i in range(len(d)) if d[i]["sc"] == -20000), None)
    if i is not None and x["fin"] == -20000:
        n += 1; t += x["used"] - d[i]["used"]
print(f"time spent after the first -20000 draft in searches ending -20000: {t/1000:.1f}s in {n} moves")
t = 0; n = 0
for x in allm:
    d = x["dr"]
    i = next((i for i in range(len(d)) if d[i]["sc"] == 20000), None)
    if i is not None and x["fin"] == 20000:
        n += 1; t += x["used"] - d[i]["used"]
print(f"time spent after the first +20000 draft in searches ending +20000: {t/1000:.1f}s in {n} moves")
# early moves at max
e = [x for x in allm if x["m"] <= 10]
print(f"moves 1-10: {len(e)}, at max {sum(x['atmax'] for x in e)}, used {sum(x['used'] for x in e)/1000:.0f}s, exte before last it {sum(x['extelast'] for x in e)}, any exte {sum(x['nexte']>0 for x in e)}, np>=20 {sum(x['np']>=20 for x in e)}")
am = [x for x in allm if x["atmax"] and abs(x["fin"]) < 500]
print(f"at max with |final| < 500: {len(am)}, used {sum(x['used'] for x in am)/1000:.0f}s; exte before last it {sum(x['extelast'] for x in am)}; queens<=8 pieces {sum(x['nq']>0 and x['np']<=8 for x in am)}; sd>2d {sum(x['sdr']>2 for x in am)}")
