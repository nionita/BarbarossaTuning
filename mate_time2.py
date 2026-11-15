"""More detail from mate_time.py (same folder, same argument): clock when the score first falls under -150 / -500 /
mate per game, mate flips, research at a mate bound, the won queen endings, the lost phase, time over target by
class. Usage: mate_time2.py <log>
"""
import sys, runpy, io, contextlib, collections
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(sys.argv[0].replace("mate_time2", "mate_time"), run_name="x")
allm, games, MATE = ns["allm"], ns["games"], ns["MATE"]

print("clock (ms) and move when the final score first <= -150 / <= -500 / mate, by game; start clock", )
for g in games:
    v = [x for x in g if "fin" in x]
    def first(th):
        for x in v:
            if x["fin"] <= th: return f"m{x['m']} {x['clock']/1000:.1f}s"
        return "-"
    print(f"  g{v[0]['g']}: start {v[0]['clock']/1000:.0f}s  -150: {first(-150)}  -500: {first(-500)}  mate: {first(-MATE)}  moves {len(v)} last {v[-1]['fin']}")

print("\nmoves with mate score seen and gone at next draft:")
for x in allm:
    if x["mflip"]:
        print(f"  g{x['g']} m{x['m']} np {x['np']} q {x['nq']} used {x['used']} lim {x['lim']} scores {[(e['d'], e['sc']) for e in x['dr'] if e['d'] >= 5]}")

print("\nmoves with research at a mate bound:")
n = 0
for x in allm:
    if x["resm"]:
        n += 1
        print(f"  g{x['g']} m{x['m']} np {x['np']} resm {x['resm']} res {x['restot']} used {x['used']} lim {x['lim']} fin {x['fin']} gls {x['gls']} atmax {x['atmax']} d {x['dr'][-1]['d']}")
print(n, "moves; time", sum(x["used"] for x in allm if x["resm"]) / 1000)

print("\nmate for us (final), by game: moves, draft hist, at max, mean used, sd>2d")
for g in games:
    v = [x for x in g if "fin" in x and x["fin"] >= MATE]
    if not v: continue
    h = collections.Counter(x["dr"][-1]["d"] for x in v)
    print(f"  g{v[0]['g']}: {len(v)} moves m{v[0]['m']}-m{v[-1]['m']}, at max {sum(x['atmax'] for x in v)}, mean used {sum(x['used'] for x in v)/len(v):.0f} ms, mean lim {sum(x['lim'] for x in v)/len(v):.0f}, sd>2d {sum(x['sdr']>2 for x in v)}, draft25 {h[25]}, drafts {sorted(h.items())}")
    pc = collections.Counter((x["np"], x["nq"]) for x in v)
    print("     pieces/queens:", sorted(pc.items()))
    for x in v:
        if x["atmax"] or x["m"] % 12 == 0:
            d = x["dr"]
            print(f"     m{x['m']} np {x['np']} q {x['nq']} d {d[-1]['d']} used {x['used']} lim {x['lim']} lastit {x['lastit']} sd {max((e['sd'] or 0) for e in d)} lv {d[-1]['lv']} nodes(last compl) {d[-2]['nodes'] if len(d)>1 else None} flags {x['flags']} fen {x['fen']}" )

print("\nmated (final <= -MATE): draft hist, at max")
v = [x for x in allm if x["fin"] <= -MATE]
h = collections.Counter(x["dr"][-1]["d"] for x in v)
print(" ", sorted(h.items()), "at max", sum(x["atmax"] for x in v), "ended by draftmax", sum(x["flags"][0] for x in v))

print("\nlost phase (-MATE < fin <= -500): exte and iteration")
v = [x for x in allm if -MATE < x["fin"] <= -500]
print(f"  {len(v)} moves, used {sum(x['used'] for x in v)/1000:.1f}s, mean lim {sum(x['lim'] for x in v)/len(v):.0f} ms, mean clock {sum(x['clock'] for x in v)/len(v):.0f} ms, exte any {sum(x['nexte']>0 for x in v)}, jumps {sum(x['jumps']>0 for x in v)}, sd>2d {sum(x['sdr']>2 for x in v)}, research mate {sum(x['resm']>0 for x in v)}")
# score jump towards mate within the search: non-mate -> mate against
jm = [x for x in allm if any(i > 0 and x['dr'][i]['sc'] <= -MATE and x['dr'][i-1]['sc'] > -MATE for i in range(len(x['dr'])))]
print(f"\nmoves where the score falls to mated within the search: {len(jm)}; at max {sum(x['atmax'] for x in jm)}; used {sum(x['used'] for x in jm)/1000:.1f}s; ended draftmax {sum(x['flags'][0] for x in jm)}")
for x in jm:
    d = x["dr"]; i = next(i for i in range(1, len(d)) if d[i]['sc'] <= -MATE and d[i-1]['sc'] > -MATE)
    print(f"  g{x['g']} m{x['m']} at draft {d[i]['d']} from {d[i-1]['sc']} used then {d[i]['used']} total {x['used']} lim {x['lim']} clock {x['clock']} final d {d[-1]['d']} flags {x['flags']}")
# time share
tot = sum(x["used"] for x in allm)
print(f"\ntotal time {tot/1000:.0f}s; in moves with clock < 10% start: {sum(x['used'] for x in allm if x['clock'] < 24600)/1000:.0f}s; fin<=-500 or mate: {sum(x['used'] for x in allm if x['fin'] <= -500 or abs(x['fin'])>=MATE)/1000:.0f}s")
print(f"time over target (used - ms, positive only): all {sum(max(0,x['used']-x['tgt']) for x in allm)/1000:.0f}s; fin in (-150,150): {sum(max(0,x['used']-x['tgt']) for x in allm if abs(x['fin'])<150)/1000:.0f}s; fin <= -500 or mate: {sum(max(0,x['used']-x['tgt']) for x in allm if x['fin'] <= -500 or abs(x['fin'])>=MATE)/1000:.0f}s; drop 24..300: {sum(max(0,x['used']-x['tgt']) for x in allm if 24 <= x['drop'] < 300)/1000:.0f}s; first 10 moves: {sum(max(0,x['used']-x['tgt']) for x in allm if x['m']<=10)/1000:.0f}s")
print(f"sd>2d moves by final class: mate for {sum(x['sdr']>2 for x in allm if x['fin']>=MATE)}, mated {sum(x['sdr']>2 for x in allm if x['fin']<=-MATE)}, other {sum(x['sdr']>2 for x in allm if abs(x['fin'])<MATE)}; max sd/d {max(x['sdr'] for x in allm):.2f}")
