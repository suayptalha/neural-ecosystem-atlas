"""Experiment 3: cascading extinction (conditional causal importance).
usage: exp3.py TASK CTX
Greedy chain on the discovery half: remove the most important component,
re-measure every component's importance, remove the one whose importance
*rose* the most, repeat. Then: held-out validation, an order/subset lattice,
token-timing variants, and a random-removal control.
"""
import sys, json, random, itertools, time
sys.path.insert(0, "/content")
import numpy as np, torch, eco

task, ctx = sys.argv[1], sys.argv[2]
STEPS = 5
tok, model = eco.load()
comps = eco.all_components(model)
ds, acc = eco.build(tok, model, task, ctx, n=900)
idx = np.arange(len(ds))
disc, val = ds.subset(idx[0::2]), ds.subset(idx[1::2])
t0 = time.time()

def cond_imp(r, removed, cands, positions="all"):
    base = r.ld({c: ("patch", None) for c in removed}, positions)
    out = {}
    for c in cands:
        if c in removed:
            continue
        e = {x: ("patch", None) for x in removed + [c]}
        out[c] = base - r.ld(e, positions)
    return base, out

res = {"task": task, "ctx": ctx, "N_disc": len(disc), "N_val": len(val)}
rd = eco.Runner(model, disc)
chain, imps = [], []
prev = None
pool_c = comps
for s in range(STEPS):
    base, imp = cond_imp(rd, chain, pool_c)
    m = {c.name: float(v.mean()) for c, v in imp.items()}
    imps.append({"removed": [c.name for c in chain], "base_ld": float(base.mean()), "imp": m})
    if s == 0:
        nxt = max(imp, key=lambda c: m[c.name])
        pool_c = sorted(comps, key=lambda c: -abs(m.get(c.name, 1e9)))[:97]
    else:
        # the component that became MORE CRITICAL: largest rise in importance among
        # components that are helpers (positive importance) in the damaged system
        helpers = [c for c in imp if m[c.name] > 0]
        nxt = max(helpers, key=lambda c: m[c.name] - prev[c.name])
    prev = m
    chain.append(nxt)
    print("step", s, "remove", nxt.name, "base", round(float(base.mean()), 3), f"{time.time()-t0:.0f}s", flush=True)
res["chain"] = [c.name for c in chain]
res["disc_steps"] = imps

# held-out validation of the chain: importance of each member before/after each removal
rv = eco.Runner(model, val)
lattice = {}
for X in chain:
    others = [c for c in chain if c != X]
    for k in range(len(others) + 1):
        for sub in itertools.combinations(others, k):
            sub = list(sub)
            if len(sub) > 3:
                continue
            for pos in (["all", "last", "prefix"] if len(sub) <= 2 else ["all"]):
                base, imp = cond_imp(rv, sub, [X], pos)
                v = imp[X].cpu().numpy()
                lattice[f"{X.name}|{'+'.join(c.name for c in sub)}|{pos}"] = list(eco.boot_ci(v))
print("lattice", len(lattice), f"{time.time()-t0:.0f}s", flush=True)
res["lattice_val"] = lattice
res["val_clean"] = float(rv.clean.mean())
res["val_ld_after"] = [float(rv.ld({c: ("patch", None) for c in chain[:k]}).mean()) for k in range(len(chain) + 1)]

# random-removal control: how much does importance rise elsewhere after removing a random component?
screen = {c.name: float(v.mean()) for c, v in zip(comps, [None] * 0)} if False else None
pool = sorted([c for c in comps if c.name in imps[0]["imp"]], key=lambda c: -abs(imps[0]["imp"][c.name]))[:48]
rng = random.Random(0)
ctrl = []
first = chain[0]
for R in [first] + rng.sample([c for c in pool if c != first], 8):
    base, imp = cond_imp(rd, [R], pool)
    gains = sorted(((float(v.mean()) - imps[0]["imp"][c.name], c.name) for c, v in imp.items()), reverse=True)
    ctrl.append({"removed": R.name, "D_removed": imps[0]["imp"][R.name], "top_gains": gains[:5],
                 "sum_pos_gain": sum(g for g, _ in gains if g > 0)})
print("ctrl done", f"{time.time()-t0:.0f}s", flush=True)
res["random_removal_ctrl"] = ctrl
json.dump(res, open(f"/content/results/exp3_{task}_{ctx}.json", "w"))
print("done")
