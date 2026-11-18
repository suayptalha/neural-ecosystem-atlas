"""Experiment 1 baseline: interactions of random component pairs.
usage: exp1_random.py TASK CTX"""
import sys, json, random, time
sys.path.insert(0, "/content")
import numpy as np, torch, eco

task, ctx = sys.argv[1], sys.argv[2]
tok, model = eco.load()
names = json.load(open("/content/results/atlas_components.json"))
screen = json.load(open("/content/results/screen.json"))[f"{task}/{ctx}"]["D"]
allc = [c for c in eco.all_components(model) if c.name not in names]
ranked = sorted(allc, key=lambda c: -abs(screen[c.name]))
ds, acc = eco.build(tok, model, task, ctx, n=900)
r = eco.Runner(model, ds)
rng = random.Random(0)
out = {}
for tag, pool in [("uniform", allc), ("top64", ranked[:64])]:
    rows = []
    cache = {}
    def single(c):
        if c not in cache:
            cache[c] = r.D([c])
        return cache[c]
    for _ in range(120):
        a, b = rng.sample(pool, 2)
        I = (r.D([a, b]) - single(a) - single(b))
        rows.append({"a": a.name, "b": b.name, "Da": float(single(a).mean()), "Db": float(single(b).mean()),
                     "I": float(I.mean()), "I_se": float(I.std() / len(I) ** .5)})
    out[tag] = rows
    print(tag, "mean|I|", np.mean([abs(x["I"]) for x in rows]), flush=True)
json.dump(out, open(f"/content/results/random_pairs_{task}_{ctx}.json", "w"))
