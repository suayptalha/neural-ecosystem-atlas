"""Experiments 1 & 2: pairwise interaction matrices per context.

usage: exp12.py TASK CTX1,CTX2,... OUTNAME
Uses the 16 atlas components in /content/results/atlas_components.json.
For each context: single and pairwise damage per prompt for
  - patch ablation, resample seed 0   (main)
  - patch ablation, resample seed 1   (seed robustness)
  - zero ablation                     (artifact control)
Saves per-prompt arrays so CIs / held-out splits can be computed offline.
"""
import sys, json, itertools, time, os
sys.path.insert(0, "/content")
import numpy as np, torch, eco

task, ctxs, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
NS = {"neutral": 900, "distractor": 900, "indirection": 1600}
tok, model = eco.load()
names = json.load(open("/content/results/atlas_components.json"))
comps = [eco.parse(n) for n in names]
pairs = list(itertools.combinations(range(len(comps)), 2))
res = {"components": names, "pairs": pairs, "contexts": {}}
t0 = time.time()
for ctx in ctxs:
    ds, acc = eco.build(tok, model, task, ctx, n=NS[ctx])
    R = {"N": len(ds), "acc": acc}
    for tag, d, mode in [("patch_s0", ds, "patch"), ("patch_s1", ds.repair(1), "patch"), ("zero", ds, "zero")]:
        r = eco.Runner(model, d)
        single = torch.stack([r.D([c], mode) for c in comps])            # [16, N]
        pair = torch.stack([r.D([comps[i], comps[j]], mode) for i, j in pairs])  # [120, N]
        R[tag] = {"clean": r.clean.cpu().numpy().tolist(),
                  "single": single.cpu().numpy().round(4).tolist(),
                  "pair": pair.cpu().numpy().round(4).tolist()}
        print(task, ctx, tag, len(ds), f"{time.time()-t0:.0f}s", flush=True)
    res["contexts"][ctx] = R
    json.dump(res, open(f"/content/results/{out}.json", "w"))
print("done", time.time() - t0)
