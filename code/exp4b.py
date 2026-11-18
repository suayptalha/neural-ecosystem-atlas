"""Experiment 4b: restoration by *removing a competitor* ("predator control").
usage: exp4b.py TASK CTX N_DAMAGE
After damaging A, ablate K in {1,3} further components chosen by:
  ECO     lowest conditional importance D_k|A measured with the ablation map (damaged system)
  INTACT  lowest intact importance D_k (independent ranking, ablation-measured)
  ATP     largest attribution-patching gain estimate in the damaged system (gradient baseline)
  RANDOM  random components
Selection on the discovery half; evaluation on the held-out half.
"""
import sys, json, random, time
sys.path.insert(0, "/content")
import numpy as np, torch, torch.nn.functional as F, eco
exec(open("/content/gen_text.py").read())   # defines TEXT

task, ctx, ND = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok, model = eco.load()
comps = eco.all_components(model)
ds, acc = eco.build(tok, model, task, ctx, n=900)
idx = np.arange(len(ds))
disc, val = ds.subset(idx[0::2]), ds.subset(idx[1::2])
other_task = "capital" if task == "binding" else "binding"
ods, _ = eco.build(tok, model, other_task, "neutral", n=600)
t0 = time.time()
rd, rv, ro = eco.Runner(model, disc), eco.Runner(model, val), eco.Runner(model, ods)
iv = rd.iv

W = 24
tt = tok.encode(TEXT.replace("\n", " "))
tt = tt[: (len(tt) // W) * W]
gen_ids = torch.tensor(tt, device=eco.DEV).view(-1, W)
gen_cids = gen_ids.roll(1, 0)
def gen_logprobs(edits):
    iv.edits = {}; iv.grad_scales = None; iv.cache = {}; iv.recording = True
    with torch.no_grad():
        model(gen_cids)
    iv.recording = False; iv.edits = dict(edits); iv.positions = "all"
    with torch.no_grad():
        lp = F.log_softmax(model(gen_ids).logits.float(), -1)
    iv.edits = {}
    return lp
lp_intact = gen_logprobs({})
def kl(edits):
    lp = gen_logprobs(edits)
    return float((lp_intact.exp() * (lp_intact - lp)).sum(-1).mean())

P = ("patch", None)
D0 = {c: float(rd.D([c]).mean()) for c in comps}
print("D0 done", f"{time.time()-t0:.0f}s", flush=True)
targets = sorted(comps, key=lambda c: -D0[c])[:ND]
rng = random.Random(0)
out = {"task": task, "ctx": ctx, "N_disc": len(disc), "N_val": len(val), "D0": {c.name: v for c, v in D0.items()}, "targets": []}

for A in targets:
    tA = time.time()
    cands = [c for c in comps if c != A]
    bd = rd.ld({A: P})
    Dc = {c: float((bd - rd.ld({A: P, c: P})).mean()) for c in cands}
    atp = rd.grad_attr({A: P}, cands, mode="atp")          # d LD/dt ~ LD(ablate k) - LD
    vd = rv.ld({A: P}); gap = float((rv.clean - vd).mean()); vdm = float(vd.mean())
    od = float(ro.ld({A: P}).mean())
    T = {"A": A.name, "val_clean": float(rv.clean.mean()), "val_damaged": vdm, "gap": gap,
         "other_clean": float(ro.clean.mean()), "other_damaged": od, "kl_damaged": kl({A: P}),
         "Dcond": {c.name: v for c, v in Dc.items()}, "atp": {c.name: v for c, v in atp.items()}, "sel": {}}
    rank = {"ECO": sorted(cands, key=lambda c: Dc[c]),
            "INTACT": sorted(cands, key=lambda c: D0[c]),
            "ATP": sorted(cands, key=lambda c: -atp[c])}
    def ev(ks):
        e = {A: P}; e.update({k: P for k in ks})
        v = float(rv.ld(e).mean())
        return {"comps": [k.name for k in ks], "val_ld": v, "recovery": (v - vdm) / gap,
                "other_ld": float(ro.ld(e).mean()), "kl": kl(e)}
    for K in (1, 3):
        for m, r in rank.items():
            T["sel"][f"{m}_k{K}"] = ev(r[:K])
        rs = []
        for _ in range(24):
            ks = rng.sample(cands, K)
            e = {A: P}; e.update({k: P for k in ks})
            rs.append((float(rv.ld(e).mean()) - vdm) / gap)
        T["sel"][f"RANDOM_k{K}"] = {"recovery_all": rs, "recovery": float(np.mean(rs))}
    # ---- amplification arm (scale one helper by alpha)
    gs_ = rd.grad_attr({A: P}, cands, mode="scale")
    Iak = {c: Dc[c] - D0[c] for c in cands}
    apicks = {"ECO_conditional": max(cands, key=lambda c: Dc[c]), "TOP_intact": max(cands, key=lambda c: D0[c]),
              "GRAD_damaged": max(cands, key=lambda c: gs_[c]), "ECO_interaction": max(cands, key=lambda c: Iak[c])}
    T["picks"] = {m: c.name for m, c in apicks.items()}
    T["pick_stats"] = {m: {"D0": D0[c], "Dcond": Dc[c], "I": Iak[c], "grad": gs_[c]} for m, c in apicks.items()}
    T["alpha"] = {}
    for m, k in apicks.items():
        T["alpha"][m] = {}
        for a in (2.0, 5.0):
            e = {A: P, k: ("scale", a)}
            v = float(rv.ld(e).mean())
            T["alpha"][m][str(a)] = {"recovery": (v - vdm) / gap, "other_ld": float(ro.ld(e).mean()), "kl": kl(e)}
    rand_a = rng.sample(cands, 12)
    T["random_alpha"] = {str(a): [(float(rv.ld({A: P, k: ("scale", a)}).mean()) - vdm) / gap for k in rand_a] for a in (2.0, 5.0)}
    # held-out truth for a broad pool -> oracle + how well each score predicts
    pool = set()
    for r in rank.values():
        pool |= set(r[:20])
    truth = {}
    for k in pool:
        truth[k.name] = (float(rv.ld({A: P, k: P}).mean()) - vdm) / gap
    T["pool_truth"] = truth
    out["targets"].append(T)
    json.dump(out, open(f"/content/results/exp4b_{task}_{ctx}.json", "w"))
    print("target", A.name, "gap", round(gap, 2), {k: (v["comps"] if "comps" in v else None, round(v["recovery"], 3)) for k, v in T["sel"].items()},
          f"{time.time()-tA:.0f}s", flush=True)
print("done", time.time() - t0)
