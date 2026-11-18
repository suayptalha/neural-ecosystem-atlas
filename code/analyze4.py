"""Analysis + figure for Experiment 4 (restoration)."""
import json, os, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from style import *
setup()
TASKS = ["binding", "capital"]
B4 = {t: json.load(open(f"../results/exp4b_{t}_neutral.json")) for t in TASKS if os.path.exists(f"../results/exp4b_{t}_neutral.json")}
C4 = {t: json.load(open(f"../results/exp4c_{t}_neutral.json")) for t in TASKS if os.path.exists(f"../results/exp4c_{t}_neutral.json")}
rng = np.random.default_rng(0)
rows = []
for t, res in B4.items():
    cmap = {r["A"]: r for r in C4.get(t, {"targets": []})["targets"]}
    for T in res["targets"]:
        if T["A"] in cmap:
            for k, v in cmap[T["A"]].items():
                if k.startswith(("ECO_greedy", "ATP_greedy")):
                    T["sel"][k] = v
            T["eco_trace"] = cmap[T["A"]]["eco_trace"]
        rows.append((t, T))
have_greedy = all("ECO_greedy_k3" in T["sel"] for _, T in rows)
METH = {  # key -> (label, color, hatch)
    "ECO_k1": ("Ecological map", S1, None), "INTACT_k1": ("Independent rank", S2, None), "ATP_k1": ("Gradient (AtP)", S3, None), "RANDOM_k1": ("Random", MUTED, None),
    "ECO_greedy_k3": ("Ecological map,\nre-measured each step", S1, None), "ECO_k3": ("Ecological map,\none-shot", S1, "//"),
    "INTACT_k3": ("Independent rank", S2, None), "ATP_k3": ("Gradient,\none-shot", S3, "//"), "ATP_greedy_k3": ("Gradient,\nre-computed each step", S3, None),
    "RANDOM_k3": ("Random", MUTED, None)}
def rec(T, k):
    return T["sel"][k]["recovery"] if k in T["sel"] else np.nan
summary = {"removal": {}, "amplification": {}}
for k in METH:
    if not all(k in T["sel"] for _, T in rows):
        continue
    v = [rec(T, k) for _, T in rows]
    d = {"mean": float(np.mean(v)), "per_target": {f"{t}:{T['A']}": float(rec(T, k)) for t, T in rows}}
    if not k.startswith("RANDOM"):
        d["kl"] = float(np.mean([T["sel"][k]["kl"] for _, T in rows]))
        d["other_drop"] = float(np.mean([T["other_clean"] - T["sel"][k]["other_ld"] for _, T in rows]))
        d["comps"] = {f"{t}:{T['A']}": T["sel"][k]["comps"] for t, T in rows}
    summary["removal"][k] = d
def wins(a, b):
    return int(sum(rec(T, a) > rec(T, b) + 1e-6 for _, T in rows)), int(sum(abs(rec(T, a) - rec(T, b)) <= 1e-6 for _, T in rows))
if have_greedy:
    for b in ["INTACT_k3", "ATP_k3", "ATP_greedy_k3", "ECO_k3"]:
        summary["removal"][f"ECO_greedy_k3_vs_{b}"] = {"wins": wins("ECO_greedy_k3", b)[0], "ties": wins("ECO_greedy_k3", b)[1], "n": len(rows)}
summary["removal"]["targets"] = [{"task": t, "A": T["A"], "gap": T["gap"], "val_clean": T["val_clean"], "val_damaged": T["val_damaged"],
                                  "kl_damaged": T["kl_damaged"], "other_clean": T["other_clean"], "other_damaged": T["other_damaged"],
                                  "eco_trace": T.get("eco_trace")} for t, T in rows]
pv = {"ECO": [], "INTACT": [], "ATP": []}; oracle = []
for t, T in rows:
    D0 = B4[t]["D0"]; pool = list(T["pool_truth"]); truth = [T["pool_truth"][c] for c in pool]
    pv["ECO"].append(spearmanr([-T["Dcond"][c] for c in pool], truth)[0])
    pv["INTACT"].append(spearmanr([-D0[c] for c in pool], truth)[0])
    pv["ATP"].append(spearmanr([T["atp"][c] for c in pool], truth)[0])
    oracle.append(max(truth))
summary["removal"]["spearman_pool"] = {m: float(np.mean(v)) for m, v in pv.items()}
summary["removal"]["spearman_pool_per_target"] = {m: [float(x) for x in v] for m, v in pv.items()}
summary["removal"]["oracle_k1"] = [float(x) for x in oracle]
AM = ["ECO_conditional", "TOP_intact", "GRAD_damaged", "ECO_interaction"]
for key in ["2.0", "5.0"]:
    d = {m: float(np.mean([T["alpha"][m][key]["recovery"] for _, T in rows])) for m in AM}
    d["RANDOM"] = float(np.mean([np.mean(T["random_alpha"][key]) for _, T in rows]))
    d["kl"] = {m: float(np.mean([T["alpha"][m][key]["kl"] for _, T in rows])) for m in AM}
    summary["amplification"][f"alpha{key}"] = d
summary["amplification"]["picks"] = {f"{t}:{T['A']}": T["picks"] for t, T in rows}
summary["amplification"]["naive_pick_is_competitor"] = int(sum(T["pick_stats"]["ECO_interaction"]["D0"] < 0 for _, T in rows))
summary["amplification"]["n"] = len(rows)

# ------------------------------------------------------------------ specificity from existing data
spec = []
for t, T in rows:
    D0 = B4[t]["D0"]; k = T["sel"]["ECO_k1"]["comps"][0]; ki = T["sel"]["INTACT_k1"]["comps"][0]
    spec.append({"scenario": f"{t}:{T['A']}", "pick": k, "gain_intact": -D0[k], "gain_damaged": -T["Dcond"][k],
                 "intact_pick": ki, "intact_pick_gain_damaged": -T["Dcond"][ki], "gap_disc": None})
summary["removal"]["specificity_k1"] = spec
summary["removal"]["eco_k1_equals_oracle"] = int(sum(abs(T["sel"]["ECO_k1"]["recovery"] - o) < 1e-9 for (_, T), o in zip(rows, oracle)))
summary["removal"]["intact_k1_equals_oracle"] = int(sum(abs(T["sel"]["INTACT_k1"]["recovery"] - o) < 1e-9 for (_, T), o in zip(rows, oracle)))
summary["removal"]["atp_k1_equals_oracle"] = int(sum(abs(T["sel"]["ATP_k1"]["recovery"] - o) < 1e-9 for (_, T), o in zip(rows, oracle)))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(16.5, 10.4))
gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 1.05, 1.05], hspace=0.55, wspace=0.32, left=0.055, right=0.985, top=0.86, bottom=0.09)
# A: the AT11.3 story
ax = fig.add_subplot(gs[0, 0])
tA, TA = [(t, T) for t, T in rows if T["A"] == "AT11.3"][0] if any(T["A"] == "AT11.3" for _, T in rows) else rows[0]
e1 = TA["sel"]["ECO_k1"]; i1 = TA["sel"]["INTACT_k1"]
bars = [("intact", TA["val_clean"], INK2), (f"damaged\n−{TA['A']}", TA["val_damaged"], POS),
        (f"remove\n{e1['comps'][0]}\n(eco map)", e1["val_ld"], S1), (f"remove\n{i1['comps'][0]}\n(indep. rank)", i1["val_ld"], S2)]
for q, (l, v, c) in enumerate(bars):
    ax.bar(q, v, color=c, width=0.72, edgecolor=SURF, linewidth=2)
    ax.text(q, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars], fontsize=8.2)
ax.set_ylabel("held-out logit diff (answer − foil)")
ax.set_title(f"A   Losing {TA['A']} wakes up a dormant competitor", fontsize=11)
# B: k=1 recovery
ax = fig.add_subplot(gs[0, 1])
for q, k in enumerate(["ECO_k1", "INTACT_k1", "ATP_k1", "RANDOM_k1"]):
    per = [rec(T, k) for _, T in rows]
    ax.bar(q, np.mean(per), color=METH[k][1], width=0.68, edgecolor=SURF, linewidth=2)
    ax.scatter(q + rng.uniform(-0.15, 0.15, len(per)), per, s=18, color=INK, alpha=0.6, zorder=3, edgecolors="none")
    ax.text(q, max(np.mean(per), 0) + 0.03, f"{np.mean(per)*100:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(["Ecological map\n(damaged system)", "Independent rank\n(intact system)", "Gradient\n(AtP, damaged)", "Random"], fontsize=8.8)
ax.set_ylabel("fraction of lost logit-diff recovered")
R = summary["removal"]
ax.set_title(f"B   Remove ONE competitor: how much comes back?", fontsize=11, pad=30)
ax.text(0.98, 1.0, f"picked the best possible removal in\n{R['eco_k1_equals_oracle']}/6 (map) · {R['intact_k1_equals_oracle']}/6 (independent) · {R['atp_k1_equals_oracle']}/6 (gradient)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color=INK2)
# C: specificity / dormant competitors
ax = fig.add_subplot(gs[0, 2])
gi = np.array([d["gain_intact"] for d in spec]); gd = np.array([d["gain_damaged"] for d in spec])
lim = (-0.1, max(gi.max(), gd.max()) * 1.18)
ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
ax.fill_between(lim, lim, [lim[1]] * 2, color=S1, alpha=0.06)
ax.scatter(gi, gd, s=80, color=S1, edgecolors=SURF, linewidths=1.5, zorder=3)
for d in spec:
    ax.annotate(f"{d['pick']}\n(after −{d['scenario'].split(':')[1]})", (d["gain_intact"], d["gain_damaged"]), xytext=(-6, 6), textcoords="offset points", fontsize=7.8, ha="right" if d["gain_intact"] > 1.0 else "left")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.text(0.04, 0.96, "above the line = helps MORE\nbecause of the damage", transform=ax.transAxes, va="top", fontsize=8.5, color=S1)
ax.set_xlabel("gain from removing it in the intact model"); ax.set_ylabel("gain from removing it after the damage")
ax.set_title("C   Generic competitor or damage-specific?", fontsize=11)
# D: predictive validity
ax = fig.add_subplot(gs[1, 0])
for q, m in enumerate(["ECO", "INTACT", "ATP"]):
    ax.bar(q, np.mean(pv[m]), color=[S1, S2, S3][q], width=0.62, edgecolor=SURF, linewidth=2)
    ax.scatter(q + rng.uniform(-0.12, 0.12, len(pv[m])), pv[m], s=18, color=INK, alpha=0.6, zorder=3, edgecolors="none")
    ax.text(q, 1.02, f"ρ = {np.mean(pv[m]):.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.set_xticks(range(3)); ax.set_xticklabels(["Ecological\nmap", "Independent\nrank", "Gradient\n(AtP)"], fontsize=9)
ax.set_ylabel("rank correlation with the\nreal held-out effect")
ax.set_title("D   Which map predicts the outcome?", fontsize=11)
# E: amplification (x2), clipped
ax = fig.add_subplot(gs[1, 1])
AML = [("ECO_conditional", "Ecological\nmap", S1), ("TOP_intact", "Independent\nrank", S2), ("GRAD_damaged", "Gradient", S3),
       ("ECO_interaction", "Naive: largest\nrise in I", S5), ("RANDOM", "Random", MUTED)]
for q, (m, lab, c) in enumerate(AML):
    per = [np.mean(T["random_alpha"]["2.0"]) for _, T in rows] if m == "RANDOM" else [T["alpha"][m]["2.0"]["recovery"] for _, T in rows]
    ax.bar(q, np.mean(per), color=c, width=0.68, edgecolor=SURF, linewidth=2)
    ax.scatter(q + rng.uniform(-0.15, 0.15, len(per)), np.clip(per, -1.2, 1.2), s=18, color=INK, alpha=0.6, zorder=3, edgecolors="none")
    ax.text(q, (np.mean(per) if np.mean(per) > 0 else 0) + 0.04, f"{np.mean(per)*100:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-1.25, 1.25)
ax.set_xticks(range(5)); ax.set_xticklabels([a[1] for a in AML], fontsize=8.4)
ax.set_ylabel("fraction recovered")
ax.set_title("E   Alternative: boost one helper ×2", fontsize=11)
# F: side effects
ax = fig.add_subplot(gs[1, 2])
labs = [f"{t[:4]} · −{T['A']}" for t, T in rows]
yy = np.arange(len(rows))
ax.barh(yy - 0.18, [T["other_damaged"] - T["other_clean"] for _, T in rows], height=0.34, color=POS, label="damaged")
ax.barh(yy + 0.18, [T["sel"]["ECO_k1"]["other_ld"] - T["other_clean"] for _, T in rows], height=0.34, color=S1, label="damaged + map restoration")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel("change in the OTHER task's logit diff")
ax.set_title("F   Side effect on the other task", fontsize=11)
ax.legend(fontsize=8, loc="lower left")
suptitle(fig, "Experiment 4 — Ecosystem Restoration",
         f"Knock out a key component, then remove ONE competitor chosen on the discovery half. The ecological map recovers {R['ECO_k1']['mean']*100:.0f}% of the lost signal "
         f"vs {R['INTACT_k1']['mean']*100:.0f}% for the independent ranking and {R['RANDOM_k1']['mean']*100:.0f}% for random,\nand it predicts the outcome best (ρ = {R['spearman_pool']['ECO']:.2f}).", y=0.985)
footnote(fig, f"Recovery = (LD_restored − LD_damaged)/(LD_intact − LD_damaged) on held-out prompts. {len(rows)} damage scenarios (binding: 4 keystones, capital: {len(rows)-4}). Panel C uses discovery-half estimates.")
fig.savefig("../figures/fig4_restoration.png"); plt.close(fig)
json.dump(summary, open("../results/summary_exp4.json", "w"), indent=1)
print(json.dumps(spec, indent=0)); print(R["eco_k1_equals_oracle"], R["intact_k1_equals_oracle"], R["atp_k1_equals_oracle"])
