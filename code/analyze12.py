"""Analysis + figures for Experiments 1 & 2."""
import json, itertools, numpy as np, matplotlib.pyplot as plt, networkx as nx
from matplotlib.lines import Line2D
from style import *
setup()
R = {}
for t in ["binding", "capital"]:
    R[t] = json.load(open(f"../results/exp12_{t}.json"))
names = R["binding"]["components"]
pairs = [tuple(p) for p in R["binding"]["pairs"]]
ENVS = [("binding", "neutral"), ("binding", "distractor"), ("binding", "indirection"), ("capital", "neutral"), ("capital", "distractor")]
ELAB = {e: f"{e[0].capitalize()} · {e[1]}" for e in ENVS}
rng = np.random.default_rng(0)
summary = {}

def I_arr(task, ctx, tag="patch_s0", idx=None):
    d = R[task]["contexts"][ctx][tag]
    S = np.array(d["single"]); P = np.array(d["pair"])
    I = np.stack([P[k] - S[i] - S[j] for k, (i, j) in enumerate(pairs)])  # [120,N]
    if idx is not None:
        I = I[:, idx]; S = S[:, idx]
    return I, S, np.array(d["clean"]) if idx is None else np.array(d["clean"])[idx]

def ci(x, B=2000):
    n = x.shape[-1]
    bi = rng.integers(0, n, (B, n))
    m = x[..., bi].mean(-1)
    return x.mean(-1), np.quantile(m, .025, -1), np.quantile(m, .975, -1)

# ------------------------------------------------------------ Experiment 1
I, S, clean = I_arr("binding", "neutral")
N = I.shape[1]
m, lo, hi = ci(I)
ev, od = np.arange(0, N, 2), np.arange(1, N, 2)
md, lod, hid = ci(I[:, ev]); mv, lov, hiv = ci(I[:, od])
sig_disc = (lod > 0) | (hid < 0)
rep = sig_disc & (np.sign(mv) == np.sign(md)) & ((lov > 0) | (hiv < 0))
I1, _, _ = I_arr("binding", "neutral", "patch_s1"); Iz, _, _ = I_arr("binding", "neutral", "zero")
summary["exp1"] = {
    "N": int(N), "n_pairs": len(pairs),
    "n_sig_full": int(((lo > 0) | (hi < 0)).sum()),
    "n_sig_disc": int(sig_disc.sum()), "n_replicated_val": int(rep.sum()),
    "n_pos_rep": int((rep & (md > 0)).sum()), "n_neg_rep": int((rep & (md < 0)).sum()),
    "r_disc_val": float(np.corrcoef(md, mv)[0, 1]),
    "r_seed0_seed1": float(np.corrcoef(m, I1.mean(1))[0, 1]),
    "r_patch_zero": float(np.corrcoef(m, Iz.mean(1))[0, 1]),
    "sign_agree_patch_zero_sig": float((np.sign(Iz.mean(1)[rep]) == np.sign(m[rep])).mean()),
    "mean_abs_I_atlas": float(np.abs(m).mean()),
}
Sm = S.mean(1)
norm_I = np.abs(m) / (np.abs(Sm[[i for i, j in pairs]]) + np.abs(Sm[[j for i, j in pairs]]))
import os
rp = json.load(open("../results/random_pairs_binding_neutral.json")) if os.path.exists("../results/random_pairs_binding_neutral.json") else {"uniform": [{"I": 0, "I_se": 1, "Da": 0, "Db": 0}], "top64": [{"I": 0, "I_se": 1, "Da": 0, "Db": 0}]}
for tag in ["uniform", "top64"]:
    x = rp[tag]
    summary["exp1"][f"mean_abs_I_random_{tag}"] = float(np.mean([abs(r["I"]) for r in x]))
    summary["exp1"][f"frac_sig_random_{tag}"] = float(np.mean([abs(r["I"]) > 1.96 * r["I_se"] for r in x]))
    summary["exp1"][f"norm_I_random_{tag}"] = float(np.median([abs(r["I"]) / (abs(r["Da"]) + abs(r["Db"]) + 1e-9) for r in x]))
summary["exp1"]["frac_sig_atlas"] = float(((lo > 0) | (hi < 0)).mean())
summary["exp1"]["norm_I_atlas_median"] = float(np.median(norm_I))
order = np.argsort(-np.abs(m))
summary["exp1"]["top_pairs"] = [{"pair": f"{names[pairs[k][0]]} × {names[pairs[k][1]]}", "I": float(m[k]), "ci": [float(lo[k]), float(hi[k])],
                                  "Di": float(Sm[pairs[k][0]]), "Dj": float(Sm[pairs[k][1]]), "val": float(mv[k])} for k in order[:12]]

# --- Figure 1
fig = plt.figure(figsize=(16, 9.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 0.8], height_ratios=[1, 1], wspace=0.42, hspace=0.42,
                      left=0.03, right=0.98, top=0.87, bottom=0.08)
ax = fig.add_subplot(gs[:, 0])
G = nx.Graph()
for i, n in enumerate(names):
    G.add_node(n, D=Sm[i])
thr = np.sort(np.abs(m[rep]))[-36] if rep.sum() > 36 else 0
for k, (i, j) in enumerate(pairs):
    if rep[k] and abs(m[k]) >= thr:
        G.add_edge(names[i], names[j], w=m[k])
# layout: layer on y, kind spread on x (a "food web" ordered by depth)
pos = {}
layers = {}
for n in names:
    kd = kind_of(n); l = int(n[3:]) if kd == "mlp" else int(n[2:].split(".")[0])
    layers.setdefault(l, []).append(n)
for l, ns in layers.items():
    for q, n in enumerate(sorted(ns)):
        pos[n] = ((q - (len(ns) - 1) / 2) * 1.25 + 0.18 * np.sin(l), -l)
emax = max(abs(G[u][v]["w"]) for u, v in G.edges) if G.edges else 1
for u, v in sorted(G.edges, key=lambda e: abs(G[e[0]][e[1]]["w"])):
    w = G[u][v]["w"]
    (x1, y1), (x2, y2) = pos[u], pos[v]
    rad = 0.25 if abs(y1 - y2) > 2 else 0.4
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color=POS if w > 0 else NEG, lw=0.6 + 5.5 * abs(w) / emax,
                                alpha=0.35 + 0.55 * abs(w) / emax, connectionstyle=f"arc3,rad={rad}", shrinkA=9, shrinkB=9))
for n in names:
    i = names.index(n); kd = kind_of(n)
    sz = 120 + 900 * abs(Sm[i]) / np.abs(Sm).max()
    ax.scatter(*pos[n], s=sz, c=KIND[kd] if Sm[i] > 0 else SURF, edgecolors=KIND[kd], linewidths=2.2, zorder=3)
    ax.text(pos[n][0], pos[n][1] - 0.62, n, ha="center", va="top", fontsize=8.6, color=INK, zorder=4)
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-24.5, 1.2); ax.axis("off")
for l in [0, 7, 11, 14, 15, 16, 19, 20, 23]:
    ax.text(-2.55, -l, f"L{l}", fontsize=8, color=MUTED, va="center")
ax.set_title("A   The hidden ecosystem (binding · neutral, 36 strongest links)", loc="left", fontsize=11.5)
leg = [Line2D([], [], color=POS, lw=4, label="I > 0  redundancy / synergistic fragility"),
       Line2D([], [], color=NEG, lw=4, label="I < 0  dependency / sub-additive overlap"),
       Line2D([], [], marker="o", ls="", mfc=KIND["dn"], mec=KIND["dn"], ms=9, label="DeltaNet head"),
       Line2D([], [], marker="o", ls="", mfc=KIND["attn"], mec=KIND["attn"], ms=9, label="attention head"),
       Line2D([], [], marker="o", ls="", mfc=KIND["mlp"], mec=KIND["mlp"], ms=9, label="MLP"),
       Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=9, label="hollow = competitor (D < 0)")]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8.6)
# heatmap
ax2 = fig.add_subplot(gs[0, 1])
M = np.zeros((16, 16)); Msig = np.zeros((16, 16), bool)
for k, (i, j) in enumerate(pairs):
    M[i, j] = M[j, i] = m[k]; Msig[i, j] = Msig[j, i] = rep[k]
o = np.argsort([int(n[3:]) if n.startswith("MLP") else int(n[2:].split(".")[0]) + 0.01 * int(n.split(".")[1]) for n in names])
vm = np.abs(M).max()
im = ax2.imshow(M[o][:, o], cmap=DIV, vmin=-vm, vmax=vm)
for a in range(16):
    for b in range(16):
        if Msig[o[a], o[b]]:
            ax2.text(b, a, "•", ha="center", va="center", fontsize=7, color=INK)
ax2.set_xticks(range(16)); ax2.set_xticklabels([names[i] for i in o], rotation=90, fontsize=7.5)
ax2.set_yticks(range(16)); ax2.set_yticklabels([names[i] for i in o], fontsize=7.5)
for sp in ax2.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02); cb.outline.set_visible(False)
ax2.set_title("B   Interaction matrix\n      (• = replicated on held-out half)", loc="left", fontsize=11)
# reliability
ax3 = fig.add_subplot(gs[0, 2])
ax3.axhline(0, color=GRID, lw=1); ax3.axvline(0, color=GRID, lw=1)
ax3.scatter(md, mv, s=22, c=np.where(rep, np.where(md > 0, POS, NEG), MUTED), edgecolors=SURF, linewidths=0.8, zorder=3)
lim = max(np.abs(md).max(), np.abs(mv).max()) * 1.1
ax3.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--")
ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim)
ax3.set_xlabel("I on discovery half"); ax3.set_ylabel("I on held-out half")
ax3.set_title("C   It replicates\n ", loc="left", fontsize=11)
ax3.text(0.04, 0.95, f"r = {summary['exp1']['r_disc_val']:.2f}  (held-out)\nr = {summary['exp1']['r_seed0_seed1']:.2f}  (new resample seed)\nr = {summary['exp1']['r_patch_zero']:.2f}  (zero vs. patch)",
         transform=ax3.transAxes, va="top", fontsize=8.8, color=INK2)
# vs random
ax4 = fig.add_subplot(gs[1, 1:])
groups = [("Atlas pairs\n(16 strongest components)", np.abs(m)), ("Random pairs from\nthe top-64 pool", np.array([abs(r["I"]) for r in rp["top64"]])),
          ("Random pairs from\nall 360 components", np.array([abs(r["I"]) for r in rp["uniform"]]))]
for q, (lab, x) in enumerate(groups):
    jit = rng.uniform(-0.28, 0.28, len(x))
    ax4.scatter(x, q + jit, s=14, color=[S1, S4, MUTED][q], alpha=0.75, edgecolors="none")
    ax4.plot([np.median(x)] * 2, [q - 0.38, q + 0.38], color=INK, lw=2)
    ax4.text(np.median(x), q - 0.42, f"median {np.median(x):.3f}", ha="center", va="bottom", fontsize=8.8, color=INK2)
ax4.set_xscale("symlog", linthresh=0.01)
ax4.set_yticks(range(3)); ax4.set_yticklabels([g[0] for g in groups], fontsize=9)
ax4.invert_yaxis(); ax4.set_xlabel("|interaction I|  (logits, symlog)")
ax4.set_title("D   Strong interactions concentrate among the important components", loc="left", fontsize=11)
ax4.spines["left"].set_visible(False)
s1 = summary["exp1"]
suptitle(fig, "Experiment 1 — The Hidden Ecosystem",
         f"Pairwise resample-ablation of 16 components on {N} binding prompts.  {s1['n_replicated_val']}/120 pairs show an interaction that is significant on the "
         f"discovery half and replicates on the held-out half ({s1['n_pos_rep']} positive, {s1['n_neg_rep']} negative).", y=0.985)
footnote(fig, "I_ij = D_ij − D_i − D_j, D = drop in logit(answer) − logit(foil). 95% bootstrap CIs over prompts. Node size = |single-component damage D_i|.")
fig.savefig("../figures/fig1_ecosystem.png"); plt.close(fig)

# ------------------------------------------------------------ Experiment 2
E = {}
for t, c in ENVS:
    Ie, Se, ce = I_arr(t, c)
    Iz_, _, _ = I_arr(t, c, "zero")
    n = Ie.shape[1]; ev, od = np.arange(0, n, 2), np.arange(1, n, 2)
    E[(t, c)] = dict(I=Ie, S=Se, clean=ce, full=ci(Ie), disc=ci(Ie[:, ev]), val=ci(Ie[:, od]), zero=Iz_.mean(1))

def sgn(t):  # +1 / -1 / 0 by CI
    mm, l, h = t
    return np.where(l > 0, 1, np.where(h < 0, -1, 0))

flips = []
for k in range(len(pairs)):
    for a, b in itertools.permutations(ENVS, 2):
        if sgn(E[a]["disc"])[k] == 1 and sgn(E[b]["disc"])[k] == -1:
            ok = sgn(E[a]["val"])[k] == 1 and sgn(E[b]["val"])[k] == -1
            zero_ok = E[a]["zero"][k] > 0 and E[b]["zero"][k] < 0
            flips.append(dict(k=k, pair=f"{names[pairs[k][0]]} × {names[pairs[k][1]]}", pos_env=ELAB[a], neg_env=ELAB[b],
                              I_pos=float(E[a]["full"][0][k]), I_neg=float(E[b]["full"][0][k]), replicated=bool(ok), zero_agrees=bool(zero_ok),
                              within_task=a[0] == b[0]))

# difficulty control: match clean-LD distributions across environments of the same task by quantile-matched subsampling
def matched_idx(cl_list, nb=8, seed=0):
    r = np.random.default_rng(seed)
    allc = np.concatenate(cl_list)
    edges = np.quantile(allc, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-6
    out = [[] for _ in cl_list]
    for b in range(nb):
        ids = [np.where((c >= edges[b]) & (c < edges[b + 1]))[0] for c in cl_list]
        mn = min(len(x) for x in ids)
        for q, x in enumerate(ids):
            out[q] += list(r.choice(x, mn, replace=False)) if mn > 0 else []
    return [np.array(sorted(x)) for x in out]

ctrl = {}
for task in ["binding", "capital"]:
    envs = [e for e in ENVS if e[0] == task]
    mi = matched_idx([E[e]["clean"] for e in envs])
    for e, ix in zip(envs, mi):
        ctrl[e] = dict(full=ci(E[e]["I"][:, ix]), n=len(ix), clean=float(E[e]["clean"][ix].mean()))
for f in flips:
    k = f["k"]
    a = [e for e in ENVS if ELAB[e] == f["pos_env"]][0]; b = [e for e in ENVS if ELAB[e] == f["neg_env"]][0]
    if a[0] == b[0]:
        f["difficulty_matched"] = bool(sgn(ctrl[a]["full"])[k] == 1 and sgn(ctrl[b]["full"])[k] == -1)
    else:
        f["difficulty_matched"] = None
uniq = {}
for f in flips:
    uniq.setdefault(f["pair"], []).append(f)
robust = [p for p, fs in uniq.items() if any(x["replicated"] and x["zero_agrees"] for x in fs)]
robust_within = [p for p, fs in uniq.items() if any(x["replicated"] and x["zero_agrees"] and x["within_task"] and x["difficulty_matched"] for x in fs)]
summary["exp2_extra"] = {}
summary["exp2"] = {"n_flip_events_disc": len(flips), "n_pairs_with_flip_disc": len(uniq),
                   "n_pairs_flip_replicated_and_zero": len(robust), "robust_pairs": robust,
                   "n_pairs_flip_within_task_robust_difficulty_matched": len(robust_within), "robust_within": robust_within,
                   "flips": flips,
                   "env_clean_ld": {ELAB[e]: float(E[e]["clean"].mean()) for e in ENVS},
                   "env_clean_ld_matched": {ELAB[e]: ctrl[e]["clean"] for e in ENVS},
                   "env_N": {ELAB[e]: int(E[e]["I"].shape[1]) for e in ENVS},
                   "env_n_sig": {ELAB[e]: int((sgn(E[e]["full"]) != 0).sum()) for e in ENVS}}
# rank pairs to show: robust within-task first, then robust others by contrast size
def contrast(p):
    return max(x["I_pos"] - x["I_neg"] for x in uniq[p])
show = sorted(robust, key=lambda p: -contrast(p))[:1]
show += [p for p in sorted(robust_within, key=lambda p: -contrast(p)) if p not in show][:1]
show += [p for p in sorted(robust, key=lambda p: -contrast(p)) if p not in show][: max(0, 6 - len(show))]
summary["exp2"]["shown"] = show

# --- Figure 2
fig = plt.figure(figsize=(16, 9.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.62, wspace=0.25, left=0.09, right=0.98, top=0.86, bottom=0.17)
cols = [S1, S2, S3, S4, S5]
for q, p in enumerate(show[:6]):
    ax = fig.add_subplot(gs[q // 3, q % 3]) if q < 3 else None
    if ax is None:
        break
    k = [x["k"] for x in uniq[p]][0]
    ax.axhline(0, color=INK2, lw=1)
    for e_i, e in enumerate(ENVS):
        mm, l, h = (x[k] for x in E[e]["full"])
        mc, lc, hc = (x[k] for x in ctrl[e]["full"])
        ax.plot([e_i - 0.12] * 2, [l, h], color=cols[e_i], lw=2.5)
        ax.scatter([e_i - 0.12], [mm], s=70, color=cols[e_i], edgecolors=SURF, linewidths=2, zorder=3)
        ax.plot([e_i + 0.12] * 2, [lc, hc], color=cols[e_i], lw=1.4, alpha=0.6)
        ax.scatter([e_i + 0.12], [mc], s=40, facecolors=SURF, edgecolors=cols[e_i], linewidths=1.6, zorder=3)
    ax.set_xticks(range(5)); ax.set_xticklabels([ELAB[e].replace(" · ", "\n") for e in ENVS], fontsize=8.5)
    ax.set_ylabel("interaction I (logits)")
    yl = ax.get_ylim(); yr = max(abs(yl[0]), abs(yl[1]))
    ax.axhspan(0, yr * 1.2, color=POS, alpha=0.04); ax.axhspan(-yr * 1.2, 0, color=NEG, alpha=0.04)
    ax.set_ylim(-yr * 1.1, yr * 1.1)
    ax.set_title(p, fontsize=11.5)
    ax.text(0.99, 0.97, "redundant", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=POS)
    ax.text(0.99, 0.03, "dependent", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=NEG)
# bottom: sign map of all pairs x environments
ax = fig.add_subplot(gs[1, :])
sg = np.stack([sgn(E[e]["full"]) for e in ENVS])  # [5,120]
flipping = np.where((sg == 1).any(0) & (sg == -1).any(0))[0]
others = np.where(~((sg == 1).any(0) & (sg == -1).any(0)) & (sg != 0).any(0))[0]
ordk = list(flipping[np.argsort(-(np.stack([E[e]["full"][0] for e in ENVS])[:, flipping].max(0) - np.stack([E[e]["full"][0] for e in ENVS])[:, flipping].min(0)))])
vals = np.stack([E[e]["full"][0] for e in ENVS])[:, ordk]
vm = np.quantile(np.abs(vals), 0.95)
im = ax.imshow(np.where(sg[:, ordk] == 0, np.nan, vals), cmap=DIV, vmin=-vm, vmax=vm, aspect="auto", interpolation="nearest")
ax.set_facecolor(SURF)
for a in range(5):
    for b in range(len(ordk)):
        if sg[a, ordk[b]] != 0:
            ax.text(b, a, "+" if sg[a, ordk[b]] > 0 else "−", ha="center", va="center", fontsize=8, color=INK if abs(vals[a, b]) < 0.6 * vm else SURF)
ax.set_yticks(range(5)); ax.set_yticklabels([ELAB[e] for e in ENVS], fontsize=9)
ax.set_xticks(range(len(ordk))); ax.set_xticklabels([f"{names[pairs[k][0]]}×{names[pairs[k][1]]}" for k in ordk], rotation=60, ha="right", fontsize=7.6)
cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01); cb.outline.set_visible(False); cb.set_label("I (logits)", color=INK2)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title(f"All {len(flipping)} pairs whose significant interaction sign differs between environments (full sample; blank = not significant)", loc="left", fontsize=11)
s2 = summary["exp2"]
suptitle(fig, "Experiment 2 — Ecological Role Reversal",
         f"Same weights, same two components, different prompt environment.  {s2['n_pairs_flip_replicated_and_zero']} pairs flip sign with replication on held-out prompts "
         f"and agreement under zero-ablation; {s2['n_pairs_flip_within_task_robust_difficulty_matched']} do so within one task after difficulty matching.", y=0.985)
footnote(fig, "Filled = all prompts; hollow = subsample with clean logit-diff distribution matched across environments of the same task (difficulty control). Bars = 95% bootstrap CI.")
fig.savefig("../figures/fig2_role_reversal.png"); plt.close(fig)
json.dump(summary, open("../results/summary_exp12.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("flips", "top_pairs")} for k, v in summary.items()}, indent=1))
print(json.dumps(summary["exp1"]["top_pairs"][:8], indent=1))
