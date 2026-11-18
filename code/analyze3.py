"""Analysis + figure for Experiment 3 (cascading extinction)."""
import json, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import *
setup()
TASKS = [t for t in ["binding", "capital"] if os.path.exists(f"../results/exp3_{t}_neutral.json")]
R = {t: json.load(open(f"../results/exp3_{t}_neutral.json")) for t in TASKS}
summary = {}

def L(res, X, sub, pos="all"):
    return res["lattice_val"].get(f"{X}|{'+'.join(sub)}|{pos}")

def role(n, d0, d1):
    if d0 < -0.08:
        return "competitor"
    if d0 > 0.15 and d1 < 0.5 * d0:
        return "dependent"
    if d1 > d0 + 0.08 and d1 > 0.1:
        return "backup"
    return "other"
RC = {"competitor": S5, "dependent": S2, "backup": S1, "other": MUTED}
RL = {"competitor": "competitor (hurts the answer)", "dependent": "dependent (loses its job)", "backup": "backup (becomes more critical)", "other": "other"}

nrow = len(TASKS)
fig = plt.figure(figsize=(16.5, 5.4 * nrow + 1.4))
gs = fig.add_gridspec(nrow, 4, width_ratios=[1.25, 1, 0.95, 0.95], hspace=0.5, wspace=0.36, left=0.05, right=0.985,
                      top=1 - 1.45 / (5.4 * nrow + 1.4), bottom=1.0 / (5.4 * nrow + 1.4))
for row, t in enumerate(TASKS):
    res = R[t]; ch = res["chain"]; A = ch[0]
    d0 = res["disc_steps"][0]["imp"]; d1 = res["disc_steps"][1]["imp"]
    # ---- panel 1: everyone's importance before vs after losing the keystone
    ax = fig.add_subplot(gs[row, 0])
    names = [n for n in d1 if n in d0]
    x = np.array([d0[n] for n in names]); y = np.array([d1[n] for n in names])
    roles = [role(n, d0[n], d1[n]) for n in names]
    lim = (min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.12)
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--"); ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    for rname in ["other", "competitor", "dependent", "backup"]:
        msk = np.array([r_ == rname for r_ in roles])
        ax.scatter(x[msk], y[msk], s=34 if rname != "other" else 16, color=RC[rname], edgecolors=SURF, linewidths=0.8, zorder=3, label=RL[rname])
    lab = sorted(names, key=lambda n: -abs(d1[n] - d0[n]))[:7] + [ch[1]]
    for n in set(lab):
        ax.annotate(n, (d0[n], d1[n]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("importance in the intact model"); ax.set_ylabel(f"importance after removing {A}")
    ax.set_title(f"{'AB'[row]}1   {t.capitalize()}: everyone, before vs after\n      {A} goes extinct", fontsize=10.5)
    ax.legend(fontsize=7.6, loc="upper left")
    counts = {k: int(sum(r_ == k for r_ in roles)) for k in RC}
    # ---- panel 2: staircase along the chain (held-out)
    ax = fig.add_subplot(gs[row, 1])
    conds = [[]] + [ch[:k] for k in range(1, 4)]
    clabs = ["intact"] + [f"−{ch[0]}"] + [f"−…−{ch[k-1]}" for k in range(2, 4)]
    stair = {}
    for q, X in enumerate(ch[1:4], start=1):
        pts = []
        for ci_, sub in enumerate(conds):
            if X in sub:
                break
            v = L(res, X, sub)
            if v is not None:
                pts.append((ci_, *v))
        pts = np.array(pts)
        c = [S1, S3, S4][q - 1]
        ax.fill_between(pts[:, 0], pts[:, 2], pts[:, 3], color=c, alpha=0.18, lw=0)
        ax.plot(pts[:, 0], pts[:, 1], color=c, marker="o", ms=7, mec=SURF, mew=1.5)
        ax.text(pts[-1, 0] + 0.1, pts[-1, 1], X, fontsize=9, fontweight="bold", va="center")
        stair[X] = pts.tolist()
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(clabs, fontsize=8, rotation=25, ha="right")
    ax.set_xlim(-0.2, 3.7)
    ax.set_ylabel("importance (held-out prompts)")
    ax.set_title(f"{'AB'[row]}2   The chain: who steps up next?\n ", fontsize=10.5)
    # ---- panel 3: subset / order control for chain[1]
    ax = fig.add_subplot(gs[row, 2])
    B, C = ch[1], ch[2]
    groups = [(B, [[], [C], [A], [A, C]]), (C, [[], [A], [B], [A, B]])]
    xt, xl, x0 = [], [], 0
    for X, subs in groups:
        start = x0
        for q, sub in enumerate(subs):
            v = L(res, X, sub)
            if v is None:
                continue
            ax.bar(x0, v[0], width=0.8, color=[MUTED, S4, S1, S2][q], edgecolor=SURF, linewidth=2)
            ax.plot([x0, x0], [v[1], v[2]], color=INK, lw=1.2)
            xt.append(x0); xl.append("intact" if not sub else ("−" + sub[0] if len(sub) == 1 else "−both")); x0 += 1
        ax.text((start + x0 - 1) / 2, 1.01, f"of {X}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
        x0 += 0.8
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7.8, rotation=35, ha="right")
    ax.set_title(f"{'AB'[row]}3   Which removal unlocks it?\n      importance…", fontsize=10.5, pad=16)
    # ---- panel 4: timing
    ax = fig.add_subplot(gs[row, 3])
    poss = [("all", "every token"), ("prefix", "context tokens\nonly"), ("last", "final token\nonly")]
    for q, (p, plab) in enumerate(poss):
        for s_i, (X, sub, c, lb) in enumerate([(A, [], POS, f"{A}"), (B, [A], S1, f"{B} (after −{A})")]):
            v = L(res, X, sub, p)
            if v is None:
                continue
            xx = q + (s_i - 0.5) * 0.38
            ax.bar(xx, v[0], width=0.36, color=c, edgecolor=SURF, linewidth=1.5, label=lb if q == 0 else None)
            ax.plot([xx, xx], [v[1], v[2]], color=INK, lw=1.1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in poss], fontsize=8.2)
    ax.set_ylabel("importance when ablated at…")
    ax.set_title(f"{'AB'[row]}4   When does it act?\n ", fontsize=10.5)
    ax.legend(fontsize=7.6, loc="upper right")
    ctrl = res["random_removal_ctrl"]
    summary[t] = {"chain": ch, "val_clean": res["val_clean"], "val_ld_after": res["val_ld_after"], "role_counts_after_keystone": counts,
                  "biggest_changes": [(n, d0[n], d1[n]) for n in sorted(names, key=lambda n: -abs(d1[n] - d0[n]))[:10]],
                  "staircase": stair,
                  "timing": {f"{X}|{'+'.join(sub)}": {p: L(res, X, sub, p) for p, _ in poss} for X, sub in [(A, []), (B, []), (B, [A])]},
                  "subsets": {f"{X}|{'+'.join(sub)}": L(res, X, sub) for X, subs in groups for sub in subs},
                  "ctrl": [{"removed": c["removed"], "D": c["D_removed"], "sum_pos_gain": c["sum_pos_gain"]} for c in ctrl]}
suptitle(fig, "Experiment 3 — Cascading Extinction",
         "Remove the most important component and re-measure everyone. Dependents lose their job, competitors lose their power, and a hidden backup steps up. "
         "Then remove that backup and repeat.", y=0.992)
footnote(fig, "Panel 1: discovery half, top-96 components. Panels 2–4: held-out half, 95% bootstrap CIs. Importance = drop in logit-diff when resample-ablated, given the listed removals.")
fig.savefig("../figures/fig3_cascade.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for q, t in enumerate(TASKS):
    res = R[t]; y = res["val_ld_after"]
    axes[0].plot(range(len(y)), y, marker="o", color=[S1, S2][q], label=f"{t}: " + " → ".join(res["chain"]), mec=SURF, mew=1.5, ms=7)
    ctrl = res["random_removal_ctrl"]
    axes[1].scatter([c["D_removed"] for c in ctrl[1:]], [c["sum_pos_gain"] for c in ctrl[1:]], s=40, color=[S1, S2][q], alpha=0.5, edgecolors="none")
    axes[1].scatter([ctrl[0]["D_removed"]], [ctrl[0]["sum_pos_gain"]], s=150, color=[S1, S2][q], edgecolors=INK, linewidths=1.2, label=f"{t}: removing keystone {ctrl[0]['removed']}")
axes[0].axhline(0, color=INK2, lw=0.8)
axes[0].set_xlabel("number of chain members removed"); axes[0].set_ylabel("held-out logit diff (answer − foil)")
axes[0].set_title("A   Collapse along the extinction chain", fontsize=11); axes[0].legend(fontsize=8)
axes[1].set_xlabel("importance of the removed component"); axes[1].set_ylabel("total importance gained by others")
axes[1].set_title("B   Rewiring happens after almost any removal", fontsize=11); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/figS3_collapse.png"); plt.close(fig)
json.dump(summary, open("../results/summary_exp3.json", "w"), indent=1)
for t in summary:
    s = summary[t]
    print(t, s["chain"], [round(v, 2) for v in s["val_ld_after"]], s["role_counts_after_keystone"])
    print("  timing", json.dumps(s["timing"]))
    print("  subsets", {k: [round(x, 3) for x in v] if v else None for k, v in s["subsets"].items()})
    print("  stair", {k: [[round(a, 3) for a in p] for p in v] for k, v in s["staircase"].items()})
