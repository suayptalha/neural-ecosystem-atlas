"""Figure 0: the causal-importance landscape of all 384 components in 5 environments."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from style import *
setup()
S = json.load(open("../results/screen.json"))
atlas = json.load(open("../results/atlas_components.json"))
envs = [("binding/neutral", "Binding · neutral"), ("binding/distractor", "Binding · distractor"),
        ("binding/indirection", "Binding · indirection"), ("capital/neutral", "Capital · neutral"),
        ("capital/distractor", "Capital · distractor")]
ATT = [3, 7, 11, 15, 19, 23]

def grid(D):
    G = np.full((24, 17), np.nan)
    for l in range(24):
        nh = 8 if l in ATT else 16
        for h in range(nh):
            G[l, h] = D[f"AT{l}.{h}" if l in ATT else f"DN{l}.{h}"]
        G[l, 16] = D[f"MLP{l}"]
    return G

fig, axes = plt.subplots(1, 5, figsize=(16, 7.2), sharey=True)
norm = TwoSlopeNorm(vcenter=0, vmin=-1.6, vmax=3.4)
for ax, (k, lab) in zip(axes, envs):
    G = grid(S[k]["D"])
    im = ax.imshow(G, cmap=DIV, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(lab, fontsize=11)
    ax.set_xticks([0, 7, 14, 16]); ax.set_xticklabels(["h0", "h7", "h14", "MLP"])
    ax.axvline(15.5, color=SURF, lw=3)
    for l in ATT:
        ax.add_patch(plt.Rectangle((-0.5, l - 0.5), 8, 1, fill=False, ec=KIND["attn"], lw=1.4))
    for n in atlas:
        kd = kind_of(n)
        if kd == "mlp":
            l, h = int(n[3:]), 16
        else:
            l, h = map(int, n[2:].split("."))
        ax.plot(h, l, marker="o", ms=3.2, mfc="none", mec=INK, mew=0.9)
    ax.text(0.98, -0.065, f"N={S[k]['N']} prompts", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
axes[0].set_yticks(range(0, 24, 1)); axes[0].set_yticklabels([f"L{l}" + (" · attn" if l in ATT else "") for l in range(24)], fontsize=8)
axes[0].set_ylabel("layer")
cax = fig.add_axes([0.935, 0.2, 0.011, 0.6]); cb = fig.colorbar(im, cax=cax)
cb.set_label("damage D (logit-diff drop when resample-ablated)", color=INK2)
cb.outline.set_visible(False)
suptitle(fig, "The causal landscape of Qwen3.5-0.8B",
         "Every DeltaNet head, gated-attention head (orange boxes) and MLP block, ablated one at a time. Red = helps the right answer, blue = competes with it. "
         "Circles = the 16 atlas components.", y=0.99)
fig.subplots_adjust(left=0.07, right=0.915, top=0.86, bottom=0.07, wspace=0.08)
fig.savefig("../figures/fig0_landscape.png")
print("ok")
