import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8984"; GRID = "#e6e5e1"
KIND = {"dn": "#2a78d6", "attn": "#eb6834", "mlp": "#1baf7a"}
KIND_LABEL = {"dn": "Gated DeltaNet head", "attn": "Gated attention head", "mlp": "MLP block"}
POS = "#d03b3b"; NEG = "#2a78d6"; MID = "#f0efec"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#b7d3f6", MID, "#f6c0bd", "#e34948", "#8f1d1d"])

def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9.5, "savefig.dpi": 200,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })

def kind_of(name):
    return "mlp" if name.startswith("MLP") else ("dn" if name.startswith("DN") else "attn")

def suptitle(fig, title, sub=None, y=0.995):
    fig.text(0.012, y, title, fontsize=16, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.012, y - 0.045, sub, fontsize=10.5, color=INK2, va="top")

def footnote(fig, txt):
    fig.text(0.012, 0.008, txt, fontsize=8.5, color=MUTED, va="bottom")
