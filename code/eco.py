"""
Neural Ecosystem Atlas — shared library.

Components = individual heads of every token-mixer (Gated DeltaNet heads and
gated-attention heads) plus whole MLP blocks of Qwen3.5-0.8B.

A head's "contribution" is its slice of the input of out_proj / o_proj (after
the DeltaNet gated RMSNorm, resp. after the attention output gate). Intervening
there edits what the head *writes* to the residual stream and never touches the
DeltaNet recurrent state, so no out-of-distribution state is produced.

Interventions
  patch : replace slice with the activation from a matched corrupted prompt
          (same template, same token length, resampled bindings)  [default]
  zero  : set slice to 0                                          [control]
  scale : multiply slice by a factor                               [restoration]
"""
import random, math, json, itertools, time, zlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "Qwen/Qwen3.5-0.8B"
DEV = "cuda"

# --------------------------------------------------------------------- model
def load(mid=MID):
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


# ------------------------------------------------------------------ datasets
LETTERS = list("ABCDFGHJKMNPRSTW")   # E reserved for indirection, single tokens
COLORS = ["red", "blue", "green", "black", "white", "yellow", "pink", "brown",
          "gray", "orange", "purple", "gold", "silver"]
CITIES = [("Paris", "France"), ("Berlin", "Germany"), ("Rome", "Italy"),
          ("Madrid", "Spain"), ("Tokyo", "Japan"), ("Moscow", "Russia"),
          ("Cairo", "Egypt"), ("Athens", "Greece"), ("Lima", "Peru"),
          ("Ottawa", "Canada"), ("Vienna", "Austria"), ("Dublin", "Ireland"),
          ("Oslo", "Norway"), ("Beijing", "China"), ("Lisbon", "Portugal"),
          ("London", "England"), ("Warsaw", "Poland"), ("Prague", "Czechia")]
NEUTRAL = ["it is late", "it is fine", "it is cold", "it is calm"]
ADJ = ["lovely", "famous", "large", "busy"]


def _binding(rng, ctx):
    """Symbolic variable binding. Returns (prompt, answer, foil)."""
    L = rng.sample(LETTERS, 4)
    Cc = rng.sample(COLORS, 4)
    q = rng.randrange(4)
    others = [i for i in range(4) if i != q]
    f = rng.choice(others)
    pairs = ", ".join(f"{l} = {c}" for l, c in zip(L, Cc))
    if ctx == "neutral":
        note, query = rng.choice(NEUTRAL), L[q]
    elif ctx == "distractor":           # foil is re-mentioned right before the query
        note, query = f"{L[f]} is {Cc[f]}", L[q]
    elif ctx == "indirection":          # one extra hop: E -> L[q] -> colour
        note, query = f"E is {L[q]}", "E"
    else:
        raise ValueError(ctx)
    prompt = f"{pairs}. Note that {note}. The color of {query} is"
    return prompt, " " + Cc[q], " " + Cc[f]


def _capital(rng, ctx):
    """In-context factual recall. Returns (prompt, answer, foil)."""
    P = rng.sample(CITIES, 3)
    q = rng.randrange(3)
    f = rng.choice([i for i in range(3) if i != q])
    facts = " ".join(f"{c} is in {k}." for c, k in P)
    if ctx == "neutral":
        note = rng.choice(NEUTRAL)
    elif ctx == "distractor":
        note = f"{P[f][0]} is {rng.choice(ADJ)}"
    else:
        raise ValueError(ctx)
    prompt = f"{facts} Note that {note}. The capital of {P[q][1]} is"
    return prompt, " " + P[q][0], " " + P[f][0]


TASKS = {"binding": (_binding, ["neutral", "distractor", "indirection"]),
         "capital": (_capital, ["neutral", "distractor"])}


class Dataset:
    """Fixed-length clean prompts, each paired with a same-length corrupted prompt."""
    def __init__(self, tok, task, ctx, n=400, seed=0, pair_seed=None):
        gen = TASKS[task][0]
        rng = random.Random(zlib.crc32(f"{task}|{ctx}|{seed}".encode()))
        rows = []
        tries = 0
        while len(rows) < n * 3 and tries < n * 50:
            tries += 1
            p, a, f = gen(rng, ctx)
            ia, ifo = tok.encode(a), tok.encode(f)
            if len(ia) != 1 or len(ifo) != 1:
                continue
            rows.append((p, a, f, tok.encode(p), ia[0], ifo[0]))
        # keep the modal length so no padding is ever needed
        lens = [len(r[3]) for r in rows]
        T = max(set(lens), key=lens.count)
        rows = [r for r in rows if len(r[3]) == T]
        self.T = T
        self.rows = rows[: n]
        self.task, self.ctx = task, ctx
        self.ids = torch.tensor([r[3] for r in self.rows], device=DEV)
        self.ans = torch.tensor([r[4] for r in self.rows], device=DEV)
        self.foil = torch.tensor([r[5] for r in self.rows], device=DEV)
        # corrupted partner: a *different* prompt from a disjoint pool, same length
        prng = random.Random(1000 + (pair_seed if pair_seed is not None else seed))
        pool = rows[n:] if len(rows) > n + 10 else rows
        self.cids = torch.tensor([prng.choice(pool)[3] for _ in self.rows], device=DEV)

    def __len__(self):
        return len(self.rows)

    def subset(self, idx):
        d = object.__new__(Dataset)
        d.__dict__.update(self.__dict__)
        idx = torch.as_tensor(idx, device=DEV)
        d.rows = [self.rows[i] for i in idx.tolist()]
        for k in ["ids", "ans", "foil", "cids"]:
            setattr(d, k, getattr(self, k)[idx])
        return d

    def repair(self, pair_seed):
        """New random corrupted partners (a 'resampling seed')."""
        prng = random.Random(1000 + pair_seed)
        pool = self.cids.tolist() + self.ids.tolist()
        pool = [p for p in pool]
        perm = list(range(len(self)))
        prng.shuffle(perm)
        d = self.subset(list(range(len(self))))
        d.cids = self.cids[torch.tensor(perm, device=DEV)]
        return d


# -------------------------------------------------------------- components
class Comp(tuple):
    """('dn'|'attn', layer, head) or ('mlp', layer, -1)."""
    def __new__(cls, kind, layer, head=-1):
        return super().__new__(cls, (kind, int(layer), int(head)))

    @property
    def name(self):
        k, l, h = self
        return {"dn": f"DN{l}.{h}", "attn": f"AT{l}.{h}", "mlp": f"MLP{l}"}[k]


def all_components(model, mlps=True):
    types = model.config.text_config.layer_types if hasattr(model.config, "text_config") else model.config.layer_types
    out = []
    for l, t in enumerate(types):
        if t == "full_attention":
            out += [Comp("attn", l, h) for h in range(8)]
        else:
            out += [Comp("dn", l, h) for h in range(16)]
        if mlps:
            out.append(Comp("mlp", l))
    return out


def parse(name):
    if name.startswith("MLP"):
        return Comp("mlp", int(name[3:]))
    k = "dn" if name.startswith("DN") else "attn"
    l, h = name[2:].split(".")
    return Comp(k, int(l), int(h))


HD = {"dn": 128, "attn": 256}


# ------------------------------------------------------------- intervener
class Intervener:
    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.edits = {}          # Comp -> (mode, value)
        self.positions = "all"   # or "last"
        self.cache = None        # {('proj'|'mlp', layer): tensor}
        self.recording = False
        self.grad_scales = None  # {Comp: tensor scalar requiring grad}
        self.grad_mode = "scale" # or "atp" (attribution patching toward corrupted cache)
        self.handles = []
        for l, layer in enumerate(self.layers):
            proj = layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj
            kind = "attn" if hasattr(layer, "self_attn") else "dn"
            self.handles.append(proj.register_forward_pre_hook(self._mk_proj_hook(l, kind)))
            self.handles.append(layer.mlp.register_forward_hook(self._mk_mlp_hook(l)))

    def _pos(self, x):
        if self.positions == "all":
            return slice(None)
        if self.positions == "prefix":
            return slice(0, x.shape[1] - 1)
        return slice(x.shape[1] - 1, None)

    def _mk_proj_hook(self, l, kind):
        def hook(mod, args):
            x = args[0]
            if self.recording:
                self.cache[("proj", l)] = x.detach().clone()
                return None
            mine = [(c, e) for c, e in self.edits.items() if c[0] == kind and c[1] == l]
            gs = [] if self.grad_scales is None else [(c, s) for c, s in self.grad_scales.items() if c[0] == kind and c[1] == l]
            if not mine and not gs:
                return None
            x = x.clone()
            hd = HD[kind]
            P = self._pos(x)
            for c, (mode, val) in mine:
                sl = slice(c[2] * hd, (c[2] + 1) * hd)
                if mode == "patch":
                    x[:, P, sl] = self.cache[("proj", l)][:, P, sl]
                elif mode == "zero":
                    x[:, P, sl] = 0
                elif mode == "scale":
                    x[:, P, sl] = x[:, P, sl] * val
            if gs:
                nh = x.shape[-1] // hd
                hs = {c[2]: s for c, s in gs}
                one = torch.ones((), device=x.device, dtype=x.dtype)
                if self.grad_mode == "atp":
                    zero = torch.zeros((), device=x.device, dtype=x.dtype)
                    vec = torch.stack([hs.get(h, zero) for h in range(nh)]).repeat_interleave(hd)
                    x = x + (self.cache[("proj", l)] - x) * vec
                else:
                    vec = torch.stack([hs.get(h, one) for h in range(nh)]).repeat_interleave(hd)
                    x = x * vec
            return (x,)
        return hook

    def _mk_mlp_hook(self, l):
        def hook(mod, args, out):
            if self.recording:
                self.cache[("mlp", l)] = out.detach().clone()
                return None
            c = Comp("mlp", l)
            y = out
            changed = False
            if c in self.edits:
                mode, val = self.edits[c]
                y = y.clone(); P = self._pos(y); changed = True
                if mode == "patch":
                    y[:, P] = self.cache[("mlp", l)][:, P]
                elif mode == "zero":
                    y[:, P] = 0
                elif mode == "scale":
                    y[:, P] = y[:, P] * val
            if self.grad_scales is not None and c in self.grad_scales:
                if self.grad_mode == "atp":
                    y = y + (self.cache[("mlp", l)] - y) * self.grad_scales[c]
                else:
                    y = y * self.grad_scales[c]
                changed = True
            return y if changed else None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------ running
class Runner:
    """Holds a dataset, its corrupted cache, and evaluates intervention sets."""
    def __init__(self, model, ds, bs=512):
        self.model, self.ds, self.bs = model, ds, bs
        self.iv = getattr(model, "_eco_iv", None)
        if self.iv is None:
            self.iv = Intervener(model)
            model._eco_iv = self.iv
        self.caches = []
        self.iv.edits = {}
        self.iv.grad_scales = None
        for s in range(0, len(ds), bs):
            self.iv.cache = {}
            self.iv.recording = True
            with torch.no_grad():
                self.model(ds.cids[s:s + bs], logits_to_keep=1)
            self.iv.recording = False
            self.caches.append(self.iv.cache)
        self.clean = self.ld({})

    def ld(self, edits, positions="all", grad=False):
        """Per-prompt logit difference (answer - foil) at the final position."""
        iv = self.iv
        iv.edits = dict(edits)
        iv.positions = positions
        outs = []
        for bi, s in enumerate(range(0, len(self.ds), self.bs)):
            iv.cache = self.caches[bi]
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                lg = self.model(self.ds.ids[s:s + self.bs], logits_to_keep=1).logits[:, -1].float()
                a = self.ds.ans[s:s + self.bs]; f = self.ds.foil[s:s + self.bs]
                d = lg.gather(1, a[:, None])[:, 0] - lg.gather(1, f[:, None])[:, 0]
            outs.append(d if grad else d.detach())
        iv.edits = {}
        return torch.cat(outs)

    def D(self, comps, mode="patch", positions="all"):
        """Per-prompt damage: drop in logit diff when `comps` are ablated."""
        return self.clean - self.ld({c: (mode, None) for c in comps}, positions)

    def grad_attr(self, base_edits, cands, mode="scale"):
        """mode=scale: d LD / d scale_c at 1.  mode=atp: d LD / d t_c at 0 where the
        slice moves toward its corrupted value (first-order estimate of -damage)."""
        iv = self.iv
        iv.grad_mode = mode
        init = torch.ones if mode == "scale" else torch.zeros
        iv.grad_scales = {c: init((), device=DEV, dtype=torch.bfloat16, requires_grad=True) for c in cands}
        tot = 0
        MB = 48
        for bi, s in enumerate(range(0, len(self.ds), self.bs)):
            n = min(self.bs, len(self.ds) - s)
            for u in range(0, n, MB):
                iv.edits = dict(base_edits)
                iv.cache = {k: v[u:u + MB] for k, v in self.caches[bi].items()}
                with torch.enable_grad():
                    lg = self.model(self.ds.ids[s + u:s + min(u + MB, n)], logits_to_keep=1).logits[:, -1].float()
                    a = self.ds.ans[s + u:s + min(u + MB, n)]; f = self.ds.foil[s + u:s + min(u + MB, n)]
                    d = (lg.gather(1, a[:, None])[:, 0] - lg.gather(1, f[:, None])[:, 0]).sum()
                    d.backward()
        g = {c: float(v.grad) / len(self.ds) for c, v in iv.grad_scales.items()}
        iv.grad_scales = None; iv.edits = {}; iv.grad_mode = "scale"
        return g


# ---------------------------------------------------------------- stats
def boot_ci(x, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def filter_correct(tok, model, ds, margin=0.5):
    r = Runner(model, ds)
    keep = (r.clean > margin).nonzero()[:, 0]
    acc = float((r.clean > 0).float().mean())
    return ds.subset(keep), acc


def build(tok, model, task, ctx, n=600, seed=0, n_disc=None):
    ds = Dataset(tok, task, ctx, n=n, seed=seed)
    ds, acc = filter_correct(tok, model, ds)
    return ds, acc
