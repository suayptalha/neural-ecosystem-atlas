# Neural Ecosystem Atlas

Code, results and figures for the blog post **"Neural Networks Are Secretly Biological Processes"**, which studies how the components of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) interact. The 288 Gated DeltaNet heads, 48 gated-attention heads and 24 MLP blocks are treated as an ecosystem and turned off one at a time and in pairs, using resample ablation.

The four experiments:
1. **Hidden Ecosystem**: pairwise interactions among the 16 most important components
2. **Role Reversal**: interaction signs that flip across prompt environments
3. **Cascading Extinction**: conditional importance after removing a keystone component
4. **Ecosystem Restoration**: repairing a damaged model by removing competitors or amplifying helpers

## Layout

| Path | Contents |
|---|---|
| `code/eco.py` | Shared library: datasets, component definitions, intervention hooks, metrics |
| `code/exp12.py`, `code/exp1_random.py` | Experiments 1–2 and the random-pair control |
| `code/exp3.py` | Experiment 3 |
| `code/exp4b.py`, `code/gen_text.py` | Experiment 4 |
| `code/analyze12.py`, `code/analyze3.py`, `code/analyze4.py`, `code/fig0_landscape.py`, `code/style.py` | Analyses and figures |
| `results/` | Raw results (per-prompt values) and summary statistics |
| `figures/` | All figures used in the post |

## Reproducing

The experiment scripts were run on Google Colab GPUs (A100 / L4), in bf16. They expect `eco.py` (and `gen_text.py`) in `/content` and write to `/content/results/`. `exp12.py` and `exp1_random.py` also read `results/atlas_components.json` and `results/screen.json` from there.

```bash
pip install -r requirements.txt
python exp12.py binding neutral,distractor,indirection exp12_binding
python exp12.py capital neutral,distractor exp12_capital
python exp1_random.py binding neutral
python exp3.py binding neutral        # and: capital neutral
python exp4b.py binding neutral 4     # and: capital neutral 3
```

The analysis scripts run locally from `code/`. They read `../results/` and write `../figures/`:

```bash
cd code && python fig0_landscape.py && python analyze12.py && python analyze3.py && python analyze4.py
```

All prompts are generated deterministically with fixed seeds.
