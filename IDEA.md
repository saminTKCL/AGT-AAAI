# AGT-Implicit — Analytic Granularity + Learned Implicit Branch

---

### What problem are we solving?

Imagine a social network where you want to guess each person’s category (student, professor, etc.) by looking at their connections.

On some networks, **friends are similar** — your neighbors help a lot.  
On others, **friends are different** — your neighbors can mislead you, and you may need to look elsewhere.

The hard question for every person is:

- **How far should I look around them?** (only close friends? friends-of-friends? farther?)
- **Should I also search for similar people who are not directly connected?**

A recent method called **GRAIN** (AAAI 2025) handles this by using **reinforcement learning (RL)** — basically a trial-and-error robot that keeps experimenting until it learns a strategy. That works, but it is slow, complicated, and hard to explain.

### What are we proposing?

We split the problem into two simpler parts:

| Part | What it does | How we handle it |
|------|----------------|------------------|
| **Explicit** (look at graph neighbors) | Decide how many hops to aggregate for each node | **Use a formula** based on local patterns (homophily + degree) — no trial-and-error robot |
| **Implicit** (look beyond direct neighbors) | Find semantically similar nodes far away on the graph | **Keep a small learned module** (similarity search + mixing weight) |

Our method is **AGT-Implicit**: a smart rule for the obvious part, plus learning only where it is actually needed.

### Simple analogy

- **Old way (GRAIN):** A driver randomly tries many routes every day to “learn” the best path.
- **Our way:** Use a map and traffic rules for most routing (fast, explainable), and use live traffic learning only for tricky hidden shortcuts.

### What we have learned so far

1. **Early results were misleading** because some datasets used wrong train/test splits. We fixed that and now use official splits only.
2. **Rule-only (AGT explicit)** was not enough on its own — it lost to RL on some hard datasets (especially university webpage graphs like Texas/Wisconsin).
3. **AGT-Implicit (rule + learned implicit branch)** is the current best story:
   - Strong gains on WebKB-style graphs (Texas, Cornell, Wisconsin)
   - Faster training than GRAIN-TD3 (often ~2×)
   - Mixed results on some other benchmarks (Chameleon, Squirrel) — we report this honestly

### One-sentence pitch

**We show that GRAIN’s RL is largely unnecessary for choosing neighbor-hop depth; a theory-backed formula plus a small learned “far similarity” module gives comparable or better results with less cost and more interpretability.**

### Target venue

AAAI-27 (abstract Jul 21, 2026; full paper Jul 28, 2026)

---

## Technical summary

### Problem

GRAIN uses TD3 RL to learn per-node aggregation granularity on heterophilous graphs. RL adds replay buffers, critics, exploration noise, and training cost for what is essentially a one-step per-node decision. The released GRAIN code also combines **explicit** (multi-hop neighbor) and **implicit** (semantic similarity) views — but RL redundantly re-learns the explicit depth.

### Method: AGT-Implicit

1. **Explicit branch (AGT):** Compute per-node local statistics (homophily $h_v$, degree $d_v$, feature SNR). Map to continuous granularity $k^*(h_v, d_v)$ via closed form under degree-corrected CSBM. Apply fractional-hop GRAIN aggregation.
2. **Implicit branch:** kNN cosine-similarity aggregation over all nodes (GRAIN’s non-neighbor signal), with learnable per-node gate $\beta_v$.
3. **Output:** $\mathbf{h}_v = \mathbf{h}^{\mathrm{explicit}}_v + \beta_v \cdot \mathbf{h}^{\mathrm{implicit}}_v$. Train classifier, gate, and optional calibration jointly — **no RL**.

### Novelty vs GRAIN

| Aspect | GRAIN-TD3 | AGT-Implicit |
|--------|-----------|--------------|
| Explicit granularity | TD3 RL (actor + critic + replay) | Closed-form from CSBM theory |
| Implicit semantics | Learned (in paper; partial in released code) | kNN + learnable gate |
| Training cost | High (RL loop + meta-training) | Lower (GNN phase only) |
| Interpretability | Opaque policy | $k^*$ correlates with local homophily |
| Theory | None for explicit depth | CSBM optimality (see `paper/theory_appendix.md`) |

### Code

```bash
pip install -r requirements.txt
python3 src/experiments/run_benchmark.py --seeds 10
python3 src/experiments/run_gonogo.py --datasets Texas Actor --seeds 3
python3 src/experiments/run_ablations.py --dataset Actor --seeds 10
python3 src/experiments/generate_results.py
```

Results: `RESULTS.md`, `results/benchmark_summary.csv`
