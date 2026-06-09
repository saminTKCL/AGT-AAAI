#!/usr/bin/env python3
"""Comprehensive analysis:
1. k* vs local homophily correlation (validates theory)
2. k* vs TD3 learned actions correlation (confirms RL re-learns analytic map)
3. Graph structure analysis for Chameleon/Squirrel (explains regression)
4. Per-dataset implicit gate beta distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.load import load_dataset, DATASETS
from src.models.grain_env import GrainEnvironment
from src.models.local_stats import compute_local_stats
from src.train.trainer import train_grain_td3


# ─── k* vs Homophily ────────────────────────────────────────────────────────

def analyze_kstar_homophily(dataset: str, seeds: int, device: torch.device, out_dir: Path):
    """Compute k* vs homophily correlation across seeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    corrs, all_h, all_k = [], [], []
    for seed in range(seeds):
        data = load_dataset(dataset, seed=seed, device=device)
        env = GrainEnvironment(data, device=device)
        policy = env.make_agt_policy(use_calibration=False)
        k = policy.granularity.cpu().numpy()
        h = policy.stats["homophily"].cpu().numpy()
        corrs.append(float(np.corrcoef(h, k)[0, 1]))
        if seed == 0:
            all_h.extend(h.tolist())
            all_k.extend(k.tolist())

    mean_corr = float(np.mean(corrs))

    if plt is not None and len(all_h) > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(all_h, all_k, alpha=0.4, s=6, rasterized=True)
        ax.set_xlabel("Local homophily $h_v$")
        ax.set_ylabel("Analytic granularity $k^*_v$")
        ax.set_title(f"{dataset} — r={mean_corr:.3f}")
        fig.tight_layout()
        fig.savefig(out_dir / f"kstar_homophily_{dataset}.pdf", dpi=150)
        plt.close(fig)

    return {"dataset": dataset, "mean_corr": mean_corr, "corrs": corrs}


# ─── k* vs TD3 actions ──────────────────────────────────────────────────────

def analyze_kstar_vs_td3(dataset: str, seed: int, device: torch.device, out_dir: Path, episodes: int = 60):
    """Correlate analytic k* with TD3-learned actions on train nodes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_dataset(dataset, seed=seed, device=device)
    env = GrainEnvironment(data, device=device)

    # Analytic k*
    agt_policy = env.make_agt_policy(use_calibration=False)
    k_star = agt_policy.granularity.cpu().numpy()

    # TD3 learned actions (train nodes after training)
    train_grain_td3(env, rl_episodes=episodes, gnn_episodes=episodes)
    # Re-query the final policy on all train nodes
    td3_policy = env.policy
    states = data.x[env.train_idx].cpu().numpy()
    td3_actions = np.array(td3_policy.select_action(states)).flatten()
    k_agt = k_star[env.train_idx.cpu().numpy()]

    corr = float(np.corrcoef(k_agt, td3_actions)[0, 1])

    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(k_agt, td3_actions, alpha=0.4, s=8, rasterized=True)
        mn, mx = min(k_agt.min(), td3_actions.min()), max(k_agt.max(), td3_actions.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y=x")
        ax.set_xlabel("Analytic $k^*_v$")
        ax.set_ylabel("TD3 learned action")
        ax.set_title(f"{dataset} — r={corr:.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"kstar_vs_td3_{dataset}.pdf", dpi=150)
        plt.close(fig)

    return {"dataset": dataset, "seed": seed, "kstar_td3_corr": corr}


# ─── Graph structure analysis (Chameleon/Squirrel) ──────────────────────────

def analyze_graph_structure(datasets: list[str], seed: int, device: torch.device, out_dir: Path):
    """
    For each dataset compute:
    - Global homophily
    - Mean / std of local homophily
    - Fraction of nodes with h_v < 0.2 (strongly heterophilous)
    - Fraction of nodes with 0.2 <= h_v <= 0.6 (ambiguous)
    - kNN feature-label alignment: among top-10 similar nodes, fraction with same label
    This explains why implicit kNN branch hurts on Chameleon/Squirrel.
    """
    rows = []
    for ds in datasets:
        data = load_dataset(ds, seed=seed, device=device)
        stats = compute_local_stats(data)
        h = stats["homophily"].cpu().numpy()

        # kNN feature-label alignment
        x_norm = F.normalize(data.x.detach(), p=2, dim=-1)
        k = min(10, data.num_nodes - 1)
        sim = (x_norm @ x_norm.t())
        sim.fill_diagonal_(-1.0)
        topk_idx = sim.topk(k, dim=-1).indices  # (N, k)
        y = data.y
        same_label = (y[topk_idx] == y.unsqueeze(-1)).float().mean(dim=-1)
        knn_align = float(same_label.cpu().mean())

        row = {
            "dataset": ds,
            "global_homophily": float(h.mean()),
            "h_mean": float(h.mean()),
            "h_std": float(h.std()),
            "frac_strongly_hetero": float((h < 0.2).mean()),
            "frac_ambiguous": float(((h >= 0.2) & (h <= 0.6)).mean()),
            "frac_homo": float((h > 0.6).mean()),
            "knn_feature_label_align": knn_align,
        }
        rows.append(row)
        print(
            f"{ds:12}: h_mean={row['h_mean']:.3f}  "
            f"h_std={row['h_std']:.3f}  "
            f"strongly_hetero={row['frac_strongly_hetero']:.2f}  "
            f"ambiguous={row['frac_ambiguous']:.2f}  "
            f"knn_align={knn_align:.3f}"
        )

    return rows


# ─── Implicit gate analysis ──────────────────────────────────────────────────

def analyze_implicit_gate(datasets: list[str], seed: int, device: torch.device, out_dir: Path):
    """Show per-dataset mean learned beta (implicit gate weight)."""
    rows = []
    for ds in datasets:
        data = load_dataset(ds, seed=seed, device=device)
        env = GrainEnvironment(data, device=device)
        mv = env.make_multiview_aggregator(implicit_k=10)
        # Gate beta on all nodes
        stats = env._local_stats
        stat_matrix = mv.agt._stat_matrix()
        with torch.no_grad():
            beta = mv.gate(stat_matrix).cpu().numpy()
        rows.append({
            "dataset": ds,
            "beta_mean": float(beta.mean()),
            "beta_std": float(beta.std()),
            "beta_min": float(beta.min()),
            "beta_max": float(beta.max()),
        })
        print(f"{ds:12}: beta mean={rows[-1]['beta_mean']:.3f} ± {rows[-1]['beta_std']:.3f}")
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, default=5, help="Seeds for correlation analysis")
    parser.add_argument("--td3-datasets", nargs="+", default=["Texas", "Actor", "Chameleon"],
                        help="Datasets for k* vs TD3 correlation (slow, runs TD3)")
    parser.add_argument("--td3-episodes", type=int, default=60)
    parser.add_argument("--skip-td3", action="store_true", help="Skip slow TD3 correlation run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. k* vs homophily correlation
    print("\n=== k* vs Homophily Correlation ===")
    kstar_results = []
    for ds in args.datasets:
        r = analyze_kstar_homophily(ds, args.seeds, device, out_dir)
        kstar_results.append(r)
        print(f"  {ds:12}: r={r['mean_corr']:+.3f}")

    (out_dir / "kstar_homophily_corr.json").write_text(json.dumps(kstar_results, indent=2))

    # 2. Graph structure analysis
    print("\n=== Graph Structure Analysis (kNN feature-label alignment) ===")
    struct_rows = analyze_graph_structure(args.datasets, seed=0, device=device, out_dir=out_dir)
    (out_dir / "graph_structure.json").write_text(json.dumps(struct_rows, indent=2))

    # 3. k* vs TD3 (optional, slow)
    if not args.skip_td3:
        print("\n=== k* vs TD3 Correlation ===")
        td3_corrs = []
        for ds in args.td3_datasets:
            try:
                r = analyze_kstar_vs_td3(ds, seed=0, device=device, out_dir=out_dir,
                                         episodes=args.td3_episodes)
                td3_corrs.append(r)
                print(f"  {ds}: r={r['kstar_td3_corr']:+.3f}")
            except Exception as e:
                print(f"  {ds}: FAILED ({e})")
        (out_dir / "kstar_td3_corr.json").write_text(json.dumps(td3_corrs, indent=2))

    # 4. Implicit gate analysis
    print("\n=== Implicit Gate Beta Distribution ===")
    gate_rows = analyze_implicit_gate(args.datasets, seed=0, device=device, out_dir=out_dir)
    (out_dir / "implicit_gate.json").write_text(json.dumps(gate_rows, indent=2))

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
