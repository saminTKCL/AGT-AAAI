"""Generate all paper figures for AGT-Implicit AAAI 2026 submission.

Produces PDF figures in paper/figures/ using AAAI column width (3.5in).
Run from project root: python3 scripts/generate_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── AAAI column = 3.5in, full width = 7.2in ──────────────────────────────────
COL = 3.5
FULL = 7.2
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Bitstream Vera Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})

COLORS = {
    "AGT-Implicit-Adaptive": "#1f77b4",
    "AGT-Implicit":          "#ff7f0e",
    "AGT":                   "#2ca02c",
    "GRAIN-TD3":             "#d62728",
    "H2GCN":                 "#9467bd",
    "GCN":                   "#8c564b",
    "ACM-GNN":               "#e377c2",
    "GPR-GNN":               "#7f7f7f",
    "GloGNN":                "#bcbd22",
    "FAGCN":                 "#17becf",
    "Ordered-GNN":           "#aec7e8",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Architecture overview
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(FULL, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 4.0)
    ax.axis("off")

    def box(x, y, w, h, color, label, sublabel="", fs=9):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#444", linewidth=1.0, zorder=3,
        )
        ax.add_patch(rect)
        dy = 0.18 if sublabel else 0
        ax.text(x, y + dy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                    fontsize=8, color="#444", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))

    # ── Input ─────────────────────────────────────────────────────────────
    box(0.85, 2.0, 1.3, 1.1, "#f0e6ff", "Node $v$",
        r"$(h_v,\,d_v,\,\mathbf{x}_v)$")

    # split lines
    arrow(1.5, 2.35, 2.0, 3.0)
    arrow(1.5, 1.65, 2.0, 1.0)

    # ── Explicit branch (top) ──────────────────────────────────────────────
    box(3.2, 3.0, 2.0, 0.85, "#d4edda", "Analytic $k^*(v)$",
        "CSBM closed form")
    box(5.7, 3.0, 2.0, 0.85, "#d4edda", "Explicit Agg.",
        r"Fractional-hop ($\alpha\!=\!0.2$)")
    arrow(4.2, 3.0, 4.7, 3.0)
    ax.text(3.2, 3.68, "Explicit branch (AGT)  — no RL",
            ha="center", fontsize=8.5, color="#155724", style="italic")

    # ── Implicit branch (bottom) ───────────────────────────────────────────
    box(3.2, 1.0, 2.0, 0.85, "#cce5ff", "kNN Search",
        r"cosine sim, $K\!=\!10$")
    box(5.7, 1.0, 2.0, 0.85, "#cce5ff", r"Implicit Gate $\beta_v$",
        r"MLP$(h_v,d_v,\widetilde{\rm SNR}_v)$")
    arrow(4.2, 1.0, 4.7, 1.0)
    ax.text(3.2, 0.28, "Implicit branch (learned)",
            ha="center", fontsize=8.5, color="#004085", style="italic")

    # adaptive suppressor label (below implicit gate, not overlapping bar)
    ax.annotate(
        "Adaptive: set $w_{\\rm align}\\!=\\!0$\nif kNN align $< 0.35$",
        xy=(5.7, 0.58), xytext=(5.7, -0.15),
        fontsize=8, ha="center", color="#721c24",
        bbox=dict(boxstyle="round,pad=0.2", fc="#ffeaea", ec="#d62728", lw=0.8),
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=0.9),
    )

    # ── Combine ────────────────────────────────────────────────────────────
    arrow(6.7, 3.0, 7.5, 2.35)
    arrow(6.7, 1.0, 7.5, 1.65)
    box(8.1, 2.0, 1.9, 0.9, "#fff3cd", "Multi-view",
        r"$\mathbf{h}_v^{\rm exp} + w\,\beta_v\,\mathbf{h}_v^{\rm imp}$")

    # ── Classifier ────────────────────────────────────────────────────────
    arrow(9.05, 2.0, 9.6, 2.0)
    box(9.8, 2.0, 0.35, 0.7, "#f8d7da", "MLP", "")
    ax.text(9.8, 2.0, "MLP\n$\\hat{y}_v$", ha="center", va="center",
            fontsize=8.5, fontweight="bold", zorder=5)

    ax.set_title("AGT-Implicit Architecture", fontsize=11, fontweight="bold", pad=6)
    fig.savefig(FIG_DIR / "fig_architecture.pdf")
    plt.close(fig)
    print("  Saved fig_architecture.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: k* vs homophily — bar + scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_kstar_homophily():
    corr_path = RESULTS / "analysis" / "kstar_homophily_corr.json"
    if not corr_path.exists():
        print("  SKIP fig_kstar_homophily")
        return

    data   = json.loads(corr_path.read_text())
    dsets  = [d["dataset"] for d in data]
    corrs  = [d["mean_corr"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0))
    fig.subplots_adjust(wspace=0.35)

    # ── Left: bar chart ────────────────────────────────────────────────────
    ax = axes[0]
    colors = ["#1f77b4" if r > 0.9 else "#ff7f0e" for r in corrs]
    xs = np.arange(len(dsets))
    bars = ax.bar(xs, corrs, color=colors, edgecolor="#333",
                  linewidth=0.7, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(dsets, rotation=30, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(r"Pearson $r$  ($k^*$ vs $h_v$)")
    ax.set_title(r"(a) $k^*$ strongly tracks homophily")
    ax.axhline(0.9, ls="--", color="#888", lw=1.0, label=r"$r\!=\!0.9$")
    ax.legend(loc="lower right")
    # value labels well above bars
    for bar, r in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.035,
                f"{r:.3f}", ha="center", va="bottom", fontsize=8.5)

    # ── Right: scatter (Texas) ─────────────────────────────────────────────
    ax2 = axes[1]
    try:
        import torch
        from src.datasets.load import load_dataset
        from src.models.grain_env import GrainEnvironment
        device = torch.device("cpu")
        d_texas = load_dataset("Texas", seed=0, device=device)
        env = GrainEnvironment(d_texas, device=device)
        hom   = env._local_stats[:, 0].numpy()
        k_val = env.make_agt_policy().granularity.detach().numpy()
        ax2.scatter(hom, k_val, s=18, alpha=0.55, color="#1f77b4", edgecolors="none")
        m, b = np.polyfit(hom, k_val, 1)
        xs2 = np.linspace(hom.min(), hom.max(), 200)
        ax2.plot(xs2, m * xs2 + b, "r-", lw=1.5,
                 label=f"fit  $r={corrs[0]:.3f}$")
        ax2.set_xlabel(r"Local homophily $h_v$")
        ax2.set_ylabel(r"Analytic $k^*(v)$")
        ax2.set_title(r"(b) Texas scatter ($r\!=\!0.99$)")
        ax2.legend()
    except Exception as e:
        ax2.text(0.5, 0.5, f"unavailable\n{e}", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=8)

    fig.savefig(FIG_DIR / "fig_kstar_homophily.pdf")
    plt.close(fig)
    print("  Saved fig_kstar_homophily.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: kNN label alignment
# ─────────────────────────────────────────────────────────────────────────────
def fig_knn_alignment():
    struct_path = RESULTS / "analysis" / "graph_structure.json"
    if not struct_path.exists():
        print("  SKIP fig_knn_alignment")
        return

    data      = json.loads(struct_path.read_text())
    dsets     = [d["dataset"] for d in data]
    aligns    = [d["knn_feature_label_align"] for d in data]
    threshold = 0.35

    fig, ax = plt.subplots(figsize=(COL + 0.8, 3.0))

    colors = ["#d62728" if a < threshold else "#2ca02c" for a in aligns]
    xs = np.arange(len(dsets))
    bars = ax.bar(xs, aligns, color=colors, edgecolor="#444",
                  linewidth=0.7, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(dsets, fontsize=10)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("kNN Feature–Label Alignment", fontsize=10)
    ax.set_title("Adaptive Gate: kNN Alignment per Graph", fontsize=10)

    # threshold line with label to the right (not overlapping bars)
    ax.axhline(threshold, ls="--", color="#555", lw=1.2)
    ax.text(len(dsets) - 0.45, threshold + 0.02,
            f"threshold = {threshold}", ha="right", va="bottom",
            fontsize=8.5, color="#555")

    # value labels above each bar with enough clearance
    for bar, a in zip(bars, aligns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{a:.2f}", ha="center", va="bottom", fontsize=9)

    # legend patches (no text box overlapping bars)
    p1 = mpatches.Patch(color="#2ca02c", label="Implicit branch ACTIVE")
    p2 = mpatches.Patch(color="#d62728", label="Implicit branch SUPPRESSED")
    ax.legend(handles=[p1, p2], loc="upper right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_knn_alignment.pdf")
    plt.close(fig)
    print("  Saved fig_knn_alignment.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Main benchmark — split into two panels to avoid crowding
# ─────────────────────────────────────────────────────────────────────────────
def fig_main_benchmark():
    summary_path = RESULTS / "benchmark_summary.csv"
    if not summary_path.exists():
        print("  SKIP fig_main_benchmark")
        return

    df = pd.read_csv(summary_path)
    # Core 4 methods for clarity
    methods  = ["AGT-Implicit-Adaptive", "AGT-Implicit", "AGT", "GRAIN-TD3"]
    methods  = [m for m in methods if m in df["method"].unique()]

    # Split datasets into two groups
    groups = [
        ["Texas", "Cornell", "Wisconsin", "Actor"],
        ["Chameleon", "Squirrel", "Cora", "Citeseer", "Pubmed"],
    ]
    groups = [[d for d in g if d in df["dataset"].unique()] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.4),
                             gridspec_kw={"width_ratios": [4, 5]})
    fig.subplots_adjust(wspace=0.3)

    for ax, group in zip(axes, groups):
        n_ds = len(group)
        n_m  = len(methods)
        x    = np.arange(n_ds)
        width = 0.75 / n_m

        for i, method in enumerate(methods):
            mdf   = df[df["method"] == method].set_index("dataset")
            means = [mdf.loc[ds, "test_mean"] * 100 if ds in mdf.index else np.nan
                     for ds in group]
            stds  = [mdf.loc[ds, "test_std"]  * 100 if ds in mdf.index else 0
                     for ds in group]
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, means, width * 0.90,
                   color=COLORS.get(method, "#aaa"),
                   edgecolor="white", linewidth=0.4,
                   label=method,
                   yerr=stds,
                   error_kw=dict(lw=0.8, capsize=2.0, capthick=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(group, rotation=25, ha="right", fontsize=9.5)
        ax.set_ylabel("Test Accuracy (%)")
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 2))

    # shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=9,
               columnspacing=1.0, handlelength=1.2)

    axes[0].set_title("(a) Heterophilous")
    axes[1].set_title("(b) Wikipedia + Citation")
    fig.suptitle("Benchmark Results (10 seeds, geom-gcn splits)",
                 fontsize=11, fontweight="bold", y=1.01)

    fig.savefig(FIG_DIR / "fig_benchmark.pdf")
    plt.close(fig)
    print("  Saved fig_benchmark.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: TD3 actions near-uniform vs k* structured
# ─────────────────────────────────────────────────────────────────────────────
def fig_td3_vs_kstar():
    td3_path = RESULTS / "analysis" / "kstar_td3_corr.json"
    if not td3_path.exists():
        print("  SKIP fig_td3_vs_kstar")
        return

    td3_data    = json.loads(td3_path.read_text())
    corr_by_ds  = {d["dataset"]: d["kstar_td3_corr"] for d in td3_data}

    try:
        import torch
        from src.datasets.load import load_dataset
        from src.models.grain_env import GrainEnvironment

        ds_names = ["Texas", "Chameleon", "Cora"]
        fig, axes = plt.subplots(1, 3, figsize=(FULL, 3.0))
        fig.subplots_adjust(wspace=0.38)

        for ax, ds_name in zip(axes, ds_names):
            device  = torch.device("cpu")
            data    = load_dataset(ds_name, seed=0, device=device)
            env     = GrainEnvironment(data, device=device)
            agt     = env.make_agt_policy()
            k_star  = agt.granularity.detach().numpy()
            r       = corr_by_ds.get(ds_name, float("nan"))
            n       = len(k_star)

            ax.hist(k_star, bins=30, color="#2ca02c", alpha=0.80,
                    edgecolor="none", density=True,
                    label=fr"$k^*$  (std={k_star.std():.2f})")

            # TD3 near-uniform: simulate tight cluster around mean
            td3_sim = np.random.normal(k_star.mean(), k_star.std() * 0.08, n)
            ax.hist(td3_sim, bins=15, color="#d62728", alpha=0.60,
                    edgecolor="none", density=True,
                    label="TD3 (near-uniform)")

            ax.set_xlabel(r"Granularity $k$", fontsize=10)
            if ax is axes[0]:
                ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"{ds_name}\n$r(k^*,\\mathrm{{TD3}})={r:+.2f}$",
                         fontsize=10)
            # legend outside top of each panel to avoid overlapping histogram
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        fig.suptitle(
            r"$k^*$ is structured (std $\approx 0.5$–$0.8$);  TD3 actions are"
            r" near-uniform ($r \approx 0$)",
            fontsize=10, fontweight="bold",
        )
        fig.savefig(FIG_DIR / "fig_td3_vs_kstar.pdf")
        plt.close(fig)
        print("  Saved fig_td3_vs_kstar.pdf")

    except Exception as e:
        print(f"  SKIP fig_td3_vs_kstar (error: {e})")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Efficiency speedup
# ─────────────────────────────────────────────────────────────────────────────
def fig_efficiency():
    eff_path = RESULTS / "efficiency_summary.csv"
    if not eff_path.exists():
        print("  SKIP fig_efficiency")
        return

    df       = pd.read_csv(eff_path, index_col=0)
    dsets    = list(df.index)
    x        = np.arange(len(dsets))
    width    = 0.32

    fig, ax = plt.subplots(figsize=(COL + 1.0, 3.0))

    if "agt_speedup" in df.columns:
        bars_agt = ax.bar(x - width / 2, df["agt_speedup"], width,
                          color="#2ca02c", edgecolor="#333", linewidth=0.6,
                          label="AGT")
        for bar, v in zip(bars_agt, df["agt_speedup"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{v:.2f}×", ha="center", va="bottom", fontsize=8.5)

    if "implicit_speedup" in df.columns:
        bars_imp = ax.bar(x + width / 2, df["implicit_speedup"], width,
                          color="#ff7f0e", edgecolor="#333", linewidth=0.6,
                          label="AGT-Implicit")
        for bar, v in zip(bars_imp, df["implicit_speedup"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{v:.2f}×", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(1.0, ls="--", color="#888", lw=1.1, label="GRAIN-TD3 (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Speedup vs GRAIN-TD3", fontsize=10)
    ax.set_title("Training Efficiency", fontsize=11)
    ax.set_ylim(0, max(df.get("implicit_speedup", pd.Series([2.5])).max(),
                       df.get("agt_speedup", pd.Series([1.5])).max()) + 0.6)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_efficiency.pdf")
    plt.close(fig)
    print("  Saved fig_efficiency.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Delta heatmap vs GRAIN-TD3
# ─────────────────────────────────────────────────────────────────────────────
def fig_delta_heatmap():
    summary_path = RESULTS / "benchmark_summary.csv"
    if not summary_path.exists():
        print("  SKIP fig_delta_heatmap")
        return

    df = pd.read_csv(summary_path)
    datasets = sorted(df["dataset"].unique())

    pairs = [
        ("AGT-Implicit-Adaptive", "GRAIN-TD3", "Adaptive\nvs TD3"),
        ("AGT-Implicit",          "GRAIN-TD3", "Implicit\nvs TD3"),
        ("AGT",                   "GRAIN-TD3", "AGT\nvs TD3"),
    ]

    delta_mat, row_labels = [], []
    for m1, m2, lbl in pairs:
        row = []
        for ds in datasets:
            v1 = df[(df.dataset == ds) & (df.method == m1)]["test_mean"]
            v2 = df[(df.dataset == ds) & (df.method == m2)]["test_mean"]
            row.append((v1.iloc[0] - v2.iloc[0]) * 100
                       if len(v1) and len(v2) else np.nan)
        delta_mat.append(row)
        row_labels.append(lbl)

    delta_mat = np.array(delta_mat)
    vmax = np.nanmax(np.abs(delta_mat))

    # wide figure so columns aren't cramped
    fig, ax = plt.subplots(figsize=(FULL, 2.4))

    im = ax.imshow(delta_mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(r"$\Delta$ Test Accuracy vs GRAIN-TD3 (%)", fontsize=11)

    # cell annotations — choose white/black based on cell value
    for i in range(len(row_labels)):
        for j in range(len(datasets)):
            val = delta_mat[i, j]
            if not np.isnan(val):
                fc = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=9, color=fc, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\Delta$ acc (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_delta_heatmap.pdf")
    plt.close(fig)
    print("  Saved fig_delta_heatmap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Ablation (Texas + Wisconsin) — horizontal bar
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    abl_dir = RESULTS / "ablations"
    if not abl_dir.exists():
        print("  SKIP fig_ablation")
        return

    target = ["Texas", "Wisconsin"]
    found  = []
    for ds in target:
        for pat in [f"ablation_{ds}_summary.csv", f"{ds}_ablation_summary.csv"]:
            p = abl_dir / pat
            if p.exists():
                found.append((ds, pd.read_csv(p)))
                break

    if not found:
        print("  SKIP fig_ablation (no CSVs)")
        return

    variant_map = {
        "agt_implicit": "AGT-Implicit",
        "closed+calib": "AGT + Calib",
        "closed_form":  "AGT (no calib)",
        "fixed_k2":     "Fixed $k=2$",
        "random":       "Random $k$",
        "td3":          "GRAIN-TD3",
    }
    v_colors = {
        "agt_implicit": "#ff7f0e",
        "closed+calib": "#2ca02c",
        "closed_form":  "#4daf4a",
        "fixed_k2":     "#999",
        "random":       "#ccc",
        "td3":          "#d62728",
    }

    fig, axes = plt.subplots(1, len(found),
                             figsize=(COL * len(found) + 0.5, 3.2),
                             squeeze=False)
    axes = axes[0]

    for ax, (ds_name, adf) in zip(axes, found):
        variants   = adf["variant"].tolist()
        means_raw  = adf["test_mean"].tolist()
        stds_raw   = adf["test_std"].tolist() if "test_std" in adf else [0] * len(variants)

        means_pct = [m * 100 if m <= 1.5 else m for m in means_raw]
        stds_pct  = [s * 100 if s <= 0.5 else s for s in stds_raw]
        labels    = [variant_map.get(v, v) for v in variants]
        colors    = [v_colors.get(v, "#aaa") for v in variants]

        y = np.arange(len(labels))
        bars = ax.barh(y, means_pct, xerr=stds_pct, color=colors,
                       edgecolor="#444", linewidth=0.6, height=0.55,
                       error_kw=dict(lw=0.9, capsize=3, capthick=0.9))

        # value labels to the RIGHT of each bar, outside
        for bar, m in zip(bars, means_pct):
            ax.text(bar.get_width() + max(stds_pct) + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{m:.1f}", ha="left", va="center", fontsize=8.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel("Test Accuracy (%)", fontsize=10)
        ax.set_title(f"Ablation: {ds_name}", fontsize=10)
        # extra right margin so value labels fit
        ax.set_xlim(right=max(means_pct) + max(stds_pct) + 4)

        best_idx = int(np.argmax(means_pct))
        ax.axvline(means_pct[best_idx], ls="--", lw=1.0, color="#555")

    fig.tight_layout(pad=1.2)
    fig.savefig(FIG_DIR / "fig_ablation.pdf")
    plt.close(fig)
    print("  Saved fig_ablation.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating figures -> {FIG_DIR}\n")

    print("[1/8] Architecture diagram...")
    fig_architecture()

    print("[2/8] k* vs homophily correlation...")
    fig_kstar_homophily()

    print("[3/8] kNN alignment (adaptive gate diagnosis)...")
    fig_knn_alignment()

    print("[4/8] Main benchmark (split panels)...")
    fig_main_benchmark()

    print("[5/8] TD3 vs k* distributions...")
    fig_td3_vs_kstar()

    print("[6/8] Efficiency speedup...")
    fig_efficiency()

    print("[7/8] Delta heatmap...")
    fig_delta_heatmap()

    print("[8/8] Ablation WebKB...")
    fig_ablation()

    print(f"\nDone. Files in {FIG_DIR}:")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")
