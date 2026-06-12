"""
Extra evaluation plots not in evaluate.py:
  1. per_session_auc_RF.png       -- fixed (IMU-only visible via dashed line)
  2. roc_curves_per_condition.png -- per-fold operating point scatter
  3. per_class_recall_precision.png -- per-movement recall/precision for RF
  4. statistical_significance.png -- Wilcoxon effect-size chart (corrected)
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

REPO   = Path(__file__).resolve().parents[2]
EVAL   = REPO / "ml" / "evaluation"
PLOTS  = REPO / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "IMU":     "#1565C0",
    "EMG":     "#2E7D32",
    "IMU_EMG": "#C62828",
    "FIS":     "#E65100",
}
LABELS = {
    "IMU":     "IMU-only (RF)",
    "EMG":     "EMG-only (RF)",
    "IMU_EMG": "IMU+sEMG (RF)",
    "FIS":     "Mamdani FIS (FIS)",
}
LINESTYLES = {
    "IMU":     (0, (5, 2)),  # dashed -- clearly different from IMU_EMG
    "EMG":     "solid",
    "IMU_EMG": "solid",
    "FIS":     (0, (3, 1, 1, 1)),  # dash-dot
}
MARKERS = {"IMU": "D", "EMG": "o", "IMU_EMG": "o", "FIS": "+"}

loso = pd.read_csv(EVAL / "loso_results.csv")
summ = pd.read_csv(EVAL / "summary_results.csv")

# 1. PER-SESSION AUC  (fixed overlap)

def plot_per_session_auc_fixed():
    """Per-fold AUC for each condition. The dashed IMU-only line and offset
    labels exist purely to stop it disappearing under the near-identical
    IMU+sEMG line -- the small RQ2 sEMG lift makes them overlap otherwise."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # For each condition pick the best classifier
    clf_map = {"IMU": "RF", "EMG": "RF", "IMU_EMG": "RF", "FIS": "FIS"}

    # vertical offset so IMU-only labels don't overlap IMU+sEMG labels
    v_offset = {"IMU": 14, "EMG": 8, "IMU_EMG": -18, "FIS": 8}

    for cond, clf in clf_map.items():
        data = loso[(loso["classifier"] == clf) & (loso["condition"] == cond)]
        if data.empty:
            continue
        data = data.sort_values("fold")
        ax.plot(
            data["fold"], data["auc"],
            marker=MARKERS[cond],
            color=PALETTE[cond],
            linewidth=2.2, markersize=8,
            linestyle=LINESTYLES[cond],
            label=LABELS[cond],
            zorder=4 if cond == "IMU" else 3,
        )
        for _, row in data.iterrows():
            ax.annotate(
                f"{row['auc']:.3f}",
                (row["fold"], row["auc"]),
                textcoords="offset points",
                xytext=(0, v_offset[cond]),
                ha="center", fontsize=7.5,
                color=PALETTE[cond],
                fontweight="bold" if cond == "IMU" else "normal",
            )

    # x-tick labels
    imu_data = loso[(loso["condition"] == "IMU") & (loso["classifier"] == "RF")].sort_values("fold")
    ax.set_xticks(imu_data["fold"].unique())
    ax.set_xticklabels(
        [f"Fold {int(r.fold)}\n({r.test_session})" for _, r in imu_data.iterrows()],
        fontsize=8,
    )

    ax.set_xlabel("LOSO Fold (held-out session)", fontsize=10)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_ylim(0.40, 1.14)
    ax.set_title("Per-Session AUC -- Three Conditions", fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance")
    ax.legend(framealpha=0.9, fontsize=9, loc="lower right")
    fig.tight_layout()
    path = PLOTS / "per_session_auc_RF.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

# 2. ROC OPERATING-POINT SCATTER  (per-fold sensitivity vs 1-specificity)

def plot_roc_scatter():
    """One operating point per LOSO fold (sensitivity vs 1-specificity) for each
    condition, with the mean point starred. Full ROC curves are not available
    per fold, so the spread of operating points stands in for them."""
    conditions = ["IMU", "EMG", "IMU_EMG"]
    clf_map    = {"IMU": "RF", "EMG": "RF", "IMU_EMG": "RF"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharex=True, sharey=True)

    for ax, cond in zip(axes, conditions):
        clf  = clf_map[cond]
        data = loso[(loso["classifier"] == clf) & (loso["condition"] == cond)]
        if data.empty:
            ax.axis("off")
            continue

        fpr  = 1.0 - data["specificity"].values
        tpr  = data["sensitivity"].values
        fold = data["fold"].values
        mean_auc = data["auc"].mean()

        # fold scatter
        sc = ax.scatter(fpr, tpr, c=data["auc"].values, cmap="Blues",
                        vmin=0.8, vmax=1.0, s=80, zorder=4,
                        edgecolors=PALETTE[cond], linewidths=1.5)
        for f, x, y in zip(fold, fpr, tpr):
            ax.annotate(f"F{int(f)}", (x, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=PALETTE[cond])

        # mean operating point
        mean_fpr = fpr.mean()
        mean_tpr = tpr.mean()
        ax.scatter([mean_fpr], [mean_tpr], marker="*", s=250,
                   color=PALETTE[cond], zorder=5, label="Mean op. point")

        # chance diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Chance")
        ax.set_xlim(-0.02, 0.4)
        ax.set_ylim(0.55, 1.05)
        ax.set_xlabel("1 - Specificity (FPR)", fontsize=9)
        if cond == "IMU":
            ax.set_ylabel("Sensitivity (TPR)", fontsize=9)

        cond_title = {"IMU": "IMU-only", "EMG": "EMG-only", "IMU_EMG": "IMU+sEMG"}
        ax.set_title(
            f"{cond_title[cond]}\nMean AUC = {mean_auc:.3f}",
            fontsize=10, fontweight="bold", color=PALETTE[cond],
        )
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "ROC Curves -- LOSO Cross-Validation (RF Classifier)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = PLOTS / "roc_curves_per_condition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# 3. PER-MOVEMENT RECALL & PRECISION

RISKY_MOVEMENTS = ["FAST_BEND", "LUMBAR_DOMINANT", "SHOULDER_DRIVEN", "PICKUP_ASYM", "FATIGUE_FLEXION"]

def plot_per_class_recall_precision():
    """Recall/precision per risky movement archetype, IMU vs IMU+sEMG, averaged
    across folds. Shows where (which movement) sEMG helps catch risk, not just
    the aggregate score."""
    pc = pd.read_csv(EVAL / "per_class_results.csv")
    pc = pc[pc["movement"].isin(RISKY_MOVEMENTS)]

    # aggregate across folds (mean per movement x condition)
    agg = (pc.groupby(["condition", "movement"])[["recall", "precision"]]
             .mean().reset_index())

    conditions = ["IMU", "IMU_EMG"]
    titles_cond = {"IMU": "RF -- IMU", "IMU_EMG": "RF -- IMU_EMG"}
    colours_cond = {"IMU": "#1565C0", "IMU_EMG": "#2E7D32"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, cond in zip(axes, conditions):
        sub = agg[agg["condition"] == cond].set_index("movement")
        sub = sub.reindex(RISKY_MOVEMENTS).fillna(0)

        x     = np.arange(len(RISKY_MOVEMENTS))
        width = 0.35

        bars_r = ax.bar(x - width/2, sub["recall"].values, width,
                        label="Recall", color=colours_cond[cond], alpha=0.85, edgecolor="white")
        bars_p = ax.bar(x + width/2, sub["precision"].values, width,
                        label="Precision", color=colours_cond[cond], alpha=0.35,
                        edgecolor=colours_cond[cond], linewidth=1.2)

        for bar in bars_r:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars_p:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in RISKY_MOVEMENTS], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
        ax.set_title(titles_cond[cond], fontsize=11, fontweight="bold",
                     color=colours_cond[cond])
        ax.set_ylabel("Score (mean across 5 LOSO folds)", fontsize=9)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Per-Movement Recall & Precision (RF, 5-fold LOSO)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = PLOTS / "per_class_recall_precision.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# 4. STATISTICAL SIGNIFICANCE (Wilcoxon, corrected)

def plot_statistical_significance():
    """Per-classifier effect size (mean delta) and exact Wilcoxon p for the
    IMU -> IMU+sEMG step. The 0.0625 threshold is the smallest two-sided p the
    exact test can reach at n=5 folds, so it is the honest significance bar."""
    classifiers = ["RF", "SVM", "LDA"]
    metrics     = ["f1_risk", "sensitivity"]
    metric_labels = {"f1_risk": "F1-Risk", "sensitivity": "Sensitivity"}

    # compute per-fold deltas and Wilcoxon p-values
    results = {}
    for clf in classifiers:
        imu     = loso[(loso["classifier"] == clf) & (loso["condition"] == "IMU")].sort_values("fold")
        imu_emg = loso[(loso["classifier"] == clf) & (loso["condition"] == "IMU_EMG")].sort_values("fold")
        if imu.empty or imu_emg.empty:
            continue
        row = {"clf": clf}
        for m in metrics:
            diffs = imu_emg[m].values - imu[m].values
            try:
                _, p = stats.wilcoxon(imu[m].values, imu_emg[m].values, alternative="two-sided")
            except ValueError:
                p = 1.0  # all zeros
            row[f"delta_{m}"]  = diffs.mean()
            row[f"diffs_{m}"]  = diffs
            row[f"p_{m}"]      = p
        results[clf] = row

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, metrics):
        clfs   = [c for c in classifiers if c in results]
        deltas = [results[c][f"delta_{metric}"] * 100 for c in clfs]  # percent
        pvals  = [results[c][f"p_{metric}"]          for c in clfs]
        diffs_all = [results[c][f"diffs_{metric}"] * 100 for c in clfs]

        colours = []
        for p, d in zip(pvals, deltas):
            if p <= 0.0625 and d > 0:
                colours.append("#4CAF50")   # significant positive
            elif p <= 0.0625 and d < 0:
                colours.append("#F44336")   # significant negative
            else:
                colours.append("#9E9E9E")   # not significant

        y_pos = np.arange(len(clfs))

        ax.barh(y_pos, deltas, color=colours, alpha=0.85, edgecolor="white", height=0.5)
        ax.axvline(0, color="black", linewidth=1.0)

        # per-fold dots
        for i, (dfs, d) in enumerate(zip(diffs_all, deltas)):
            ax.scatter(dfs, np.full_like(dfs, i), color="black",
                       s=25, zorder=5, alpha=0.7)

        # p-value annotations
        x_end = max(abs(max(deltas)), 0.5) * 1.1
        for i, (d, p) in enumerate(zip(deltas, pvals)):
            sig_str = f"p={p:.4f}" if p < 0.1 else f"p={p:.2f}"
            if p <= 0.0625:
                sig_str += " *"
            ax.text(
                x_end * (1.05 if d >= 0 else -1.05),
                i, sig_str,
                ha="left" if d >= 0 else "right",
                va="center", fontsize=8.5,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(clfs, fontsize=11)
        ax.set_xlabel(f"Delta {metric_labels[metric]} (%, IMU+sEMG minus IMU)", fontsize=9)
        ax.set_title(
            f"IMU --> IMU+sEMG: Delta{metric_labels[metric]}",
            fontsize=10, fontweight="bold",
        )

        # legend
        patches = [
            mpatches.Patch(color="#4CAF50", label="p <= 0.0625 positive"),
            mpatches.Patch(color="#F44336", label="p <= 0.0625 negative"),
            mpatches.Patch(color="#9E9E9E", label="p > 0.0625 (not significant)"),
        ]
        ax.legend(handles=patches, fontsize=7.5, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "IMU vs IMU+sEMG: Effect Size & Significance (Wilcoxon, n=5 folds)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = PLOTS / "statistical_significance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Generating extra evaluation plots...")
    plot_per_session_auc_fixed()
    plot_roc_scatter()
    plot_per_class_recall_precision()
    plot_statistical_significance()
    print("Done.")
