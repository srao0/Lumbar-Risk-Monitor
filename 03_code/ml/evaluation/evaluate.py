#!/usr/bin/env python3
"""
Evaluation & Visualisation, Spinal Movement Risk Monitor
FYP 2025/26 | Imperial College London

Reads LOSO results from ml/evaluation/ and produces:
  1. AUC bar chart, per classifier × condition (IMU / EMG / IMU+sEMG)
  2. Sensitivity / Specificity bars, three-condition grouped
  3. Confusion matrices, normalised, summed across folds (3 × N grid)
  4. Feature importance, RF horizontal bar chart (top-N, per condition)
  5. Per-session AUC, line chart, all three conditions overlaid
  6. Delta table, printed and saved (IMU+sEMG vs IMU, vs EMG-only)

Usage:
  python ml/evaluation/evaluate.py
  python ml/evaluation/evaluate.py --top_n_features 8 --best_clf RF
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

# STYLE

# Four-condition palette:  IMU=blue, EMG=green, fusion=pink, FIS=orange
PALETTE = {
    "IMU":     "#2196F3",
    "EMG":     "#4CAF50",
    "IMU_EMG": "#E91E63",
    "FIS":     "#FF9800",
}

HATCHES = {
    "IMU":     "",
    "EMG":     "\\\\",
    "IMU_EMG": "//",
    "FIS":     "xx",
}

CONDITION_LABELS = {
    "IMU":     "IMU-only",
    "EMG":     "EMG-only",
    "IMU_EMG": "IMU+sEMG",
    "FIS":     "Mamdani FIS",
}

CLF_MARKERS = {
    "RF":  "o",
    "LR":  "D",
    "SVM": "s",
    "LDA": "^",
    "FIS": "P",   # plus-filled marker
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "figure.dpi":        150,
})

CONDITIONS_ORDER = ["IMU", "EMG", "IMU_EMG", "FIS"]


# HELPERS

def _get_conditions(df: pd.DataFrame) -> list:
    """Return ordered list of conditions present in the dataframe."""
    present = df["condition"].unique()
    return [c for c in CONDITIONS_ORDER if c in present]


# 1. AUC BAR CHART

def plot_roc_bars(df: pd.DataFrame, out_dir: Path):
    """
    Grouped bar chart: mean ROC-AUC ± std per classifier × condition.
    Each classifier group has up to 3 bars (IMU / EMG / IMU+sEMG).
    """
    conditions  = _get_conditions(df)
    classifiers = sorted(df["classifier"].unique())
    n_cond      = len(conditions)
    x           = np.arange(len(classifiers))
    width       = 0.7 / n_cond

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(classifiers)), 5))

    for i, cond in enumerate(conditions):
        sub    = df[df["condition"] == cond].set_index("classifier")
        means  = [sub.loc[c, "auc_mean"]  if c in sub.index else 0.0 for c in classifiers]
        stds   = [sub.loc[c, "auc_std"]   if c in sub.index else 0.0 for c in classifiers]
        offset = (i - (n_cond - 1) / 2) * width
        bars   = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4,
            color=PALETTE[cond], hatch=HATCHES[cond],
            label=CONDITION_LABELS[cond],
            alpha=0.85, edgecolor="white",
        )
        for bar, m in zip(bars, means):
            if m > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, fontsize=11)
    ax.set_ylabel("ROC-AUC (mean ± std, LOSO)", fontsize=10)
    ax.set_title("Classifier AUC — Three-Condition Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.08)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
    ax.legend(framealpha=0.9, fontsize=9)

    fig.tight_layout()
    path = out_dir / "roc_auc_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# 2. SENSITIVITY / SPECIFICITY

def plot_sens_spec(df: pd.DataFrame, out_dir: Path):
    """
    Two-panel grouped bar chart: sensitivity (left) and specificity (right),
    for each classifier × condition (three bars per group).
    """
    conditions  = _get_conditions(df)
    classifiers = sorted(df["classifier"].unique())
    n_cond      = len(conditions)
    x           = np.arange(len(classifiers))
    width       = 0.7 / n_cond
    metrics     = ["sensitivity", "specificity"]
    titles      = {"sensitivity": "Sensitivity (TPR)", "specificity": "Specificity (TNR)"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, metric in zip(axes, metrics):
        for i, cond in enumerate(conditions):
            sub    = df[df["condition"] == cond].set_index("classifier")
            means  = [sub.loc[c, f"{metric}_mean"] if c in sub.index else 0.0 for c in classifiers]
            stds   = [sub.loc[c, f"{metric}_std"]  if c in sub.index else 0.0 for c in classifiers]
            offset = (i - (n_cond - 1) / 2) * width
            bars   = ax.bar(
                x + offset, means, width,
                yerr=stds, capsize=4,
                color=PALETTE[cond], hatch=HATCHES[cond],
                label=CONDITION_LABELS[cond],
                alpha=0.85, edgecolor="white",
            )
            for bar, m in zip(bars, means):
                if m > 0.01:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=10)
        ax.set_title(titles[metric], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
        if metric == "sensitivity":
            ax.set_ylabel("Score (mean ± std, LOSO)", fontsize=9)
        ax.legend(framealpha=0.9, fontsize=8)

    fig.suptitle(
        "Sensitivity & Specificity — Three-Condition Comparison",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "sensitivity_specificity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# 3. CONFUSION MATRICES

def plot_confusion_matrices(fold_df: pd.DataFrame, out_dir: Path):
    """
    Grid of normalised confusion matrices: rows = classifiers, cols = conditions.
    Counts are summed across LOSO folds then row-normalised.
    """
    conditions  = _get_conditions(fold_df)
    classifiers = sorted(fold_df["classifier"].unique())
    nrows       = len(classifiers)
    ncols       = len(conditions)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for r, clf in enumerate(classifiers):
        for c, cond in enumerate(conditions):
            ax  = axes[r][c]
            sub = fold_df[(fold_df["classifier"] == clf) & (fold_df["condition"] == cond)]
            if sub.empty:
                ax.axis("off")
                continue

            tn = sub["tn"].sum(); fp = sub["fp"].sum()
            fn = sub["fn"].sum(); tp = sub["tp"].sum()
            cm      = np.array([[tn, fp], [fn, tp]])
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sum > 0, cm.astype(float) / row_sum, 0.0)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_norm,
                display_labels=["Safe (0)", "Risk (1)"],
            )
            disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
            ax.set_title(
                f"{clf} | {CONDITION_LABELS[cond]}",
                fontsize=8.5, fontweight="bold",
            )
            ax.set_xlabel("Predicted", fontsize=7.5)
            ax.set_ylabel("True" if c == 0 else "", fontsize=7.5)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Confusion Matrices (normalised, summed LOSO folds)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 4. FEATURE IMPORTANCE (RF)

def plot_feature_importance(fi_df: pd.DataFrame, out_dir: Path, top_n: int = 10):
    """
    Horizontal bar charts of RF feature importances, one panel per condition
    that has RF importance data (IMU, IMU_EMG typically; EMG uses LR).
    Features are colour-coded blue (IMU) vs green (EMG).
    """
    conditions = [c for c in CONDITIONS_ORDER if c in fi_df["condition"].unique()]
    n_panels   = len(conditions)
    if n_panels == 0:
        print("  No feature importance data found — skipping.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6.5))
    if n_panels == 1:
        axes = [axes]

    imu_colour = "#2196F3"
    emg_colour = "#4CAF50"

    for ax, cond in zip(axes, conditions):
        sub = fi_df[fi_df["condition"] == cond].nlargest(top_n, "importance_mean")
        sub = sub.sort_values("importance_mean", ascending=True)

        colours = [
            emg_colour if f.startswith("emg_") else imu_colour
            for f in sub["feature"]
        ]

        ax.barh(
            sub["feature"], sub["importance_mean"],
            xerr=sub["importance_std"], capsize=3,
            color=colours, edgecolor="white", alpha=0.85,
        )
        ax.set_xlabel("Mean decrease in impurity", fontsize=9)
        ax.set_title(
            f"RF Feature Importance — {CONDITION_LABELS[cond]}",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(labelsize=8)

        patches = [
            mpatches.Patch(color=imu_colour, label="IMU feature"),
            mpatches.Patch(color=emg_colour, label="EMG feature"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    path = out_dir / "feature_importance_RF.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 5. PER-SESSION AUC LINE CHART

def plot_per_session_auc(fold_df: pd.DataFrame, out_dir: Path, clf_name: str = "RF"):
    """
    Line chart of AUC per LOSO fold for the nominated classifier.
    All three conditions are overlaid to visualise cross-session variability
    and the added value of EMG.

    If clf_name is not available for a condition (e.g. EMG uses LR, not RF),
    the function falls back to the primary classifier for that condition.
    """
    conditions = _get_conditions(fold_df)
    fig, ax    = plt.subplots(figsize=(8, 4.5))

    # For EMG-only condition, fall back to LR if RF is absent
    clf_map = {}
    for cond in conditions:
        avail = fold_df[fold_df["condition"] == cond]["classifier"].unique()
        clf_map[cond] = clf_name if clf_name in avail else avail[0]

    for cond in conditions:
        clf_use = clf_map[cond]
        data    = fold_df[(fold_df["classifier"] == clf_use) & (fold_df["condition"] == cond)]
        if data.empty:
            continue
        data = data.sort_values("fold")
        ax.plot(
            data["fold"], data["auc"],
            marker=CLF_MARKERS.get(clf_use, "o"),
            color=PALETTE[cond], linewidth=2, markersize=7,
            label=f"{CONDITION_LABELS[cond]} ({clf_use})",
        )
        for _, row in data.iterrows():
            ax.annotate(
                f"{row['auc']:.3f}",
                (row["fold"], row["auc"]),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7, color=PALETTE[cond],
            )

    ax.set_xlabel("LOSO Fold (held-out session)", fontsize=10)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_title(
        f"Per-Session AUC — Three Conditions",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0.4, 1.1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance")

    # x-tick labels from IMU fold data
    imu_data = fold_df[
        (fold_df["condition"] == "IMU") &
        (fold_df["classifier"] == clf_map.get("IMU", fold_df["classifier"].iloc[0]))
    ].sort_values("fold")
    if not imu_data.empty:
        ax.set_xticks(imu_data["fold"].unique())
        ax.set_xticklabels(
            [f"Fold {int(row.fold)}\n({row.test_session})"
             for _, row in imu_data.iterrows()],
            fontsize=7.5,
        )

    ax.legend(framealpha=0.9, fontsize=8)
    fig.tight_layout()

    path = out_dir / f"per_session_auc_{clf_name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# 6. DELTA TABLE

def print_and_save_delta_table(summary_df: pd.DataFrame, out_dir: Path):
    """
    Show pairwise lift:
      (a) IMU+sEMG vs IMU-only, does EMG add value over kinematics?
      (b) IMU+sEMG vs EMG-only, does IMU add value over EMG alone?
    Positive Δ = first condition outperforms second.
    """
    metrics    = ["auc", "sensitivity", "specificity", "f1_risk", "accuracy"]
    conditions = _get_conditions(summary_df)
    pairs      = []
    if "IMU_EMG" in conditions and "IMU" in conditions:
        pairs.append(("IMU_EMG", "IMU",  "Δ(IMU+sEMG − IMU)"))
    if "IMU_EMG" in conditions and "EMG" in conditions:
        pairs.append(("IMU_EMG", "EMG",  "Δ(IMU+sEMG − EMG)"))
    if "IMU" in conditions and "EMG" in conditions:
        pairs.append(("IMU",     "EMG",  "Δ(IMU − EMG)"))

    all_rows = []
    for cond_a, cond_b, label in pairs:
        rows = []
        for clf in sorted(summary_df["classifier"].unique()):
            row = {"comparison": label, "classifier": clf}
            for m in metrics:
                a_val = summary_df.loc[
                    (summary_df["classifier"] == clf) & (summary_df["condition"] == cond_a),
                    f"{m}_mean"
                ]
                b_val = summary_df.loc[
                    (summary_df["classifier"] == clf) & (summary_df["condition"] == cond_b),
                    f"{m}_mean"
                ]
                if not a_val.empty and not b_val.empty:
                    row[f"Δ{m}"] = round(float(a_val.values[0]) - float(b_val.values[0]), 4)
                else:
                    row[f"Δ{m}"] = None
            rows.append(row)
        all_rows.extend(rows)

    delta_df   = pd.DataFrame(all_rows)
    delta_path = out_dir / "delta_conditions.csv"
    delta_df.to_csv(delta_path, index=False)

    print("\n" + "=" * 75)
    print("PAIRWISE LIFT TABLE  (positive = first condition outperforms second)")
    print("=" * 75)
    for label in delta_df["comparison"].unique():
        print(f"\n{label}")
        sub = delta_df[delta_df["comparison"] == label].drop(columns=["comparison"])
        print(sub.to_string(index=False, float_format="{:+.4f}".format))

    print(f"\nSaved: {delta_path}")
    return delta_df


# 7. FIS vs RF COMPARISON  (four-metric grouped bar)

def plot_fis_comparison(summary_df: pd.DataFrame, out_dir: Path):
    """
    Grouped bar chart comparing the Mamdani FIS against the two strongest
    RF baselines (IMU-only RF and IMU+sEMG RF) on four key metrics:
    AUC, sensitivity, specificity, and F1-risk.

    This chart is the primary visual for answering: 'How does the FIS
    compare to the best single-model classifiers?'
    """
    # Rows to compare: RF/IMU, RF/IMU_EMG, FIS/FIS
    comparators = [
        ("RF",  "IMU",     "IMU-only\n(RF)"),
        ("RF",  "IMU_EMG", "IMU+sEMG\n(RF)"),
        ("FIS", "FIS",     "Mamdani\nFIS"),
    ]
    metrics = ["auc", "sensitivity", "specificity", "f1_risk"]
    metric_labels = ["AUC", "Sensitivity", "Specificity", "F1-risk"]
    colours = [PALETTE["IMU"], PALETTE["IMU_EMG"], PALETTE["FIS"]]
    hatches = [HATCHES["IMU"], HATCHES["IMU_EMG"], HATCHES["FIS"]]

    n_metrics = len(metrics)
    n_models  = len(comparators)
    x = np.arange(n_metrics)
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (clf, cond, label) in enumerate(comparators):
        row = summary_df[(summary_df["classifier"] == clf) & (summary_df["condition"] == cond)]
        if row.empty:
            continue
        means = [float(row[f"{m}_mean"].values[0]) for m in metrics]
        stds  = [float(row[f"{m}_std"].values[0])  for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4,
            color=colours[i], hatch=hatches[i],
            label=label.replace("\n", " "),
            alpha=0.87, edgecolor="white",
        )
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (mean +/- std, LOSO)", fontsize=10)
    ax.set_title(
        "Mamdani FIS vs RF Baselines — Key Metrics",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0.5, 1.12)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
    ax.legend(framealpha=0.9, fontsize=9)

    fig.tight_layout()
    path = out_dir / "fis_vs_rf_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")



# MAIN

def main(top_n: int = 10, best_clf: str = "RF"):
    eval_dir  = Path("ml/evaluation")
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    loso_path    = eval_dir / "loso_results.csv"
    summary_path = eval_dir / "summary_results.csv"
    fi_path      = eval_dir / "feature_importance_RF.csv"

    for p in [loso_path, summary_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}.\nRun:  python ml/training/train_classifier.py first."
            )

    fold_df    = pd.read_csv(loso_path)
    summary_df = pd.read_csv(summary_path)

    if "feature_set" in fold_df.columns and "condition" not in fold_df.columns:
        fold_df    = fold_df.rename(columns={"feature_set": "condition"})
        summary_df = summary_df.rename(columns={"feature_set": "condition"})
        print("  Note: renamed legacy feature_set column to condition.")

    conditions = _get_conditions(fold_df)
    print(f"Loaded results -- {len(fold_df)} fold records | "
          f"{fold_df['classifier'].nunique()} classifiers | "
          f"{len(conditions)} conditions: {conditions}")

    print("\nGenerating plots...")
    plot_roc_bars(summary_df, plots_dir)
    plot_sens_spec(summary_df, plots_dir)
    plot_confusion_matrices(fold_df, plots_dir)
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        if "feature_set" in fi_df.columns and "condition" not in fi_df.columns:
            fi_df = fi_df.rename(columns={"feature_set": "condition"})
        plot_feature_importance(fi_df, plots_dir, top_n=top_n)
    else:
        print(f"  Skipping feature importance (no {fi_path})")
    plot_per_session_auc(fold_df, plots_dir, clf_name=best_clf)

    if "FIS" in fold_df["condition"].unique():
        plot_fis_comparison(summary_df, plots_dir)

    print_and_save_delta_table(summary_df, eval_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and plot LOSO results.")
    parser.add_argument("--top_n_features", type=int, default=10)
    parser.add_argument("--best_clf", default="RF")
    args = parser.parse_args()
    main(top_n=args.top_n_features, best_clf=args.best_clf)
