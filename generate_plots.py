"""
Generate publication-quality plots for README / portfolio.

Usage:
    python generate_plots.py                          # auto-detect outputs path from config
    python generate_plots.py --output-dir outputs2    # specify outputs path manually

Generates:
    assets/
    ├── sensitivity_by_patient.png   — Bar chart: sensitivity per patient
    ├── confusion_matrix.png         — Aggregated confusion matrix (all patients)
    ├── roc_curve.png                — ROC curve (aggregated + per-patient)
    ├── fa_vs_sensitivity.png        — Scatter: FA/24h vs Sensitivity trade-off
    └── class_balance.png            — Pie chart: preictal vs interictal
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
sns.set_style("whitegrid")

ASSETS_DIR = Path(__file__).parent / "assets"


# ── Helpers ──────────────────────────────────────────────────────────────
def load_patient_data(tl_dir: Path):
    """Load metrics and windows for every patient in transfer_learning/."""
    patients = {}
    for patient_dir in sorted(tl_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        metrics_path = patient_dir / "metrics_dl.json"
        windows_path = patient_dir / "windows_dl.csv"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        windows = pd.read_csv(windows_path) if windows_path.exists() else None
        patients[pid] = {"metrics": metrics, "windows": windows}
    return patients


def load_results_json(tl_dir: Path):
    """Load results_transfer_learning.json for train/test split info."""
    path = tl_dir.parent / "results_transfer_learning.json"
    if not path.exists():
        path = tl_dir / "results_transfer_learning.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def resolve_transfer_learning_dir(output_dir: Path) -> Path:
    """
    Resolve transfer-learning results directory from different layouts.

    Supported layouts:
    1) output_dir/transfer_learning/chbXX/... (new)
    2) output_dir/chbXX/... (legacy, e.g. outputs2/)
    """
    nested = output_dir / "transfer_learning"
    if nested.exists():
        return nested

    has_patient_dirs = any(
        p.is_dir() and p.name.startswith("chb")
        for p in output_dir.iterdir()
    )
    has_results_file = (output_dir / "results_transfer_learning.json").exists()
    if has_patient_dirs or has_results_file:
        return output_dir

    return nested


# ── Plot 1: Sensitivity by patient ──────────────────────────────────────
def plot_sensitivity(patients, results_json):
    """Horizontal bar chart — sensitivity per patient, colored by train/test."""
    pids = sorted(patients.keys())
    sensitivities = [patients[p]["metrics"]["sensitivity"] * 100 for p in pids]

    # Determine train/test split
    test_patients = set()
    if results_json:
        for pid, res in results_json.items():
            if res.get("is_test"):
                test_patients.add(pid)

    colors = ["#e74c3c" if p in test_patients else "#3498db" for p in pids]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(pids, sensitivities, color=colors, edgecolor="white", height=0.7)

    # Value labels
    for bar, val in zip(bars, sensitivities):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=10)

    ax.set_xlim(0, 115)
    ax.set_xlabel("Sensitivity (%)")
    ax.set_title("Seizure Detection Sensitivity by Patient")
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, label="80% target")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(color="#3498db", label="Train"),
                    Patch(color="#e74c3c", label="Test (unseen)")]
    ax.legend(handles=legend_items, loc="lower right")

    fig.savefig(ASSETS_DIR / "sensitivity_by_patient.png")
    plt.close(fig)
    print("[+] sensitivity_by_patient.png")


# ── Plot 2: Aggregated confusion matrix ─────────────────────────────────
def plot_confusion_matrix(patients):
    """Confusion matrix aggregated across all patients."""
    all_true, all_pred = [], []
    for pid, data in patients.items():
        w = data["windows"]
        if w is None or "label" not in w.columns or "pred_label" not in w.columns:
            continue
        valid = w[w["label"].isin([0, 1])]
        all_true.extend(valid["label"].values)
        all_pred.extend(valid["pred_label"].values)

    if not all_true:
        print("[!] No window data — skipping confusion_matrix.png")
        return

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    # Normalize by row (true label) for percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Interictal", "Preictal"],
        yticklabels=["Interictal", "Preictal"],
        cbar_kws={"label": "Count"},
        linewidths=0.5
    )
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Aggregated Confusion Matrix (all patients)")

    fig.savefig(ASSETS_DIR / "confusion_matrix.png")
    plt.close(fig)
    print("[+] confusion_matrix.png")


# ── Plot 3: ROC curve ───────────────────────────────────────────────────
def plot_roc_curve(patients):
    """ROC curve: aggregated + individual per-patient curves."""
    fig, ax = plt.subplots(figsize=(8, 7))

    all_true, all_probs = [], []

    for pid in sorted(patients.keys()):
        w = patients[pid]["windows"]
        if w is None or "prob_preictal" not in w.columns:
            continue
        valid = w[w["label"].isin([0, 1])]
        y = valid["label"].values
        p = valid["prob_preictal"].values

        if len(np.unique(y)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y, p)
        pat_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.25, linewidth=1,
                label=f"{pid} (AUC={pat_auc:.2f})" if pat_auc >= 0.8 else None)

        all_true.extend(y)
        all_probs.extend(p)

    if all_true:
        fpr, tpr, _ = roc_curve(all_true, all_probs)
        overall_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#e74c3c", linewidth=2.5,
                label=f"Aggregated (AUC={overall_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Seizure Prediction")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.savefig(ASSETS_DIR / "roc_curve.png")
    plt.close(fig)
    print("[+] roc_curve.png")


# ── Plot 4: FA/24h vs Sensitivity scatter ───────────────────────────────
def plot_fa_vs_sensitivity(patients, results_json):
    """Scatter plot: trade-off between false alarms and sensitivity."""
    test_patients = set()
    if results_json:
        for pid, res in results_json.items():
            if res.get("is_test"):
                test_patients.add(pid)

    fig, ax = plt.subplots(figsize=(9, 7))

    for pid, data in sorted(patients.items()):
        m = data["metrics"]
        is_test = pid in test_patients
        color = "#e74c3c" if is_test else "#3498db"
        marker = "^" if is_test else "o"
        ax.scatter(m["fa_per_24h"], m["sensitivity"] * 100,
                   c=color, marker=marker, s=80, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(pid.replace("chb", ""), (m["fa_per_24h"], m["sensitivity"] * 100),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.4, label="80% sensitivity target")
    ax.set_xlabel("False Alarms per 24h")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_title("Sensitivity vs False Alarm Rate")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="Train"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c", markersize=8, label="Test"),
    ]
    ax.legend(handles=legend_items, loc="lower right")

    fig.savefig(ASSETS_DIR / "fa_vs_sensitivity.png")
    plt.close(fig)
    print("[+] fa_vs_sensitivity.png")


# ── Plot 5: Class balance ───────────────────────────────────────────────
def plot_class_balance(patients):
    """Pie chart of preictal vs interictal windows."""
    total_preictal, total_interictal = 0, 0
    for data in patients.values():
        w = data["windows"]
        if w is None:
            continue
        total_interictal += (w["label"] == 0).sum()
        total_preictal += (w["label"] == 1).sum()

    if total_preictal + total_interictal == 0:
        print("[!] No window data — skipping class_balance.png")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sizes = [total_interictal, total_preictal]
    labels = [
        f"Interictal\n{total_interictal:,} ({total_interictal / sum(sizes) * 100:.1f}%)",
        f"Preictal\n{total_preictal:,} ({total_preictal / sum(sizes) * 100:.1f}%)",
    ]
    colors = ["#3498db", "#e74c3c"]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90,
           textprops={"fontsize": 12}, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Class Distribution (all patients)")

    fig.savefig(ASSETS_DIR / "class_balance.png")
    plt.close(fig)
    print("[+] class_balance.png")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate plots for README")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to outputs/ directory (auto-detected from config if omitted)")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        import yaml
        config_path = Path(__file__).parent / "config" / "default.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            output_dir = Path(cfg["paths"]["output_dir"])
        else:
            print("ERROR: No --output-dir and config/default.yaml not found")
            sys.exit(1)

    tl_dir = resolve_transfer_learning_dir(output_dir)
    if not tl_dir.exists():
        print(f"ERROR: {tl_dir} not found. Run the training pipeline first:")
        print("  python run_transfer.py --config config/default.yaml")
        sys.exit(1)

    # Load data
    print(f"Loading data from {tl_dir} ...")
    patients = load_patient_data(tl_dir)
    results_json = load_results_json(tl_dir)

    if not patients:
        print("ERROR: No patient data found")
        sys.exit(1)

    print(f"Found {len(patients)} patients\n")

    # Create assets dir
    ASSETS_DIR.mkdir(exist_ok=True)

    # Generate all plots
    plot_sensitivity(patients, results_json)
    plot_confusion_matrix(patients)
    plot_roc_curve(patients)
    plot_fa_vs_sensitivity(patients, results_json)
    plot_class_balance(patients)

    print(f"\nAll plots saved to {ASSETS_DIR.resolve()}/")


if __name__ == "__main__":
    main()
