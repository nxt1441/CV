import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 1. Training curves (all methods) ─────────────────────────────────────────
files = {
    "Teacher": "results/teacher_log.csv",
    "Baseline-A": "results/baseline_a_log.csv",
    "Baseline-B": "results/baseline_b_log.csv",
    "Hinton KD (T=4)": "results/hinton_log.csv",
    "Feature KD": "results/feature_log.csv",
    "Attention": "results/attention_log.csv",
    "RKD": "results/rkd_log.csv",
    "Combined KD": "results/combined_log.csv",
}

fig, ax = plt.subplots(figsize=(10, 5))
for label, path in files.items():
    try:
        epochs, vals = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                vals.append(float(row["val_top1"]))
        ax.plot(epochs, vals, label=label)
    except FileNotFoundError:
        pass
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Top-1 (%)")
ax.set_title("Validation Accuracy — All Methods")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.close()
print("Saved: results/training_curves.png")

# ── 2. Accuracy vs Model Size scatter ────────────────────────────────────────
methods = ["Teacher", "Baseline-A", "Baseline-B",
           "Hinton T=2", "Hinton T=4", "Hinton T=8",
           "Feature KD", "Attention", "RKD", "Combined KD"]
top1 = [83.22, 55.78, 76.30, 80.27, 80.88, 80.76, 77.30, 77.80, 79.05, 80.24]
params = [23.92, 2.48, 2.48, 2.48, 2.48, 2.48, 2.48, 2.48, 2.48, 2.48]
colors = ["red"] + ["gray"] + ["blue"] + ["green"] * 3 + ["orange"] * 2 + ["purple"] + ["green"]

fig, ax = plt.subplots(figsize=(8, 5))
for m, t, p, c in zip(methods, top1, params, colors):
    marker = "*" if m == "Teacher" else "o"
    size = 300 if m == "Teacher" else 100
    ax.scatter(p, t, c=c, marker=marker, s=size, zorder=3)
    ax.annotate(m, (p, t), textcoords="offset points", xytext=(6, 2), fontsize=7)
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("Val Top-1 (%)")
ax.set_title("Accuracy vs Model Size")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/accuracy_vs_size.png", dpi=150)
plt.close()
print("Saved: results/accuracy_vs_size.png")

# ── 3. Temperature ablation bar chart ────────────────────────────────────────
temps = [2, 4, 8]
scores = [80.27, 80.88, 80.76]

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar([str(t) for t in temps], scores, color=["#4c9be8", "#2563eb", "#1e40af"], width=0.5)
ax.set_ylim(79, 81.5)
ax.set_xlabel("Temperature (T)")
ax.set_ylabel("Val Top-1 (%)")
ax.set_title("Hinton KD — Temperature Sweep")
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{score:.2f}%", ha="center", fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/temperature_ablation.png", dpi=150)
plt.close()
print("Saved: results/temperature_ablation.png")

# ── 4. Final bar chart — all methods ─────────────────────────────────────────
labels = ["Teacher", "Base-A\n(scratch)", "Base-B\n(pretrain)",
          "Hinton\nT=2", "Hinton\nT=4", "Hinton\nT=8",
          "Feature\nKD", "Attention\nTransfer", "RKD", "Combined\nKD"]
values = [83.22, 55.78, 76.30, 80.27, 80.88, 80.76, 77.30, 77.80, 79.05, 80.24]
colors = ["#dc2626", "#9ca3af", "#6b7280",
          "#86efac", "#22c55e", "#16a34a",
          "#fdba74", "#f97316",
          "#a78bfa", "#7c3aed"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(83.22, color="red", linestyle="--", linewidth=1, label="Teacher (83.22%)")
ax.set_ylim(40, 88)
ax.set_ylabel("Val Top-1 (%)")
ax.set_title("Knowledge Distillation — All Methods Comparison")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", fontsize=8)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/all_methods_comparison.png", dpi=150)
plt.close()
print("Saved: results/all_methods_comparison.png")
