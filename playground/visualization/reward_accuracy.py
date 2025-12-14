import matplotlib.pyplot as plt

# Data
models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
params_b = [0.6, 1.7, 4.0, 8.0]  # Model size in billions of parameters
accuracy_helpful = [71.77, 71.99, 73.52, 74.01]
accuracy_harmless = [74.06, 76.69, 77.73, 78.41]

# Plot style
try:
    plt.style.use("seaborn-whitegrid")
except Exception:
    pass

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0
})

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot lines
ax.plot(params_b, accuracy_helpful, marker="o", color="tab:blue", label="Helpful", zorder=3)
ax.plot(params_b, accuracy_harmless, marker="s", color="tab:green", label="Harmless", zorder=3)

# Customize axes
ax.set_title("Reward Model Accuracy vs Base Model Size")
ax.set_xlabel("Model Size (Billions of Parameters)")
ax.set_ylabel("Accuracy (%)")

# Set x-ticks as formatted model sizes
ax.set_xticks(params_b)
ax.set_xticklabels([f"{p}B" for p in params_b])

# Y-axis limits for cleaner presentation
ax.set_ylim(70, 80)

# Grid
ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.5)
ax.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.3)
ax.minorticks_on()

# Legend
legend = ax.legend(loc="lower right", frameon=False)

# Annotate points with accuracy values
def annotate_points(x, y, color):
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    color=color,
                    fontsize=10)

annotate_points(params_b, accuracy_helpful, "tab:blue")
annotate_points(params_b, accuracy_harmless, "tab:green")

# Tight layout and export
plt.tight_layout()
# plt.savefig("reward_model_accuracy_vs_params.png", dpi=300)
plt.savefig("reward_model_accuracy_vs_params.pdf")
# plt.show()