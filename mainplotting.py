import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Directory containing experiment logs
base_dir = "ploter"
results_by_experiment = {}

# Find all evaluation result .npy files
search_pattern = os.path.join(base_dir, "*", "evaluation", "results_*.npy")
result_files = sorted(glob.glob(search_pattern))

print(f"{len(result_files)} result files found.\n")

# Load and organize data by experiment
for filepath in result_files:
    parts = filepath.split(os.sep)
    experiment_name = parts[-3]  # e.g., Riverraid-Atari2600_20250719_163933

    data = np.load(filepath, allow_pickle=True).item()
    #print(f"{filepath}: type={type(data)}, content={data}")
    #print("######################")

    if experiment_name not in results_by_experiment:
        results_by_experiment[experiment_name] = []

    results_by_experiment[experiment_name].append(data)

# Plotting setup
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
})

plt.figure(figsize=(12, 3))


for experiment, evaluations in results_by_experiment.items():
    combined = {}


    for eval_dict in evaluations:
        for checkpoint, stats in eval_dict.items():
            if checkpoint == "final" or stats[0] is None:
                continue
            if checkpoint not in combined:
                combined[checkpoint] = []
            combined[checkpoint].append(stats)

    # Average across multiple files (if repeated checkpoints)
    x_vals = []
    mean_scores = []
    std_scores = []

    for checkpoint in sorted(combined.keys(), key=lambda x: int(x.split("_")[1])):
        step = int(checkpoint.split("_")[1])

        stats = np.array(combined[checkpoint])  # shape (N, 2)
        mean = stats[:, 0].mean()
        std = stats[:, 1].mean()

        x_vals.append(step)
        mean_scores.append(mean)
        std_scores.append(std)

    if not x_vals:
        print(f"Skipping {experiment} (no valid checkpoints).")
        continue

    x_vals = np.array(x_vals)
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)

    # Normalize steps (optional, display in units of 100k steps)
    x_vals_normalized = x_vals / 100000

    plt.plot(x_vals_normalized, mean_scores, marker="o", label=experiment)
    plt.errorbar(x_vals_normalized, mean_scores, yerr=std_scores,
                 fmt="none", ecolor="gray", alpha=0.3, capsize=5, capthick=1.5)

# Final plot configuration
plt.xlabel(r"Steps [$\times 10^5$]", fontsize=14)
plt.ylabel("Mean score", fontsize=14)
#plt.title("Combined Evaluation Results", fontsize=16)
plt.grid(alpha=0.3)
#plt.legend(fontsize=10)
plt.tight_layout()

# Save plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join("output", f"combined_evaluation_plot_{timestamp}.pdf")
os.makedirs("output", exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")
