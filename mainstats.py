import os
import numpy as np

base_dir = 'logs'

for subdir in os.listdir(base_dir):
    eval_dir = os.path.join(base_dir, subdir, 'evaluation')
    if not os.path.isdir(eval_dir):
        continue

    npy_files = [f for f in os.listdir(eval_dir) if f.endswith('.npy')]
    if not npy_files:
        continue

    npy_path = os.path.join(eval_dir, npy_files[0])
    data = np.load(npy_path, allow_pickle=True).item()

    values = []
    for key, value in data.items():
        if not key.startswith('checkpoint_'):
            continue
        mean, std = value
        mean = float(mean)
        std = float(std)
        checkpoint = int(key.replace('checkpoint_', ''))
        lower = mean - std
        upper = mean + std
        efficiency = mean / checkpoint * 1000 if checkpoint != 0 else 0
        values.append((checkpoint, mean, std, lower, upper, efficiency))

    values.sort(key=lambda x: x[0])

    lines = []
    for checkpoint, mean, std, lower, upper, efficiency in values:
        line = f'{checkpoint}; {mean:.2f}±{std:.2f}; min: {lower:.2f}, max: {upper:.2f}; efficiency: {efficiency:.6f}'
        lines.append(line)

    best = max(values, key=lambda x: x[3])
    best_line = f'\nMax mean-std: {best[3]:.2f} at checkpoint_{best[0]}'

    checkpoints = [v[0] for v in values]
    means = [v[1] for v in values]
    auc = np.trapezoid(means, checkpoints) / checkpoints[-1]
    auc_line = f'Normalized AUC (robustness): {auc:.2f}'

    if len(means) > 1:
        mean_diffs = [abs(means[i+1] - means[i]) for i in range(len(means) - 1)]
        mean_diff_avg = sum(mean_diffs) / len(mean_diffs)
        diff_line = f'Mean delta (mean between steps): {mean_diff_avg:.2f}'
    else:
        diff_line = 'Mean delta (mean between steps): N/A'

    output_path = os.path.join(eval_dir, f'mean_minus_std_{subdir}.txt')
    print(output_path)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n' + best_line + '\n' + auc_line + '\n' + diff_line)
