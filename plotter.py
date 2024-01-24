import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.stats as st

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)                        


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def smooth(scalar, weight=0.6):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def gen_ydata(ys, min_len, weight):
    ymin_len = min([len(y) for y in ys])
    min_len = min(min_len, ymin_len)
    y_matrix = np.vstack([y[:min_len] for y in ys])
    y_mean, low_CI_bound, high_CI_bound = mean_confidence_interval(y_matrix)
    y_min = np.min(y_matrix, axis=0)
    y_max= np.max(y_matrix, axis=0)
    y_low = np.maximum(y_min, low_CI_bound)
    y_high = np.minimum(y_max, high_CI_bound)
    return smooth(y_mean, weight), smooth(y_low, weight), smooth(y_high, weight)

def plot_distribution(xs, ys, ax, set_x_label=False, set_y_label=False, weight=0.6):
    ax.grid(c='w')
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.2)
    ax.patch.set_width(1)
    ax.patch.set_height(1)
    # ax.set(facecolor = "whitesmoke")
    ax.spines['bottom'].set_linewidth('0')
    ax.spines['top'].set_linewidth('0')
    ax.spines['right'].set_linewidth('0')
    ax.spines['left'].set_linewidth('0')
    ax.set_title(f"{game_name}", size=10)
    if set_x_label:
        ax.set_xlabel('Epoch')
    if set_y_label:
        ax.set_ylabel('Success Rate')
    x = xs[0]
    y, y_low, y_high = gen_ydata(ys, 50, weight)
    ax.plot(x, y, label="VCP", linewidth=1)
    ax.fill_between(x, y_low, y_high, alpha=0.2)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlim([0, 50])
    ax.set_aspect(0.7/ax.get_data_ratio()) # , adjustable='box')
    if game_name == 'HandManipulateBlockFull-v0' or game_name == 'HandManipulateBlockRotateParallel-v0':
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        if game_name == 'HandManipulateBlockFull-v0':
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.003))
        if game_name == 'HandManipulateBlockRotateParallel-v0':
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.015)) 
    else:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        if game_name == 'FetchSlide-v1': 
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.10)) 
        if game_name == 'HandReach-v0': 
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.10)) 
        if game_name == 'HandManipulateEggFull-v0': 
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01)) 
        if game_name == 'HandManipulateEggRotate-v0': 
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.15)) 
        if game_name == 'HandManipulatePenFull-v0':
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.03))
        if game_name == 'HandManipulatePenRotate-v0':
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
        if game_name == 'HandManipulateBlockRotateXYZ-v0':
            ax.yaxis.set_major_locator(mtick.MultipleLocator(0.04))


path = os.path.abspath(os.path.dirname(__file__))
game_names = [
    'PointMassEmptyEnv-v1', 'PointMassWallEnv-v1', 'Reacher-v2', 'FetchReach-v1',
    'FetchPush-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1', 'HandReach-v0', 
    'HandManipulatePenRotate-v0', 'HandManipulateEggRotate-v0', 'HandManipulatePenFull-v0', 'HandManipulateEggFull-v0',
    'HandManipulateBlockFull-v0', 'HandManipulateBlockRotateZ-v0', 'HandManipulateBlockRotateXYZ-v0', 'HandManipulateBlockRotateParallel-v0', 
]
n_row, n_col = 4, 4
fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 2 * n_row + 0.6))  # 3.6
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.95, hspace=0.3, wspace=0.22)
for i, game_name in enumerate(game_names):
    row = i // 4
    col = i % 4
    epoch_datas = []
    success_rate_datas = []
    log_path = f"{path}/raw_data/{game_name}"
    for root, dirs, files in os.walk(log_path):
        if len(files) != 0 and 'progress.csv' in files:
            logs = pd.read_csv(os.path.join(root, 'progress.csv'), sep="\t")
            x_name = "epoch/num"
            y_name = "test/success_rate"
            epoch_datas.append(logs[x_name].to_numpy())
            success_rate_datas.append(logs[y_name].to_numpy())

    weight = 0.6 if game_name in ['FetchReach-v1', 'Reacher-v2'] else 0.9
    plot_distribution(epoch_datas, success_rate_datas, ax=axes[row][col], set_x_label=row==3, set_y_label=col==0, weight=weight)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), borderaxespad=0., ncol=5, fontsize=10)
plt.savefig(f"{path}/baseline")
plt.savefig(f"{path}/baseline.pdf")
print(f"save to {path}/baseline.png")
