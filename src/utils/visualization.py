import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'Serif'

labels = ['Original', 'FE', 'Original + DT', 'FE + DT']
# Helper
def get_metric(d, key):
    if key in d:
        return d[key]
    if f"val_{key}" in d:
        return d[f"val_{key}"]
    if f"test_{key}" in d:
        return d[f"test_{key}"]
    raise KeyError(f"Key '{key}' not found in {list(d.keys())}")

def plot_model_metrics(model_name,
                       val_metric_dicts,   # list 4 dict: [val, val_fe, val_dt, val_fe_dt]
                       test_metric_dicts,  # list 4 dict: [test, test_fe, test_dt, test_fe_dt]
                       metric='mae'):

    metric_name = metric.upper() if metric != 'r2' else 'R²'

    # Lấy điểm số
    val_scores = [get_metric(d, metric) for d in val_metric_dicts]
    test_scores = [get_metric(d, metric) for d in test_metric_dicts]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))

    rects1 = ax.bar(x - width/2, val_scores, width,
                    label='Validation', color='tab:blue',
                    edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, test_scores, width,
                    label='Test', color='tab:red',
                    edgecolor='black', linewidth=1.2)

    ax.set_ylabel(f'{metric_name} (°C)' if metric != 'r2' else metric_name)
    ax.set_title(f'{model_name} — {metric_name}', fontsize=16)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(ncol=2, loc='upper center')

    # Giới hạn trục y
    if metric != 'r2':
        ymax = max(val_scores + test_scores) * 1.15
        ax.set_ylim(0, ymax)
    else:
        ymin = min(val_scores + test_scores) - 0.05
        ymax = max(val_scores + test_scores) + 0.05
        ax.set_ylim(ymin, ymax)

    # Ghi số lên cột
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}' if metric == 'r2' else f'{h:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()