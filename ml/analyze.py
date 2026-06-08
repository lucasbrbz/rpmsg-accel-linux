#!/usr/bin/env python3
"""
Compare performance metrics from exp-a (CPU baseline) and exp-b (NPU+M7).

Usage:
    python3 ml/analyze.py \
        --expa exp-a-results/features_data.csv \
        --expb exp-b-results/features_data.csv \
        --out  metrics_comparison.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ['t_feat_ms', 't_infer_ms', 't_window_ms', 'window_interval_ms',
           'cpu_percent', 'mem_rss_kb']

LABELS = {
    't_feat_ms':          'Feature extraction (ms)',
    't_infer_ms':         'Inference latency (ms)',
    't_window_ms':        'Total A53 processing (ms)',
    'window_interval_ms': 'Window interval (ms)',
    'cpu_percent':        'CPU utilization (%)',
    'mem_rss_kb':         'Memory RSS (KB)',
}


def summary(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print(f'{"Metric":<28}  {"exp-a mean":>12}  {"exp-a std":>10}  '
          f'{"exp-b mean":>12}  {"exp-b std":>10}  {"speedup":>8}')
    print('-' * 90)
    for col in METRICS:
        if col not in a.columns or col not in b.columns:
            continue
        am, as_ = a[col].mean(), a[col].std()
        bm, bs  = b[col].mean(), b[col].std()
        speedup = f'{am/bm:.2f}x' if bm > 0 else 'N/A'
        print(f'{LABELS[col]:<28}  {am:>12.3f}  {as_:>10.3f}  '
              f'{bm:>12.3f}  {bs:>10.3f}  {speedup:>8}')

    print()
    for exp, df in [('exp-a', a), ('exp-b', b)]:
        if 'label' in df.columns and 'prediction' in df.columns:
            acc = (df['label'] == df['prediction']).mean()
            print(f'{exp} inference accuracy: {acc:.1%}  ({(df["label"]==df["prediction"]).sum()}/{len(df)})')


def plot(a: pd.DataFrame, b: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle('Experiment A (CPU baseline) vs Experiment B (M7+NPU)', fontsize=14, y=0.98)

    # Inference latency — histogram
    ax = axes[0, 0]
    ax.hist(a['t_infer_ms'], bins=30, alpha=0.6, label='exp-a (CPU)', color='steelblue')
    ax.hist(b['t_infer_ms'], bins=30, alpha=0.6, label='exp-b (NPU)', color='darkorange')
    ax.set(title='Inference latency distribution', xlabel='ms', ylabel='windows')
    ax.legend()
    ax.axvline(a['t_infer_ms'].mean(), color='steelblue', linestyle='--', linewidth=1)
    ax.axvline(b['t_infer_ms'].mean(), color='darkorange', linestyle='--', linewidth=1)

    # Total A53 processing time — histogram
    ax = axes[0, 1]
    ax.hist(a['t_window_ms'], bins=30, alpha=0.6, label='exp-a', color='steelblue')
    ax.hist(b['t_window_ms'], bins=30, alpha=0.6, label='exp-b', color='darkorange')
    ax.set(title='Total A53 processing time per window', xlabel='ms', ylabel='windows')
    ax.legend()

    # Stacked bar — mean cost breakdown per experiment
    ax = axes[1, 0]
    exps       = ['exp-a\n(CPU)', 'exp-b\n(M7+NPU)']
    feat_means = [a['t_feat_ms'].mean(), b['t_feat_ms'].mean()]
    infer_means = [a['t_infer_ms'].mean(), b['t_infer_ms'].mean()]
    x = np.arange(len(exps))
    ax.bar(x, feat_means,  label='Feature extraction (A53)', color='royalblue')
    ax.bar(x, infer_means, bottom=feat_means, label='Inference', color='tomato')
    ax.set(title='Mean A53 processing cost breakdown', ylabel='ms', xticks=x)
    ax.set_xticklabels(exps)
    ax.legend()
    for i, (f, r) in enumerate(zip(feat_means, infer_means)):
        ax.text(i, f + r + 0.1, f'{f+r:.2f}ms', ha='center', fontsize=9)

    # CPU utilization over time
    ax = axes[1, 1]
    ax.plot(a['cpu_percent'].values, alpha=0.75, label='exp-a (CPU)', color='steelblue')
    ax.plot(b['cpu_percent'].values, alpha=0.75, label='exp-b (NPU)', color='darkorange')
    ax.axhline(a['cpu_percent'].mean(), color='steelblue', linestyle='--', linewidth=1)
    ax.axhline(b['cpu_percent'].mean(), color='darkorange', linestyle='--', linewidth=1)
    ax.set(title='A53 CPU utilization over time', xlabel='window index', ylabel='%')
    ax.legend()

    # Window interval (jitter) over time
    ax = axes[2, 0]
    ax.plot(a['window_interval_ms'].values, alpha=0.75, label='exp-a', color='steelblue')
    ax.plot(b['window_interval_ms'].values, alpha=0.75, label='exp-b', color='darkorange')
    ax.set(title='Window interval / jitter over time', xlabel='window index', ylabel='ms')
    ax.legend()
    for exp, df, color in [('exp-a', a, 'steelblue'), ('exp-b', b, 'darkorange')]:
        std = df['window_interval_ms'].std()
        ax.text(0.98, 0.95 if exp == 'exp-a' else 0.85,
                f'{exp} σ={std:.1f}ms', transform=ax.transAxes,
                ha='right', fontsize=9, color=color)

    # Memory RSS over time
    ax = axes[2, 1]
    ax.plot(a['mem_rss_kb'].values, alpha=0.75, label='exp-a', color='steelblue')
    ax.plot(b['mem_rss_kb'].values, alpha=0.75, label='exp-b', color='darkorange')
    ax.set(title='Process memory RSS over time', xlabel='window index', ylabel='KB')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {out}')


def main():
    parser = argparse.ArgumentParser(description='Compare exp-a vs exp-b metrics')
    parser.add_argument('--expa', required=True, help='exp-a features_data.csv path')
    parser.add_argument('--expb', required=True, help='exp-b features_data.csv path')
    parser.add_argument('--out',  default='metrics_comparison.png',
                        help='Output plot file (default: metrics_comparison.png)')
    args = parser.parse_args()

    try:
        a = pd.read_csv(args.expa)
        b = pd.read_csv(args.expb)
    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    missing = [c for c in METRICS if c not in a.columns or c not in b.columns]
    if missing:
        print(f'Warning: metrics columns missing from one or both CSVs: {missing}',
              file=sys.stderr)

    print(f'Loaded {len(a)} exp-a windows, {len(b)} exp-b windows\n')
    summary(a, b)
    plot(a, b, args.out)


if __name__ == '__main__':
    main()
