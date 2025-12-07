#!/usr/bin/env python3
"""
Visualization Script for StreamingLLM Benchmark Results

This script generates comparison charts for StreamingLLM performance analysis.

Usage:
    python scripts/plot_streamingllm_results.py
    
Output:
    - results/streamingllm_comparison.png
    - results/streamingllm_tradeoff.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Benchmark data from streamingllm_result.txt
methods = ['baseline', 'streaming_256', 'streaming_512', 'streaming_1024']
cache_sizes = [2024, 256, 512, 1024]  # Actual cache limit

data = {
    'TTFT (s)': [0.0090, 0.0065, 0.0074, 0.0076],
    'TPOT (s)': [0.0059, 0.0074, 0.0071, 0.0074],
    'Throughput': [168.71, 135.37, 140.05, 136.03],
    'PPL': [39.99, 43.06, 41.45, 40.78],
    'Accuracy (%)': [35.49, 34.49, 35.04, 35.31],
}

# Color scheme
colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60']

# =============================================================================
# Figure 1: Multi-metric comparison bar chart
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('StreamingLLM Performance Comparison\n(Pythia-70M on PG-19)', 
             fontsize=14, fontweight='bold')

metrics = ['TTFT (s)', 'TPOT (s)', 'Throughput', 'PPL', 'Accuracy (%)']
ylabels = ['Time (seconds)', 'Time (seconds)', 'Tokens/sec', 'Perplexity', 'Accuracy (%)']
better_lower = [True, True, False, True, False]  # For annotation

for idx, (metric, ylabel, lower_better) in enumerate(zip(metrics, ylabels, better_lower)):
    ax = axes[idx // 3, idx % 3]
    
    values = data[metric]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', rotation=15, labelsize=8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}' if val < 1 else f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Highlight best value
    best_idx = np.argmin(values) if lower_better else np.argmax(values)
    bars[best_idx].set_edgecolor('#F39C12')
    bars[best_idx].set_linewidth(2)
    
    ax.grid(axis='y', alpha=0.3)

# Remove empty subplot
axes[1, 2].axis('off')

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='black') 
                   for c in colors]
fig.legend(legend_elements, methods, loc='lower right', ncol=4, 
           bbox_to_anchor=(0.95, 0.15), fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('results/streamingllm_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_comparison.png")

# =============================================================================
# Figure 2: Trade-off analysis (Cache Size vs PPL/Accuracy/TTFT)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('StreamingLLM: Cache Size Trade-off Analysis', 
             fontsize=13, fontweight='bold')

# Data for streaming methods only (exclude baseline for trend)
stream_methods = methods[1:]
stream_caches = cache_sizes[1:]
stream_ppl = data['PPL'][1:]
stream_acc = data['Accuracy (%)'][1:]
stream_ttft = data['TTFT (s)'][1:]

# Baseline reference lines
baseline_ppl = data['PPL'][0]
baseline_acc = data['Accuracy (%)'][0]
baseline_ttft = data['TTFT (s)'][0]

# Plot 1: PPL vs Cache Size
ax1 = axes[0]
ax1.plot(stream_caches, stream_ppl, 'o-', color='#E74C3C', linewidth=2, markersize=8, label='StreamingLLM')
ax1.axhline(y=baseline_ppl, color='#2C3E50', linestyle='--', linewidth=1.5, label=f'Baseline ({baseline_ppl:.1f})')
ax1.fill_between(stream_caches, baseline_ppl, stream_ppl, alpha=0.2, color='#E74C3C')
ax1.set_xlabel('Cache Size (tokens)', fontsize=10)
ax1.set_ylabel('Perplexity (↓ better)', fontsize=10)
ax1.set_title('Perplexity vs Cache Size', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(stream_caches)

# Add percentage labels
for x, y in zip(stream_caches, stream_ppl):
    pct = (y - baseline_ppl) / baseline_ppl * 100
    ax1.annotate(f'+{pct:.1f}%', (x, y), textcoords="offset points", 
                 xytext=(0, 8), ha='center', fontsize=8, color='#E74C3C')

# Plot 2: Accuracy vs Cache Size
ax2 = axes[1]
ax2.plot(stream_caches, stream_acc, 'o-', color='#27AE60', linewidth=2, markersize=8, label='StreamingLLM')
ax2.axhline(y=baseline_acc, color='#2C3E50', linestyle='--', linewidth=1.5, label=f'Baseline ({baseline_acc:.1f}%)')
ax2.fill_between(stream_caches, stream_acc, baseline_acc, alpha=0.2, color='#27AE60')
ax2.set_xlabel('Cache Size (tokens)', fontsize=10)
ax2.set_ylabel('Accuracy % (↑ better)', fontsize=10)
ax2.set_title('Accuracy vs Cache Size', fontsize=11, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(stream_caches)

# Add percentage labels
for x, y in zip(stream_caches, stream_acc):
    pct = (y - baseline_acc) / baseline_acc * 100
    ax2.annotate(f'{pct:.1f}%', (x, y), textcoords="offset points", 
                 xytext=(0, -12), ha='center', fontsize=8, color='#27AE60')

# Plot 3: TTFT vs Cache Size  
ax3 = axes[2]
ax3.plot(stream_caches, stream_ttft, 'o-', color='#3498DB', linewidth=2, markersize=8, label='StreamingLLM')
ax3.axhline(y=baseline_ttft, color='#2C3E50', linestyle='--', linewidth=1.5, label=f'Baseline ({baseline_ttft:.4f}s)')
ax3.fill_between(stream_caches, stream_ttft, baseline_ttft, alpha=0.2, color='#3498DB')
ax3.set_xlabel('Cache Size (tokens)', fontsize=10)
ax3.set_ylabel('TTFT (seconds, ↓ better)', fontsize=10)
ax3.set_title('TTFT vs Cache Size', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(stream_caches)

# Add percentage labels (improvement)
for x, y in zip(stream_caches, stream_ttft):
    pct = (baseline_ttft - y) / baseline_ttft * 100
    ax3.annotate(f'-{pct:.0f}%', (x, y), textcoords="offset points", 
                 xytext=(0, -12), ha='center', fontsize=8, color='#3498DB')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/streamingllm_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_tradeoff.png")

# =============================================================================
# Figure 3: Method Comparison (StreamingLLM vs other methods)
# =============================================================================
# Data from results.txt for comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Comparison data
comparison_methods = [
    'Baseline',
    'StreamingLLM\n(1024)',
    'StreamingLLM\n(512)',
    'Fix-Size L2\n(keep_low)',
    'Fix-Size L2\n(random)',
    'Sliding Window\n(recent_only)'
]

ppl_change = [0, 2.0, 3.7, 4.3, 7.9, 9.5]  # Percentage change from baseline
acc_change = [0, -0.5, -1.3, -1.6, -3.1, -3.6]

x = np.arange(len(comparison_methods))
width = 0.35

bars1 = ax.bar(x - width/2, ppl_change, width, label='PPL Change (%)', color='#E74C3C', edgecolor='black')
bars2 = ax.bar(x + width/2, acc_change, width, label='Accuracy Change (%)', color='#3498DB', edgecolor='black')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Method', fontsize=11)
ax.set_ylabel('Change from Baseline (%)', fontsize=11)
ax.set_title('KV Cache Compression Methods Comparison\n(Lower PPL change & Higher Acc change = Better)', 
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_methods, fontsize=9)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'+{height:.1f}%' if height > 0 else f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -10), textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -10 if height < 0 else 3), textcoords="offset points",
                ha='center', va='top' if height < 0 else 'bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/streamingllm_vs_others.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_vs_others.png")

print("\nAll figures generated successfully!")
print("\nKey insights:")
print("  1. StreamingLLM (1024) achieves best trade-off: PPL +2.0%, Acc -0.5%")
print("  2. TTFT improves by 15-28% across all StreamingLLM configs")
print("  3. StreamingLLM outperforms both L2-based and sliding window methods")

