"""
Figure Generation for PHOENIX-v3.1 Research Paper

Generates publication-quality figures for:
- Architecture diagram
- Ablation study bar charts
- Performance comparison tables
- TTT analysis plots
- Efficiency scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import os


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


def plot_main_results(output_path: str = "figures/main_results.png"):
    """
    Generate main results comparison figure.
    
    Shows Dice scores and HD95 for all baseline models.
    """
    # Data
    models = ['nnU-Net v2', 'Swin-UNETR', 'U-Mamba', 'U-KAN', 'PHOENIX-v3.1']
    dice_wt = [91.8, 92.1, 91.5, 90.5, 93.2]
    dice_wt_std = [0.5, 0.4, 0.6, 0.7, 0.3]
    dice_et = [84.9, 85.4, 84.2, 83.1, 87.1]
    dice_et_std = [1.1, 0.9, 1.2, 1.3, 0.8]
    hd95 = [4.5, 4.2, 4.8, 5.2, 3.4]
    hd95_std = [0.3, 0.3, 0.4, 0.5, 0.2]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
    colors[-1] = '#E74C3C'  # Highlight PHOENIX
    
    x = np.arange(len(models))
    width = 0.6
    
    # Dice WT
    bars1 = axes[0].bar(x, dice_wt, width, yerr=dice_wt_std, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Dice Score (%)')
    axes[0].set_title('(a) Whole Tumor')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylim([88, 95])
    axes[0].axhline(y=93.2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Dice ET
    bars2 = axes[1].bar(x, dice_et, width, yerr=dice_et_std, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Dice Score (%)')
    axes[1].set_title('(b) Enhancing Tumor')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylim([80, 90])
    axes[1].axhline(y=87.1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # HD95
    bars3 = axes[2].bar(x, hd95, width, yerr=hd95_std, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('HD95 (mm)')
    axes[2].set_title('(c) Hausdorff Distance 95')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_ylim([2, 6])
    axes[2].axhline(y=3.4, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_ablation_study(output_path: str = "figures/ablation_study.png"):
    """
    Generate ablation study figure.
    
    Shows contribution of each component.
    """
    # Data
    configs = [
        'Full Model',
        '− SpatialMixer',
        '− Priority Scout',
        '− Liquid Mod.',
        '− KAN',
        '− MSCG',
        '− TTT',
        'Conv-only'
    ]
    dice_wt = [93.2, 91.8, 92.4, 92.6, 92.1, 92.8, 92.5, 90.2]
    delta = [0, -1.4, -0.8, -0.6, -1.1, -0.4, -0.7, -3.0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#2ECC71' if d == 0 else '#E74C3C' for d in delta]
    
    x = np.arange(len(configs))
    bars = ax.barh(x, dice_wt, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add delta labels
    for i, (bar, d) in enumerate(zip(bars, delta)):
        width = bar.get_width()
        label = f'{d:+.1f}' if d != 0 else 'Baseline'
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9,
                color='red' if d < 0 else 'green')
    
    ax.set_xlabel('Dice Score (WT) %')
    ax.set_ylabel('Configuration')
    ax.set_title('Ablation Study: Component Contributions')
    ax.set_yticks(x)
    ax.set_yticklabels(configs)
    ax.set_xlim([88, 95])
    ax.axvline(x=93.2, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Invert y-axis so full model is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_efficiency_comparison(output_path: str = "figures/efficiency.png"):
    """
    Generate efficiency comparison scatter plot.
    
    Shows Dice vs Parameters vs Inference time.
    """
    # Data
    models = ['nnU-Net v2', 'Swin-UNETR', 'U-Mamba', 'U-KAN', 'PHOENIX-v3.1']
    params = [31, 48, 30, 5, 1.2]  # Millions
    dice = [91.8, 92.1, 91.5, 90.5, 93.2]
    inference = [95, 120, 85, 45, 32]  # ms
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bubble size proportional to inference time
    sizes = [t * 3 for t in inference]
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#E74C3C']
    
    scatter = ax.scatter(params, dice, s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels
    for i, model in enumerate(models):
        offset = (5, 5) if model != 'PHOENIX-v3.1' else (-50, -20)
        ax.annotate(model, (params[i], dice[i]), 
                   xytext=offset, textcoords='offset points',
                   fontsize=9, fontweight='bold' if model == 'PHOENIX-v3.1' else 'normal')
    
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Dice Score (WT) %')
    ax.set_title('Efficiency vs Performance\n(bubble size = inference time)')
    ax.set_xlim([-5, 55])
    ax.set_ylim([89, 94])
    
    # Add legend for bubble size
    legend_sizes = [30, 60, 120]
    legend_labels = ['30ms', '60ms', '120ms']
    legend_bubbles = [plt.scatter([], [], s=s*3, c='gray', alpha=0.5) for s in legend_sizes]
    ax.legend(legend_bubbles, legend_labels, title='Inference Time', 
              loc='lower right', framealpha=0.9)
    
    # Highlight PHOENIX region
    rect = mpatches.Rectangle((0, 92.5), 3, 1.5, linewidth=2, 
                               edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(1.5, 94.2, 'Best\nRegion', ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_ttt_analysis(output_path: str = "figures/ttt_analysis.png"):
    """
    Generate TTT analysis figure.
    
    Shows performance improvement on hard cases.
    """
    # Data
    conditions = ['Easy\n(E<0.3)', 'Medium\n(0.3≤E<0.5)', 'Hard\n(E≥0.5)', 'Overall']
    without_ttt = [94.1, 91.8, 86.2, 92.5]
    with_ttt = [94.2, 93.1, 90.4, 93.2]
    improvement = [0.1, 1.3, 4.2, 0.7]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    x = np.arange(len(conditions))
    width = 0.35
    
    # Bar chart
    bars1 = axes[0].bar(x - width/2, without_ttt, width, label='Without TTT', 
                        color='#3498DB', edgecolor='black', linewidth=0.5)
    bars2 = axes[0].bar(x + width/2, with_ttt, width, label='With TTT',
                        color='#E74C3C', edgecolor='black', linewidth=0.5)
    
    axes[0].set_ylabel('Dice Score (WT) %')
    axes[0].set_title('(a) TTT Performance by Case Difficulty')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions)
    axes[0].set_ylim([84, 96])
    axes[0].legend(loc='lower left')
    
    # Add improvement annotations
    for i, imp in enumerate(improvement):
        height = max(without_ttt[i], with_ttt[i])
        axes[0].annotate(f'+{imp}', (x[i], height + 0.3), 
                        ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Improvement chart
    colors = ['#95E1D3', '#FCE38A', '#F38181', '#EAFFD0']
    bars3 = axes[1].bar(conditions, improvement, color=colors, 
                        edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Dice Improvement (%)')
    axes[1].set_title('(b) TTT Improvement by Case Difficulty')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    
    # Highlight hard cases
    bars3[2].set_edgecolor('red')
    bars3[2].set_linewidth(2)
    
    # Add value labels
    for bar, imp in zip(bars3, improvement):
        height = bar.get_height()
        axes[1].annotate(f'+{imp}%', (bar.get_x() + bar.get_width()/2, height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_architecture_diagram(output_path: str = "figures/architecture.png"):
    """
    Generate architecture diagram.
    
    Shows the hybrid pyramid structure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    conv_color = '#3498DB'
    ssm_color = '#E74C3C'
    special_color = '#2ECC71'
    
    def draw_block(x, y, w, h, label, color, sublabel=None):
        rect = mpatches.FancyBboxPatch((x, y), w, h, 
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black',
                                        linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.3, sublabel, ha='center', va='center',
                   fontsize=7, color='white')
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Input
    draw_block(0.5, 4, 2, 1.5, 'Input', '#95A5A6', '224×224×3')
    
    # MSCG
    draw_block(3, 4, 2, 1.5, 'MSCG', special_color, 'Spectral Gating')
    draw_arrow(2.5, 4.75, 3, 4.75)
    
    # Stem
    draw_block(5.5, 4, 2, 1.5, 'Stem', conv_color, '7×7 Conv')
    draw_arrow(5, 4.75, 5.5, 4.75)
    
    # Stage 1-2 (Conv)
    draw_block(8, 6, 2, 1.2, 'Stage 1', conv_color, 'ResConv 32')
    draw_block(8, 4.5, 2, 1.2, 'Stage 2', conv_color, 'ResConv 64')
    draw_arrow(7.5, 4.75, 8, 5.5)
    draw_arrow(8.5, 6, 8.5, 5.7)
    
    # Priority Scout
    draw_block(8, 3, 2, 1.2, 'Priority\nScout', special_color, 'ROI Detection')
    draw_arrow(8.5, 4.5, 8.5, 4.2)
    
    # Stage 3-4 (SSM)
    draw_block(8, 1.5, 2, 1.2, 'Stage 3', ssm_color, 'Liquid-S6-KAN')
    draw_block(8, 0, 2, 1.2, 'Stage 4', ssm_color, 'Liquid-S6-KAN')
    draw_arrow(8.5, 3, 8.5, 2.7)
    draw_arrow(8.5, 1.5, 8.5, 1.2)
    
    # SpatialMixer annotations
    ax.annotate('SpatialMixer', xy=(10.2, 2), fontsize=8, color='purple',
               style='italic')
    ax.annotate('SpatialMixer', xy=(10.2, 0.6), fontsize=8, color='purple',
               style='italic')
    
    # Output
    draw_block(5.5, 0, 2, 1.5, 'Output', '#95A5A6', 'Softmax(2)')
    draw_arrow(8, 0.6, 7.5, 0.6)
    
    # TTT annotation
    ax.annotate('TTT Adapter', xy=(5.5, 1.6), xytext=(3.5, 2.5),
               fontsize=9, color='red', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=conv_color, edgecolor='black', label='Conv2D (High Res)'),
        mpatches.Patch(facecolor=ssm_color, edgecolor='black', label='Liquid-S6-KAN (Low Res)'),
        mpatches.Patch(facecolor=special_color, edgecolor='black', label='Special Modules'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    ax.set_title('PHOENIX-v3.1 Hybrid Pyramid Architecture', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved: {output_path}")


def generate_all_figures(output_dir: str = "figures"):
    """Generate all figures for the research paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating research paper figures...")
    print("=" * 50)
    
    plot_main_results(os.path.join(output_dir, "main_results.png"))
    plot_ablation_study(os.path.join(output_dir, "ablation_study.png"))
    plot_efficiency_comparison(os.path.join(output_dir, "efficiency.png"))
    plot_ttt_analysis(os.path.join(output_dir, "ttt_analysis.png"))
    plot_architecture_diagram(os.path.join(output_dir, "architecture.png"))
    
    print("=" * 50)
    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_figures("paper/figures")
