import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

def read_jsonl_scores(jsonl_path):
    """
    Read helpful_score and harmless_score from JSONL file.
    
    Args:
        jsonl_path: Path to the input JSONL file
    
    Returns:
        tuple: (helpful_scores, harmless_scores)
    """
    helpful_scores = []
    harmless_scores = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            helpful_scores.append(data.get('helpful_score', 0))
            harmless_scores.append(data.get('harmless_score', 0))
    
    return helpful_scores, harmless_scores

def plot_multi_experiment_scatter(scores_dir):
    """
    Plot scatter plot of helpful_score vs harmless_score from all JSONL files in directory.
    
    Args:
        scores_dir: Path to the directory containing JSONL files
    """
    scores_path = Path(scores_dir)
    
    # Get all JSONL files in the directory
    jsonl_files = sorted(scores_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {scores_dir}")
        return
    
    # Color palette for different experiments
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # Read all data
    all_data = []
    all_helpful = []
    all_harmless = []
    
    for jsonl_file in jsonl_files:
        helpful_scores, harmless_scores = read_jsonl_scores(jsonl_file)
        experiment_name = jsonl_file.stem
        all_data.append({
            'name': experiment_name,
            'helpful': helpful_scores,
            'harmless': harmless_scores
        })
        all_helpful.extend(helpful_scores)
        all_harmless.extend(harmless_scores)
    
    # Plot style
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass
    
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.0
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot scatter points for each experiment
    for idx, data in enumerate(all_data):
        color = colors[idx % len(colors)]
        ax.scatter(data['helpful'], data['harmless'], 
                   alpha=0.6, s=50, color=color, 
                   edgecolors='tab:gray', linewidth=0.5,
                   label=data['name'])
    
    # Customize axes
    # ax.set_title('Score Distribution: Helpful vs Harmless\n(Multiple Experiments)')
    ax.set_xlabel('Helpful Score')
    ax.set_ylabel('Harmless Score')
    
    # Set axis limits based on all data
    min_helpful = min(all_helpful)
    max_helpful = max(all_helpful)
    min_harmless = min(all_harmless)
    max_harmless = max(all_harmless)
    
    ax.set_xlim(min_helpful - 0.2, max_helpful + 0.2)
    ax.set_ylim(min_harmless - 0.2, max_harmless + 0.2)
    
    # Grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.3)
    ax.minorticks_on()
    
    # Legend
    legend = ax.legend(loc="best", frameon=False, ncol=1)
    
    # # Add statistics for each experiment
    # stats_text = ""
    # for idx, data in enumerate(all_data):
    #     mean_helpful = sum(data['helpful']) / len(data['helpful'])
    #     mean_harmless = sum(data['harmless']) / len(data['harmless'])
    #     stats_text += f"{data['name']}:\n"
    #     stats_text += f"  Mean H: {mean_helpful:.2f}, Mean M: {mean_harmless:.2f}\n"
    #     stats_text += f"  Samples: {len(data['helpful'])}\n"
    
    # # Place statistics text outside the plot area
    # ax.text(1.02, 1.0, stats_text, transform=ax.transAxes, 
    #         verticalalignment='top', horizontalalignment='left',
    #         bbox=dict(boxstyle='round', alpha=0.1), fontsize=9)
    
    # Adjust layout to make room for statistics
    plt.tight_layout()
    # plt.subplots_adjust(right=0.75)
    
    # Save plots
    output_dir = scores_path.parent
    plt.savefig(output_dir / "all_experiments_scatter.pdf")
    plt.savefig(output_dir / "all_experiments_scatter.png", dpi=300)
    print(f"Plots saved as all_experiments_scatter.pdf and all_experiments_scatter.png")
    print(f"Total experiments: {len(all_data)}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Plot scatter plot of helpful vs harmless scores for all experiments")
    parser.add_argument("-d", "--dir", type=str, default="playground/visualization/scores", 
                        help="Path to the directory containing JSONL files (default: playground/visualization/scores)")
    
    # Parse arguments
    args = parser.parse_args()
    scores_dir = args.dir
    
    plot_multi_experiment_scatter(scores_dir)