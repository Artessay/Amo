import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def read_jsonl_to_df(scores_dir):
    """
    读取目录下所有 JSONL 并转换为 Pandas DataFrame，方便 Seaborn 使用
    """
    scores_path = Path(scores_dir)
    jsonl_files = sorted(scores_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {scores_dir}")
        return pd.DataFrame()

    all_records = []

    for jsonl_file in jsonl_files:
        experiment_name = jsonl_file.stem
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    all_records.append({
                        "experiment": experiment_name,
                        "helpful": data.get('helpful_score', 0),
                        "harmless": data.get('harmless_score', 0)
                    })
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(all_records)

def plot_enhanced_scatter(df, output_dir):
    """
    使用 Seaborn 绘制优化后的散点图
    """
    # 设置 Seaborn 主题
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- 关键修改 1: 添加抖动 (Jitter) ---
    # 如果分数是离散的（如整数），直接画会重叠。
    # 这里我们手动创建抖动数据用于绘图，但保留原始数据用于统计
    # 抖动幅度设为 0.15 (根据你的数据范围调整，如果分数范围是0-100，可以设大一点)
    # jitter_strength = 0.15 
    # df['helpful_jitter'] = df['helpful'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    # df['harmless_jitter'] = df['harmless'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))

    # --- 关键修改 2: 使用 Seaborn Scatterplot ---
    # alpha: 透明度，越低越能看清重叠
    # s: 点的大小
    # hue: 根据实验名称自动上色
    sns.scatterplot(
        data=df,
        x='helpful', 
        y='harmless',
        # x='helpful_jitter', 
        # y='harmless_jitter',
        hue='experiment',
        style='experiment', # 不同实验使用不同形状的点，辅助区分
        alpha=0.4,          # 透明度，解决遮挡的关键
        s=20,               # 点大小
        # edgecolor='w',      # 给点加白边，区分度更高
        linewidth=0.5,
        ax=ax
    )

    # --- 可选方案：如果你想看密度而不是点，解开下面这行注释使用 KDE ---
    # sns.kdeplot(data=df, x='helpful', y='harmless', hue='experiment', levels=5, alpha=0.7, ax=ax)

    # 设置标题和标签
    # ax.set_title('Score Distribution: Helpful vs Harmless', pad=20)
    ax.set_xlabel('Helpful Score')
    ax.set_ylabel('Harmless Score')

    # 移动图例到图外，防止遮挡数据
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # 自动调整坐标轴范围 (加一点余量)
    ax.set_xlim(df['helpful'].min() - 0.5, df['helpful'].max() + 0.5)
    ax.set_ylim(df['harmless'].min() - 0.5, df['harmless'].max() + 0.5)

    plt.tight_layout()
    
    # 保存
    output_path = Path(output_dir)
    plt.savefig(output_path / "all_experiments_seaborn.pdf", bbox_inches='tight')
    plt.savefig(output_path / "all_experiments_seaborn.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir}")

def plot_facetted_view(df, output_dir):
    """
    方案二：分面图（每个实验一张小图，彻底解决遮挡）
    """
    g = sns.relplot(
        data=df,
        x='helpful', 
        y='harmless',
        col='experiment',      # 按实验分列
        col_wrap=3,            # 每行显示3个图
        hue='experiment',
        kind='scatter',
        alpha=0.6,
        s=50,
        height=4, 
        aspect=1
    )
    
    # 同样加上微小的抖动效果（Seaborn relplot 不支持直接 jitter 参数，通常需要在数据预处理做，或者接受重叠）
    # 这里演示的是原始数据视图
    
    output_path = Path(output_dir)
    plt.savefig(output_path / "all_experiments_facet.png", dpi=300, bbox_inches='tight')
    print("Facet plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="playground/visualization/scores")
    args = parser.parse_args()
    
    # 1. 读取数据为 DataFrame
    df = read_jsonl_to_df(args.dir)
    
    if not df.empty:
        print(f"Total samples: {len(df)}")
        print(f"Experiments found: {df['experiment'].unique()}")
        
        # 2. 绘制优化后的散点图 (带抖动)
        plot_enhanced_scatter(df, Path(args.dir).parent)
        
        # 3. (可选) 绘制分面图
        # plot_facetted_view(df, Path(args.dir).parent)
    else:
        print("No data found.")