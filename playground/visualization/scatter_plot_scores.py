import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse

def read_jsonl_to_df(scores_dir):
    """
    读取目录下所有 JSONL 并转换为 Pandas DataFrame
    """
    scores_path = Path(scores_dir)
    # 兼容处理：如果没有找到目录，防止报错
    if not scores_path.exists():
        print(f"Directory not found: {scores_dir}")
        return pd.DataFrame()
        
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

    # alpha: 透明度，越低越能看清重叠
    # s: 点的大小
    # hue: 根据实验名称自动上色
    sns.scatterplot(
    data=df,
    x='helpful', 
    y='harmless',
    hue='experiment',
    style='experiment', # 不同实验使用不同形状的点，辅助区分
    alpha=0.4, # 透明度，解决遮挡的关键
    s=20, # 点大小
    linewidth=0.5,
    ax=ax
    )

    # 设置标题和标签
    ax.set_xlabel('Helpful Score')
    ax.set_ylabel('Harmless Score')

    # # 自动调整坐标轴范围 (加一点余量)
    # ax.set_xlim(df['helpful'].min() - 0.5, df['helpful'].max() + 0.5)
    # ax.set_ylim(df['harmless'].min() - 0.5, df['harmless'].max() + 0.5)

    plt.tight_layout()

    # 保存
    output_path = Path(output_dir)
    plt.savefig(output_path / "all_experiments_seaborn.pdf", bbox_inches='tight')
    plt.savefig(output_path / "all_experiments_seaborn.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir}")

def get_pareto_frontier(group):
    """
    计算 Pareto 前沿的辅助函数 (假设两个维度都是越大越好)。
    逻辑：
    1. 按照 X 轴 (helpful) 降序排列。
    2. 遍历，记录当前遇到的最大 Y 轴 (harmless) 值。
    3. 如果当前点的 Y 值大于已知的最大 Y 值，则该点属于 Pareto 前沿。
    """
    # 按照 helpful 从大到小排序，若 helpful 相同则按 harmless 从大到小排
    sorted_group = group.sort_values(by=['helpful', 'harmless'], ascending=[False, False])
    
    pareto_indices = []
    max_harmless = -float('inf')
    
    for idx, row in sorted_group.iterrows():
        # 只有当 harmless 大于当前见过的最大值时，才保留
        # (因为 helpful 已经是降序了，后面的点 helpful 一定 <= 当前点，
        #  所以只有 harmless 更高才能互不支配)
        if row['harmless'] > max_harmless:
            pareto_indices.append(idx)
            max_harmless = row['harmless']
            
    return group.loc[pareto_indices]

def plot_pareto_scatter(df, output_dir):
    """
    绘制 Pareto 前沿散点图
    """
    # 设置 Seaborn 主题
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制散点
    # 注意：因为是 Pareto 前沿，点通常比较少，可以稍微调大一点点的大小
    sns.scatterplot(
        data=df,
        x='helpful', 
        y='harmless',
        hue='experiment',
        style='experiment',
        alpha=0.8, # Pareto 点通常不重叠，透明度可以调高
        s=50,      # 点稍微大一点
        edgecolor='k', # 给点加个黑边，更清晰
        linewidth=0.5,
        ax=ax
    )

    # 可选：如果你想把同一个实验的 Pareto 点连成线，可以取消下面代码的注释
    sns.lineplot(
        data=df, 
        x='helpful', 
        y='harmless', 
        hue='experiment', 
        legend=False, 
        alpha=0.5, 
        ax=ax,
        sort=True # 自动按x轴排序连接
    )

    # 设置标题和标签
    ax.set_title('Pareto Frontier of Experiments')
    ax.set_xlabel('Helpful Score')
    ax.set_ylabel('Harmless Score')

    # # 自动调整坐标轴范围
    # if not df.empty:
    #     ax.set_xlim(df['helpful'].min() - 0.5, df['helpful'].max() + 0.5)
    #     ax.set_ylim(df['harmless'].min() - 0.5, df['harmless'].max() + 0.5)

    plt.tight_layout()

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
    plt.savefig(output_path / "pareto_frontier_seaborn.pdf", bbox_inches='tight')
    plt.savefig(output_path / "pareto_frontier_seaborn.png", dpi=300, bbox_inches='tight')
    print(f"Pareto plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="playground/visualization/scores")
    args = parser.parse_args()

    # 1. 读取原始数据
    df = read_jsonl_to_df(args.dir)

    if not df.empty:
        print(f"Total raw samples: {len(df)}")
        print(f"Experiments found: {df['experiment'].unique()}")
        # 绘制所有实验的散点图
        plot_enhanced_scatter(df, Path(args.dir).parent)

        # # 按实验分组并提取 Pareto 前沿
        # # group_keys=False 防止产生多级索引
        # pareto_df = df.groupby('experiment', group_keys=False).apply(get_pareto_frontier)
        
        # print(f"Total samples on Pareto Frontier: {len(pareto_df)}")

        # # 3. 绘制 Pareto 散点图
        # plot_pareto_scatter(pareto_df, Path(args.dir).parent)

    else:
        print("No data found.")