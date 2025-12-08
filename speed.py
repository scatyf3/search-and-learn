import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_and_plot_times(file_paths, output_dir="analysis_results"):
    # 1. 准备工作
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_data = []

    print(f"正在分析 {len(file_paths)} 个文件...")

    # 2. 读取数据
    for file_path in file_paths:
        file_name = os.path.basename(file_path) # 只取文件名，方便展示
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # 提取时间，如果没有则默认为 0
                        llm_time = entry.get('llm_gen_time', 0)
                        prm_time = entry.get('prm_score_time', 0)
                        
                        # 过滤掉异常数据 (例如时间为 None 的情况)
                        if llm_time is None: llm_time = 0
                        if prm_time is None: prm_time = 0

                        all_data.append({
                            "File": file_name,
                            "LLM Gen Time (s)": llm_time,
                            "PRM Time (s)": prm_time,
                            "Total Time (s)": llm_time + prm_time
                        })
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"⚠️ 警告: 找不到文件 {file_path}")

    if not all_data:
        print("❌ 没有读取到有效数据。")
        return

    # 转换 Pandas DataFrame
    df = pd.DataFrame(all_data)

    # 3. 计算平均值并生成 TSV
    # 按文件分组计算均值
    summary_df = df.groupby("File")[["LLM Gen Time (s)", "PRM Time (s)", "Total Time (s)"]].mean().reset_index()
    
    # 格式化保留4位小数
    summary_df = summary_df.round(4)

    tsv_path = os.path.join(output_dir, "time_comparison.tsv")
    summary_df.to_csv(tsv_path, sep='\t', index=False)
    
    print("\n=== 平均耗时对比 (TSV) ===")
    print(summary_df.to_string(index=False))
    print(f"\nTSV 文件已保存至: {tsv_path}")

    # 4. 绘图 - 耗时分布 (箱线图 Boxplot)
    # 箱线图最适合对比多个源的数据分布
    
    # 为了画图方便，我们需要把 DataFrame 从"宽格式"转换为"长格式"
    df_melted = df.melt(id_vars=["File"], 
                        value_vars=["LLM Gen Time (s)", "PRM Time (s)"],
                        var_name="Time Type", 
                        value_name="Time (Seconds)")

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # 画箱线图
    ax = sns.boxplot(data=df_melted, x="File", y="Time (Seconds)", hue="Time Type", showfliers=False) 
    # showfliers=False 不显示极端异常值点，让图更好看。如果想看异常值改成True
    
    plt.title("Time Distribution Comparison (LLM vs PRM)", fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    dist_plot_path = os.path.join(output_dir, "time_distribution_boxplot.png")
    plt.savefig(dist_plot_path, dpi=300)
    print(f"分布图已保存至: {dist_plot_path}")
    plt.close()

    # 5. 绘图 - 平均耗时堆叠图 (Stacked Bar)
    # 能够直观看到 LLM 和 PRM 在总时间里的占比
    
    summary_df.set_index("File")[["LLM Gen Time (s)", "PRM Time (s)"]].plot(
        kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon']
    )
    
    plt.title("Average Time Breakdown per File", fontsize=15)
    plt.ylabel("Average Time (Seconds)")
    plt.xlabel("File Name")
    plt.xticks(rotation=45)
    plt.tight_layout()

    avg_plot_path = os.path.join(output_dir, "average_time_breakdown.png")
    plt.savefig(avg_plot_path, dpi=300)
    print(f"均值对比图已保存至: {avg_plot_path}")
    plt.close()

if __name__ == "__main__":
    # === 在这里配置你要对比的文件路径 ===
    jsonl_files = [
        # "data/meta-llama/Llama-3.2-3B-Instruct/best_of_n_n4_completions.jsonl",
        "data/meta-llama/Llama-3.2-3B-Instruct/best_of_n_transformers_n4_completions_merged.jsonl",
        "data/meta-llama/Llama-3.2-3B-Instruct/best_of_n_speculative_n4_completions_merged.jsonl"
    ]
    
    analyze_and_plot_times(jsonl_files)