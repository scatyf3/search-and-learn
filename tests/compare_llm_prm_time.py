import json
import pandas as pd
import matplotlib.pyplot as plt

n_list = [4, 16, 64, 256]
base_path = 'data/meta-llama/Llama-3.2-3B-Instruct'
dfs = []

for n in n_list:
    file_path = f'{base_path}/best_of_n_n{n}_completions.jsonl'
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df['n'] = n
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# 分组统计
for n, group in all_df.groupby('n'):
    print(f'n={n}:')
    print('llm_gen_time 前5项:', group['llm_gen_time'].head())
    print('prm_score_time 前5项:', group['prm_score_time'].head())
    print('llm_gen_time 平均:', group['llm_gen_time'].mean())
    print('prm_score_time 平均:', group['prm_score_time'].mean())

# 绘制堆叠柱状图
bar_data = []
for n, group in all_df.groupby('n'):
    llm_mean = group['llm_gen_time'].mean()
    prm_mean = group['prm_score_time'].mean()
    total = llm_mean + prm_mean
    bar_data.append({
        'n': n,
        'llm_mean': llm_mean,
        'prm_mean': prm_mean,
        'total': total
    })

bar_df = pd.DataFrame(bar_data)

plt.figure(figsize=(8, 3))
plt.bar(bar_df['n'].astype(str), bar_df['llm_mean'], label='LLM', color='#4F81BD')
plt.bar(bar_df['n'].astype(str), bar_df['prm_mean'], bottom=bar_df['llm_mean'], label='PRM', color='#C0504D')
plt.xlabel('n')
plt.ylabel('Time (s)')
plt.title('LLM + PRM Time Breakdown by n')
plt.legend()
plt.tight_layout()
plt.savefig('llm_prm_time_breakdown.pdf', dpi=300)
plt.show()
