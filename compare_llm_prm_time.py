import json
import pandas as pd

# 读取 jsonl 文件
records = []
with open('data/meta-llama/Llama-3.2-3B-Instruct/best_of_n_completions.jsonl', 'r') as f:
    for line in f:
        records.append(json.loads(line))

# 转为 DataFrame
df = pd.DataFrame(records)

# 检查时间字段
if 'llm_gen_time' in df.columns and 'prm_score_time' in df.columns:
    print('llm_gen_time 前5项:', df['llm_gen_time'].head())
    print('prm_score_time 前5项:', df['prm_score_time'].head())
    print('llm_gen_time 平均:', df['llm_gen_time'].mean())
    print('prm_score_time 平均:', df['prm_score_time'].mean())
else:
    print('未找到 llm_gen_time 或 prm_score_time 字段，请检查数据格式。')
