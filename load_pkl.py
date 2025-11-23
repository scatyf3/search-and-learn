import pickle
import pandas as pd

with open('timing_results_all.pkl', 'rb') as f:
    timing_results = pickle.load(f)

if isinstance(timing_results, pd.DataFrame):
    print(timing_results[['llm_gen_time', 'prm_score_time']].head())
    print("llm_gen_time 平均:", timing_results['llm_gen_time'].mean())
    print("prm_score_time 平均:", timing_results['prm_score_time'].mean())

'''
llm_gen_time 平均: 8.42949687242508
prm_score_time 平均: 11.89259090423584
'''