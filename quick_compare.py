#!/usr/bin/env python
"""对比前50个问题的时间和token统计"""

import json

baseline_file = "data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n16_temp1.0_exp_20251212_084748_completions.jsonl"
dynamic_file = "data/meta-llama/Llama-3.2-1B-Instruct/beam_search_dynamic_n16_temp0.8_exp_20251213_192225_completions.jsonl"

print("=" * 80)
print("Baseline vs Dynamic Beam Search 加速比分析（前50个问题）")
print("=" * 80)
print()

# 读取两个文件的前50条
baseline_data = []
dynamic_data = []

with open(baseline_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 50:
            break
        if line.strip() and not line.startswith('#'):
            baseline_data.append(json.loads(line))

with open(dynamic_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 50:
            break
        if line.strip() and not line.startswith('#'):
            dynamic_data.append(json.loads(line))

print(f"读取数据: Baseline {len(baseline_data)} 条, Dynamic {len(dynamic_data)} 条")
print()

# 统计
baseline_llm_time = sum(d.get('llm_gen_time', 0) for d in baseline_data)
baseline_prm_time = sum(d.get('prm_score_time', 0) for d in baseline_data)
baseline_total_time = baseline_llm_time + baseline_prm_time
baseline_tokens = sum(sum(d.get('completion_tokens', [])) for d in baseline_data)

dynamic_llm_time = sum(d.get('llm_gen_time', 0) for d in dynamic_data)
dynamic_prm_time = sum(d.get('prm_score_time', 0) for d in dynamic_data)
dynamic_total_time = sum(d.get('total_time_beam_search', 0) for d in dynamic_data)
dynamic_tokens = sum(d.get('total_generated_tokens', 0) for d in dynamic_data)
dynamic_active_tokens = sum(d.get('total_active_beam_tokens', 0) for d in dynamic_data)
dynamic_pruned_tokens = sum(d.get('total_pruned_tokens', 0) for d in dynamic_data)

print("=" * 80)
print("1. Token 统计")
print("=" * 80)
print(f"\n{'指标':<40} {'Baseline':<15} {'Dynamic':<15} {'比率':<15}")
print("-" * 80)
print(f"{'总生成 tokens':<40} {baseline_tokens:>14,} {dynamic_tokens:>14,} {dynamic_tokens/baseline_tokens:>14.2%}")
print(f"{'平均 tokens/问题':<40} {baseline_tokens/50:>14.1f} {dynamic_tokens/50:>14.1f} {(dynamic_tokens/50)/(baseline_tokens/50):>14.2%}")

if dynamic_active_tokens > 0:
    print(f"{'  - Active beam tokens':<40} {'N/A':>14} {dynamic_active_tokens:>14,} {'':>15}")
    print(f"{'  - Pruned tokens':<40} {'N/A':>14} {dynamic_pruned_tokens:>14,} {'':>15}")
    prune_rate = dynamic_pruned_tokens / dynamic_tokens * 100
    print(f"{'  - Prune rate':<40} {'N/A':>14} {prune_rate:>13.1f}% {'':>15}")

token_reduction = (1 - dynamic_tokens / baseline_tokens) * 100
token_speedup = baseline_tokens / dynamic_tokens

print()
print(f"💡 Token 减少: {token_reduction:.1f}%")
print(f"💡 Token 加速比: {token_speedup:.2f}x")
print()

print("=" * 80)
print("2. 时间统计")
print("=" * 80)
print(f"\n{'指标':<40} {'Baseline':<15} {'Dynamic':<15} {'比率':<15}")
print("-" * 80)
print(f"{'LLM 生成时间 (秒)':<40} {baseline_llm_time:>14.1f} {dynamic_llm_time:>14.1f} {dynamic_llm_time/baseline_llm_time:>14.2%}")
print(f"{'PRM 评分时间 (秒)':<40} {baseline_prm_time:>14.1f} {dynamic_prm_time:>14.1f} {dynamic_prm_time/baseline_prm_time:>14.2%}")
print(f"{'总时间 (秒)':<40} {baseline_total_time:>14.1f} {dynamic_total_time:>14.1f} {dynamic_total_time/baseline_total_time:>14.2%}")
print()
print(f"{'平均 LLM 时间/问题 (秒)':<40} {baseline_llm_time/50:>14.2f} {dynamic_llm_time/50:>14.2f} {(dynamic_llm_time/50)/(baseline_llm_time/50):>14.2%}")
print(f"{'平均 PRM 时间/问题 (秒)':<40} {baseline_prm_time/50:>14.2f} {dynamic_prm_time/50:>14.2f} {(dynamic_prm_time/50)/(baseline_prm_time/50):>14.2%}")
print(f"{'平均总时间/问题 (秒)':<40} {baseline_total_time/50:>14.2f} {dynamic_total_time/50:>14.2f} {(dynamic_total_time/50)/(baseline_total_time/50):>14.2%}")

time_reduction = (1 - dynamic_total_time / baseline_total_time) * 100
time_speedup = baseline_total_time / dynamic_total_time

print()
print(f"💡 总时间节省: {time_reduction:.1f}%")
print(f"💡 总时间加速比: {time_speedup:.2f}x")
print()

# 分解时间加速比
llm_speedup = baseline_llm_time / dynamic_llm_time if dynamic_llm_time > 0 else 0
prm_speedup = baseline_prm_time / dynamic_prm_time if dynamic_prm_time > 0 else 0

print(f"  - LLM 生成加速比: {llm_speedup:.2f}x")
print(f"  - PRM 评分加速比: {prm_speedup:.2f}x")
print()

print("=" * 80)
print("3. 效率分析")
print("=" * 80)
print()

# Tokens per second
baseline_tps = baseline_tokens / baseline_llm_time if baseline_llm_time > 0 else 0
dynamic_tps = dynamic_tokens / dynamic_llm_time if dynamic_llm_time > 0 else 0

print(f"LLM 生成速度:")
print(f"  Baseline: {baseline_tps:>8.1f} tokens/sec")
print(f"  Dynamic:  {dynamic_tps:>8.1f} tokens/sec")
print()

# Time per token
baseline_ms_per_token = baseline_llm_time * 1000 / baseline_tokens if baseline_tokens > 0 else 0
dynamic_ms_per_token = dynamic_llm_time * 1000 / dynamic_tokens if dynamic_tokens > 0 else 0

print(f"每个 token 的 LLM 生成时间:")
print(f"  Baseline: {baseline_ms_per_token:.3f} ms/token")
print(f"  Dynamic:  {dynamic_ms_per_token:.3f} ms/token")
print()

# Time breakdown
print(f"时间占比:")
print(f"  Baseline: LLM {baseline_llm_time/baseline_total_time*100:.1f}%, PRM {baseline_prm_time/baseline_total_time*100:.1f}%")
print(f"  Dynamic:  LLM {dynamic_llm_time/dynamic_total_time*100:.1f}%, PRM {dynamic_prm_time/dynamic_total_time*100:.1f}%")
print()

print("=" * 80)
print("4. 关键发现")
print("=" * 80)
print()

print(f"🚀 整体加速比: {time_speedup:.2f}x ({time_reduction:.1f}% 时间节省)")
print()

print(f"📊 分解分析:")
print(f"  • Token 减少: {token_speedup:.2f}x ({token_reduction:.1f}% 节省)")
print(f"  • LLM 生成加速: {llm_speedup:.2f}x")
print(f"  • PRM 评分加速: {prm_speedup:.2f}x")
print()

# 分析不同之处
if llm_speedup > token_speedup:
    print(f"✅ LLM 加速比 ({llm_speedup:.2f}x) > Token 减少比 ({token_speedup:.2f}x)")
    print(f"   说明: Dynamic 除了减少 tokens，还提升了 LLM 生成效率")
elif token_speedup > llm_speedup:
    diff = token_speedup - llm_speedup
    print(f"⚠️  Token 减少比 ({token_speedup:.2f}x) > LLM 加速比 ({llm_speedup:.2f}x) [差异: {diff:.2f}x]")
    print(f"   说明: 虽然减少了 {token_reduction:.1f}% 的 tokens，但 LLM 加速比更小")
    print(f"   可能原因: Dynamic beam search 的额外开销（beam管理、pruning等）")
else:
    print(f"✅ LLM 加速比 ≈ Token 减少比 ({llm_speedup:.2f}x)")
    print(f"   说明: Token 减少直接转化为 LLM 时间节省")

print()

if prm_speedup < 1.0:
    print(f"⚠️  PRM 评分时间增加: {1/prm_speedup:.2f}x")
    print(f"   Dynamic: {dynamic_prm_time:.1f}秒 vs Baseline: {baseline_prm_time:.1f}秒")
    print(f"   可能原因: 更频繁的评分、更多的中间步骤评估")
elif prm_speedup > 1.0:
    print(f"✅ PRM 评分加速: {prm_speedup:.2f}x")
    print(f"   说明: Pruning 减少了需要评分的 beams")

print()
print("=" * 80)
print("5. 总结")
print("=" * 80)
print()

print(f"对比相同的 50 个问题:")
print(f"  • Baseline: 固定 16 beams，无 pruning")
print(f"  • Dynamic: 动态 beam width，25% prune rate")
print()
print(f"结果:")
print(f"  ✅ 总时间加速: {time_speedup:.2f}x (节省 {time_reduction:.1f}% 时间)")
print(f"  ✅ Token 减少: {token_speedup:.2f}x (节省 {token_reduction:.1f}% tokens)")
print(f"  ✅ LLM 生成加速: {llm_speedup:.2f}x")
print(f"  {'✅' if prm_speedup > 1.0 else '⚠️ '} PRM 评分变化: {prm_speedup:.2f}x")
print()

if time_speedup < 1.5:
    print(f"💡 改进建议:")
    print(f"  - 当前加速比 {time_speedup:.2f}x 相对有限")
    print(f"  - 建议: 更激进的 early pruning (当前 prune rate: {prune_rate:.1f}%)")
    print(f"  - 建议: 更快的 beam width 衰减")
    if prm_speedup < 1.0:
        print(f"  - 建议: 优化 PRM 评分频率（当前 PRM 时间增加了 {(1/prm_speedup - 1)*100:.1f}%）")

print()
print("=" * 80)
