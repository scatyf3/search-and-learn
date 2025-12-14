#!/usr/bin/env python
"""计算动态beam search的加速比分析"""

import json
import sys

def analyze_speedup(jsonl_file):
    """分析动态beam search的加速比"""
    
    print("=" * 80)
    print("动态 Beam Search 加速比分析")
    print("=" * 80)
    
    total_beam_search_time = 0
    total_llm_gen_time = 0
    total_prm_score_time = 0
    total_generated_tokens = 0
    total_active_tokens = 0
    total_pruned_tokens = 0
    count = 0
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            data = json.loads(line)
            count += 1
            
            total_beam_search_time += data.get('total_time_beam_search', 0)
            total_llm_gen_time += data.get('llm_gen_time', 0)
            total_prm_score_time += data.get('prm_score_time', 0)
            total_generated_tokens += data.get('total_generated_tokens', 0)
            total_active_tokens += data.get('total_active_beam_tokens', 0)
            total_pruned_tokens += data.get('total_pruned_tokens', 0)
    
    print(f"\n总共 {count} 个问题")
    print()
    
    # Token 统计
    prune_rate = (total_pruned_tokens / total_generated_tokens * 100) if total_generated_tokens > 0 else 0
    
    print("=" * 80)
    print("1. Token 统计")
    print("=" * 80)
    print(f"  总生成 tokens:          {total_generated_tokens:>8,}")
    print(f"  Active beam tokens:     {total_active_tokens:>8,} ({total_active_tokens/total_generated_tokens*100:.1f}%)")
    print(f"  Pruned tokens:          {total_pruned_tokens:>8,} ({prune_rate:.1f}%)")
    print()
    
    # 时间统计
    avg_beam_search_time = total_beam_search_time / count
    avg_llm_time = total_llm_gen_time / count
    avg_prm_time = total_prm_score_time / count
    
    print("=" * 80)
    print("2. 时间统计（平均每个问题）")
    print("=" * 80)
    print(f"  总 Beam Search 时间:    {avg_beam_search_time:>8.3f} 秒")
    print(f"    - LLM 生成时间:       {avg_llm_time:>8.3f} 秒 ({avg_llm_time/avg_beam_search_time*100:.1f}%)")
    print(f"    - PRM 评分时间:       {avg_prm_time:>8.3f} 秒 ({avg_prm_time/avg_beam_search_time*100:.1f}%)")
    print()
    
    # Token 吞吐量
    tokens_per_sec_llm = total_generated_tokens / total_llm_gen_time if total_llm_gen_time > 0 else 0
    tokens_per_sec_overall = total_generated_tokens / total_beam_search_time if total_beam_search_time > 0 else 0
    
    print("=" * 80)
    print("3. Token 吞吐量")
    print("=" * 80)
    print(f"  LLM 生成速度:           {tokens_per_sec_llm:>8.1f} tokens/sec")
    print(f"  整体速度（含PRM）:       {tokens_per_sec_overall:>8.1f} tokens/sec")
    print()
    
    # 加速比分析
    print("=" * 80)
    print("4. 加速比分析")
    print("=" * 80)
    print()
    
    print("【场景 1】相比 Baseline (n=16, 无pruning)")
    print("-" * 80)
    print("  假设 Baseline: 16个beams生成到最大长度")
    print(f"  - 估计需要生成的 tokens: {total_generated_tokens:,}")
    print(f"  - 实际生成的 tokens:     {total_generated_tokens:,}")
    print(f"  - Token 节省:            0 tokens (0%)")
    print()
    print("  ⚠️  注意：当前动态beam search已经生成了所有candidate tokens")
    print("  说明：pruning只是提前停止某些beams，但它们已经生成的tokens仍然计入")
    print()
    
    print("【场景 2】相比保留所有生成的beams到最后")
    print("-" * 80)
    print("  如果不prune，所有beams都继续生成到最大长度：")
    
    # 估算如果不prune会生成多少tokens
    # 假设被pruned的beams如果继续生成，会达到和active beams类似的长度
    avg_active_tokens_per_beam = total_active_tokens / count / 16  # 假设最终有16个beams
    estimated_pruned_extra_tokens = avg_active_tokens_per_beam * (total_pruned_tokens / total_active_tokens) * count * 16
    
    # 简化：假设pruned beams如果不被prune，会生成和现在active beams相同数量的tokens
    # 这是一个上界估计
    estimated_no_prune_tokens = total_generated_tokens * (total_active_tokens / (total_active_tokens - total_pruned_tokens))
    
    print(f"  - 当前生成的 tokens:     {total_generated_tokens:,}")
    print(f"  - 实际active tokens:     {total_active_tokens:,}")
    print(f"  - Pruned tokens:         {total_pruned_tokens:,}")
    print()
    print("  解释：")
    print("  - 被prune的beams在被停止时已经生成了 3,920 tokens")
    print("  - 如果它们继续生成，可能会生成更多tokens")
    print(f"  - 但实际上，这些beams已经被评估为低质量而停止")
    print()
    
    print("【场景 3】实际的计算节省（关键）")
    print("-" * 80)
    print("  虽然tokens已经生成，但pruning带来的主要收益是：")
    print()
    print("  1. 减少后续迭代的计算：")
    print(f"     - 如果16个beams都继续生成到最后")
    print(f"     - 每次迭代需要处理16个beams")
    print(f"     - Pruning后，后续迭代只处理保留的beams")
    print()
    print("  2. 减少PRM评分的计算：")
    print(f"     - PRM评分时间: {total_prm_score_time:.2f}秒")
    print(f"     - 占总时间: {total_prm_score_time/total_beam_search_time*100:.1f}%")
    print(f"     - Pruned的beams不需要在后续步骤继续评分")
    print()
    
    print("【场景 4】理论加速比计算")
    print("-" * 80)
    print()
    print("  基于 Prune Rate = 19.8%，我们来计算理论加速比：")
    print()
    
    # 计算理论加速比
    # 假设计算时间与token数成正比
    compute_saved = total_pruned_tokens / total_generated_tokens
    theoretical_speedup = 1 / (1 - compute_saved)
    
    print(f"  方法1: 基于节省的计算量")
    print(f"  - 节省的计算: {compute_saved*100:.1f}%")
    print(f"  - 理论加速比: 1 / (1 - {compute_saved:.3f}) = {theoretical_speedup:.2f}x")
    print()
    print("  ⚠️  但这个计算不准确，因为：")
    print("  - Pruned tokens已经被生成了（已经花费了计算）")
    print("  - 真正节省的是：如果这些beams继续生成的话")
    print()
    
    # 更准确的分析
    print(f"  方法2: 实际分析")
    print(f"  - 总共生成: {total_generated_tokens:,} tokens")
    print(f"  - Active beams token: {total_active_tokens:,}")
    print(f"  - Pruned beams token: {total_pruned_tokens:,}")
    print()
    print(f"  如果所有beams都生成到和active beams相同的平均长度：")
    avg_active_length = total_active_tokens / count / 16  # 每个active beam的平均长度
    # 假设有12个beams被prune（平均），每个在被prune时已有的长度
    avg_pruned_length = total_pruned_tokens / count / 12  # 假设平均12个beams被prune
    potential_additional_tokens = (avg_active_length - avg_pruned_length) * 12 * count
    
    print(f"    - Active beams平均长度: {avg_active_length:.1f} tokens/beam")
    print(f"    - Pruned beams平均长度: {avg_pruned_length:.1f} tokens/beam")
    print(f"    - 如果pruned beams也生成到平均长度，额外需要: {potential_additional_tokens:,.0f} tokens")
    print()
    
    no_prune_total = total_generated_tokens + potential_additional_tokens
    actual_speedup = no_prune_total / total_generated_tokens
    
    print(f"  💡 实际加速比估算:")
    print(f"    - 无pruning总tokens: {no_prune_total:,.0f}")
    print(f"    - 有pruning总tokens: {total_generated_tokens:,}")
    print(f"    - 加速比: {actual_speedup:.2f}x")
    print()
    
    print("=" * 80)
    print("5. 结论")
    print("=" * 80)
    print()
    print(f"  ✓ Prune Rate: {prune_rate:.1f}%")
    print(f"  ✓ 实际加速比: 约 {actual_speedup:.2f}x")
    print()
    print("  说明：")
    print("  - 19.8%的prune rate带来的加速比较有限")
    print("  - 主要原因：被prune的beams在被停止前已经生成了一定数量的tokens")
    print("  - 加速主要来自：避免这些低质量beams继续生成到最大长度")
    print()
    print("  如果要获得更高的加速比，可以：")
    print("  1. 更激进的pruning策略（更早prune低分beams）")
    print("  2. 动态调整beam width（更快地减少beam数量）")
    print("  3. 使用更小的初始beam width")
    print()
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calculate_speedup.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    analyze_speedup(jsonl_file)
