#!/usr/bin/env python
"""对比 dynamic beam search 和 baseline 的加速比"""

import json
import sys

def analyze_file(jsonl_file, label):
    """分析单个文件的统计信息"""
    
    total_time = 0
    total_llm_time = 0
    total_prm_time = 0
    total_tokens = 0
    count = 0
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            data = json.loads(line)
            count += 1
            
            # 时间统计
            if 'total_time_beam_search' in data:
                total_time += data['total_time_beam_search']
            if 'llm_gen_time' in data:
                total_llm_time += data['llm_gen_time']
            if 'prm_score_time' in data:
                total_prm_time += data['prm_score_time']
            
            # Token统计
            if 'total_generated_tokens' in data:
                total_tokens += data['total_generated_tokens']
            elif 'completion_tokens' in data:
                # Baseline文件：直接统计completion_tokens
                tokens = data['completion_tokens']
                if isinstance(tokens, list):
                    if isinstance(tokens[0], list):
                        # 嵌套列表
                        for sublist in tokens:
                            total_tokens += sum(sublist)
                    else:
                        total_tokens += sum(tokens)
    
    return {
        'count': count,
        'total_time': total_time,
        'llm_time': total_llm_time,
        'prm_time': total_prm_time,
        'total_tokens': total_tokens,
        'avg_time': total_time / count if count > 0 else 0,
        'avg_tokens': total_tokens / count if count > 0 else 0,
    }

def compare_speedup(baseline_file, dynamic_file):
    """对比baseline和dynamic的加速比"""
    
    print("=" * 80)
    print("Dynamic Beam Search vs Baseline 加速比对比")
    print("=" * 80)
    print()
    
    print("分析文件...")
    baseline = analyze_file(baseline_file, "Baseline")
    dynamic = analyze_file(dynamic_file, "Dynamic")
    
    print("=" * 80)
    print("1. 基本统计")
    print("=" * 80)
    print(f"\n{'指标':<30} {'Baseline':<20} {'Dynamic':<20}")
    print("-" * 80)
    print(f"{'问题数量':<30} {baseline['count']:<20} {dynamic['count']:<20}")
    print()
    
    print("=" * 80)
    print("2. Token 统计")
    print("=" * 80)
    print(f"\n{'指标':<30} {'Baseline':<20} {'Dynamic':<20} {'比率':<20}")
    print("-" * 80)
    print(f"{'总 tokens':<30} {baseline['total_tokens']:>19,} {dynamic['total_tokens']:>19,} {dynamic['total_tokens']/baseline['total_tokens']:>19.2%}")
    print(f"{'平均 tokens/问题':<30} {baseline['avg_tokens']:>19.1f} {dynamic['avg_tokens']:>19.1f} {dynamic['avg_tokens']/baseline['avg_tokens']:>19.2%}")
    print()
    
    # Token加速比
    token_speedup = baseline['total_tokens'] / dynamic['total_tokens'] if dynamic['total_tokens'] > 0 else 0
    token_reduction = (1 - dynamic['total_tokens'] / baseline['total_tokens']) * 100 if baseline['total_tokens'] > 0 else 0
    
    print(f"💡 Token 节省: {token_reduction:.1f}%")
    print(f"💡 Token 加速比: {token_speedup:.2f}x")
    print()
    
    print("=" * 80)
    print("3. 时间统计")
    print("=" * 80)
    
    if baseline['total_time'] > 0 and dynamic['total_time'] > 0:
        print(f"\n{'指标':<30} {'Baseline':<20} {'Dynamic':<20} {'比率':<20}")
        print("-" * 80)
        print(f"{'总时间 (秒)':<30} {baseline['total_time']:>19.1f} {dynamic['total_time']:>19.1f} {dynamic['total_time']/baseline['total_time']:>19.2%}")
        print(f"{'平均时间/问题 (秒)':<30} {baseline['avg_time']:>19.2f} {dynamic['avg_time']:>19.2f} {dynamic['avg_time']/baseline['avg_time']:>19.2%}")
        
        if baseline['llm_time'] > 0:
            print(f"{'  - LLM 生成时间':<30} {baseline['llm_time']/baseline['count']:>19.2f} {dynamic['llm_time']/dynamic['count']:>19.2f} {(dynamic['llm_time']/dynamic['count'])/(baseline['llm_time']/baseline['count']):>19.2%}")
        
        if baseline['prm_time'] > 0:
            print(f"{'  - PRM 评分时间':<30} {baseline['prm_time']/baseline['count']:>19.2f} {dynamic['prm_time']/dynamic['count']:>19.2f} {(dynamic['prm_time']/dynamic['count'])/(baseline['prm_time']/baseline['count']):>19.2%}")
        
        print()
        
        # 时间加速比
        time_speedup = baseline['total_time'] / dynamic['total_time'] if dynamic['total_time'] > 0 else 0
        time_reduction = (1 - dynamic['total_time'] / baseline['total_time']) * 100 if baseline['total_time'] > 0 else 0
        
        print(f"💡 时间节省: {time_reduction:.1f}%")
        print(f"💡 时间加速比: {time_speedup:.2f}x")
    else:
        print("\n⚠️  Baseline文件没有时间统计信息")
        print(f"   只有Dynamic文件有时间数据:")
        print(f"   - 平均时间/问题: {dynamic['avg_time']:.2f} 秒")
        if dynamic['llm_time'] > 0:
            print(f"   - LLM 生成时间: {dynamic['llm_time']/dynamic['count']:.2f} 秒")
        if dynamic['prm_time'] > 0:
            print(f"   - PRM 评分时间: {dynamic['prm_time']/dynamic['count']:.2f} 秒")
        time_speedup = 0
        time_reduction = 0
    
    print()
    
    print("=" * 80)
    print("4. 效率分析")
    print("=" * 80)
    
    # Tokens per second
    baseline_tps = baseline['total_tokens'] / baseline['total_time'] if baseline['total_time'] > 0 else 0
    dynamic_tps = dynamic['total_tokens'] / dynamic['total_time'] if dynamic['total_time'] > 0 else 0
    
    print(f"\nTokens per second:")
    print(f"  Baseline: {baseline_tps:.1f} tokens/sec")
    print(f"  Dynamic:  {dynamic_tps:.1f} tokens/sec")
    print()
    
    # 归一化效率（每个token的时间）
    baseline_time_per_token = baseline['total_time'] / baseline['total_tokens'] if baseline['total_tokens'] > 0 else 0
    dynamic_time_per_token = dynamic['total_time'] / dynamic['total_tokens'] if dynamic['total_tokens'] > 0 else 0
    
    print(f"Time per token:")
    print(f"  Baseline: {baseline_time_per_token*1000:.2f} ms/token")
    print(f"  Dynamic:  {dynamic_time_per_token*1000:.2f} ms/token")
    print()
    
    print("=" * 80)
    print("5. 总结")
    print("=" * 80)
    print()
    
    if time_speedup > 0:
        print(f"🚀 整体加速比（时间）: {time_speedup:.2f}x")
        print(f"   - 时间节省: {time_reduction:.1f}%")
        print()
    
    print(f"💾 Token 减少: {token_speedup:.2f}x")
    print(f"   - Token 节省: {token_reduction:.1f}%")
    print()
    
    # 分析差异
    if time_speedup > 0:
        if time_speedup > token_speedup:
            diff = time_speedup - token_speedup
            print(f"⚡ 时间加速比 > Token减少比 (差异: {diff:.2f}x)")
            print(f"   说明: Dynamic方法除了减少tokens，还提高了计算效率")
        elif token_speedup > time_speedup:
            diff = token_speedup - time_speedup
            print(f"⚠️  Token减少比 > 时间加速比 (差异: {diff:.2f}x)")
            print(f"   说明: 虽然减少了tokens，但额外的开销影响了整体速度")
        else:
            print(f"✅ 时间加速比 ≈ Token减少比")
            print(f"   说明: Token减少直接转化为时间节省")
    else:
        print(f"📊 基于Token统计的分析:")
        print(f"   - Baseline: 500个问题，平均{baseline['avg_tokens']:.0f} tokens/问题")
        print(f"   - Dynamic: 50个问题，平均{dynamic['avg_tokens']:.0f} tokens/问题")
        print(f"   - Token减少: {token_reduction:.1f}%")
        print()
        print(f"⚠️  注意: 两个文件的问题数量不同，且baseline没有时间数据")
        print(f"   Token对比可能不完全准确（需要相同的问题集）")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python compare_baseline_speedup.py <baseline_file> <dynamic_file>")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    dynamic_file = sys.argv[2]
    compare_speedup(baseline_file, dynamic_file)
