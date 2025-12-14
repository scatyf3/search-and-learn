#!/usr/bin/env python
"""检查 token 统计字段并校验token数量"""

import json
import sys

def check_token_statistics(jsonl_file):
    """检查 JSONL 文件中的 token 统计字段，并校验数量一致性"""
    
    total_count = 0
    token_mismatch_count = 0
    token_match_count = 0
    
    # 新字段统计
    total_generated_tokens_sum = 0
    total_active_beam_tokens_sum = 0
    total_pruned_tokens_sum = 0
    
    print(f"正在检查文件: {jsonl_file}\n")
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过配置行
            if line.startswith('#'):
                continue
            
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # 检查新的token统计字段
                has_total_generated = 'total_generated_tokens' in data
                has_total_active = 'total_active_beam_tokens' in data
                has_total_pruned = 'total_pruned_tokens' in data
                has_completion_tokens = 'completion_tokens' in data
                
                if total_count <= 3:  # 显示前3条详细信息
                    print(f"\n{'='*60}")
                    print(f"记录 {total_count} (行 {line_num}):")
                    
                    if has_total_generated:
                        val = data['total_generated_tokens']
                        print(f"  total_generated_tokens: {val}")
                        if isinstance(val, list) and len(val) > 0:
                            total_generated_tokens_sum += val[0]
                        elif isinstance(val, (int, float)):
                            total_generated_tokens_sum += val
                    
                    if has_total_active:
                        val = data['total_active_beam_tokens']
                        print(f"  total_active_beam_tokens: {val}")
                        if isinstance(val, list) and len(val) > 0:
                            total_active_beam_tokens_sum += val[0]
                        elif isinstance(val, (int, float)):
                            total_active_beam_tokens_sum += val
                    
                    if has_total_pruned:
                        val = data['total_pruned_tokens']
                        print(f"  total_pruned_tokens: {val}")
                        if isinstance(val, list) and len(val) > 0:
                            total_pruned_tokens_sum += val[0]
                        elif isinstance(val, (int, float)):
                            total_pruned_tokens_sum += val
                    
                    if has_completion_tokens:
                        completion_tokens = data['completion_tokens']
                        print(f"  completion_tokens: {completion_tokens}")
                        
                        # 计算 completion_tokens 总数
                        completion_total = 0
                        if isinstance(completion_tokens, list):
                            for item in completion_tokens:
                                if isinstance(item, list):
                                    completion_total += sum(item)
                                elif isinstance(item, (int, float)):
                                    completion_total += item
                        print(f"  -> completion_tokens 总数: {completion_total}")
                
                # 校验 token 数量一致性
                # 规则: total_active_beam_tokens + total_pruned_tokens = total_generated_tokens
                if has_total_generated and has_total_active and has_total_pruned:
                    # 提取实际值（可能是列表）
                    total_gen = data['total_generated_tokens']
                    if isinstance(total_gen, list):
                        total_gen = total_gen[0] if len(total_gen) > 0 else 0
                    
                    total_active = data['total_active_beam_tokens']
                    if isinstance(total_active, list):
                        total_active = total_active[0] if len(total_active) > 0 else 0
                    
                    total_pruned = data['total_pruned_tokens']
                    if isinstance(total_pruned, list):
                        total_pruned = total_pruned[0] if len(total_pruned) > 0 else 0
                    
                    # 校验
                    calculated_total = total_active + total_pruned
                    if abs(calculated_total - total_gen) < 0.01:  # 允许浮点误差
                        token_match_count += 1
                        if total_count <= 3:
                            print(f"  ✓ Token校验通过: {total_active} + {total_pruned} = {total_gen}")
                    else:
                        token_mismatch_count += 1
                        print(f"\n⚠️  记录 {total_count} (行 {line_num}): Token数量不匹配!")
                        print(f"  total_active_beam_tokens: {total_active}")
                        print(f"  total_pruned_tokens: {total_pruned}")
                        print(f"  active + pruned = {calculated_total}")
                        print(f"  total_generated_tokens: {total_gen}")
                        print(f"  差异: {calculated_total - total_gen}")
                
                # 额外校验: completion_tokens 总数 应该等于 total_active_beam_tokens
                if has_completion_tokens and has_total_active:
                    completion_tokens = data['completion_tokens']
                    completion_total = 0
                    if isinstance(completion_tokens, list):
                        for item in completion_tokens:
                            if isinstance(item, list):
                                completion_total += sum(item)
                            elif isinstance(item, (int, float)):
                                completion_total += item
                    
                    total_active = data['total_active_beam_tokens']
                    if isinstance(total_active, list):
                        total_active = total_active[0] if len(total_active) > 0 else 0
                    
                    if abs(completion_total - total_active) > 0.01:
                        print(f"\n⚠️  记录 {total_count} (行 {line_num}): completion_tokens 总数与 total_active_beam_tokens 不匹配!")
                        print(f"  completion_tokens 总数: {completion_total}")
                        print(f"  total_active_beam_tokens: {total_active}")
                        print(f"  差异: {completion_total - total_active}")
                    
            except json.JSONDecodeError as e:
                print(f"行 {line_num}: JSON解析错误 - {e}")
                continue
    
    print("\n" + "="*60)
    print(f"统计结果:")
    print(f"  总记录数: {total_count}")
    print(f"\n  Token数量校验:")
    print(f"    通过校验的记录: {token_match_count}")
    print(f"    未通过校验的记录: {token_mismatch_count}")
    
    if total_count > 0:
        print(f"\n  总Token统计 (所有记录合计):")
        print(f"    总生成token数: {total_generated_tokens_sum}")
        print(f"    最终active beam token数: {total_active_beam_tokens_sum}")
        print(f"    被prune的token数: {total_pruned_tokens_sum}")
        print(f"    Prune率: {total_pruned_tokens_sum / total_generated_tokens_sum * 100:.2f}%" if total_generated_tokens_sum > 0 else "    Prune率: N/A")
    
    print("="*60)
    
    if token_mismatch_count > 0:
        print(f"\n❌ 有 {token_mismatch_count} 条记录的token数量不匹配！")
    elif token_match_count > 0:
        print(f"\n✅ 所有 {token_match_count} 条记录的token数量都匹配！")
    else:
        print(f"\n⚠️  没有找到完整的token统计字段")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_pruned_tokens.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    check_token_statistics(jsonl_file)
