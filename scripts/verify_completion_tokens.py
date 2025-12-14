#!/usr/bin/env python
"""
验证 completion_tokens 字段的准确性

该脚本会：
1. 读取 JSONL 文件中的 completions 和 completion_tokens
2. 使用 tokenizer 重新计算每个 completion 的 token 数量
3. 对比实际值和记录值，报告差异
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from collections import defaultdict

'''
data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_temp1.0_exp_20251210_141452_completions.jsonl
data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_temp1.0_exp_20251210_143633_completions.jsonl

'''

def verify_tokens(jsonl_path: str, model_path: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """验证 JSONL 文件中的 completion_tokens 是否准确"""
    
    print(f"📁 加载文件: {jsonl_path}")
    print(f"🤖 加载 tokenizer: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 统计信息
    stats = {
        'total_examples': 0,
        'total_completions': 0,
        'exact_matches': 0,
        'mismatches': 0,
        'errors': 0,
        'max_diff': 0,
        'diffs': []
    }
    
    # 读取 JSONL 文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                data = json.loads(line)
                stats['total_examples'] += 1
                
                # 获取 completions 和 completion_tokens
                completions = data.get('completions', [])
                recorded_tokens = data.get('completion_tokens', [])
                
                if len(completions) != len(recorded_tokens):
                    print(f"⚠️  行 {line_num}: completions 数量 ({len(completions)}) 与 completion_tokens 数量 ({len(recorded_tokens)}) 不匹配")
                    stats['errors'] += 1
                    continue
                
                # 逐个验证
                for idx, (completion, recorded) in enumerate(zip(completions, recorded_tokens)):
                    stats['total_completions'] += 1
                    
                    # 使用 tokenizer 计算实际 token 数
                    actual = len(tokenizer.encode(completion, add_special_tokens=False))
                    
                    diff = abs(actual - recorded)
                    
                    if actual == recorded:
                        stats['exact_matches'] += 1
                    else:
                        stats['mismatches'] += 1
                        stats['diffs'].append(diff)
                        stats['max_diff'] = max(stats['max_diff'], diff)
                        
                        # 打印前几个不匹配的示例
                        if stats['mismatches'] <= 5:
                            print(f"\n❌ 不匹配 (行 {line_num}, completion {idx}):")
                            print(f"   记录值: {recorded}")
                            print(f"   实际值: {actual}")
                            print(f"   差异: {diff}")
                            print(f"   文本长度: {len(completion)} 字符")
                            if len(completion) < 100:
                                print(f"   文本: {completion[:100]}...")
            
            except json.JSONDecodeError as e:
                print(f"⚠️  行 {line_num}: JSON 解析错误 - {e}")
                stats['errors'] += 1
            except Exception as e:
                print(f"⚠️  行 {line_num}: 处理错误 - {e}")
                stats['errors'] += 1
    
    # 打印统计结果
    print("\n" + "="*60)
    print("📊 验证结果统计")
    print("="*60)
    print(f"总样本数: {stats['total_examples']}")
    print(f"总 completion 数: {stats['total_completions']}")
    print(f"✅ 完全匹配: {stats['exact_matches']} ({stats['exact_matches']/max(stats['total_completions'],1)*100:.1f}%)")
    print(f"❌ 不匹配: {stats['mismatches']} ({stats['mismatches']/max(stats['total_completions'],1)*100:.1f}%)")
    print(f"⚠️  错误: {stats['errors']}")
    
    if stats['mismatches'] > 0:
        print(f"\n差异统计:")
        print(f"  最大差异: {stats['max_diff']} tokens")
        print(f"  平均差异: {sum(stats['diffs'])/len(stats['diffs']):.2f} tokens")
        print(f"  中位数差异: {sorted(stats['diffs'])[len(stats['diffs'])//2]} tokens")
    
    print("="*60)
    
    # 返回是否全部准确
    return stats['mismatches'] == 0 and stats['errors'] == 0


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_completion_tokens.py <jsonl_file> [model_path]")
        print("\n示例:")
        print("  python verify_completion_tokens.py data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_*.jsonl")
        print("  python verify_completion_tokens.py data/results.jsonl meta-llama/Llama-3.2-3B-Instruct")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-1B-Instruct"
    
    if not Path(jsonl_path).exists():
        print(f"❌ 文件不存在: {jsonl_path}")
        sys.exit(1)
    
    is_accurate = verify_tokens(jsonl_path, model_path)
    
    if is_accurate:
        print("\n✅ 所有 completion_tokens 都准确！")
        sys.exit(0)
    else:
        print("\n⚠️  发现不准确的 completion_tokens")
        sys.exit(1)


if __name__ == "__main__":
    main()

'''
(base) [yf3005@ga007 search-and-learn]$ python scripts/calculate_tokens_per_second.py data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_temp1.0_exp_20251210_143633_completions.jsonl
Line 1: LLM Gen - 167.22 tokens/s
Line 1: PRM Score - 437.62 tokens/s
Line 2: LLM Gen - 56.49 tokens/s
Line 2: PRM Score - 76.03 tokens/s
Line 3: LLM Gen - 199.34 tokens/s
Line 3: PRM Score - 280.00 tokens/s
Line 4: LLM Gen - 172.63 tokens/s
Line 4: PRM Score - 509.62 tokens/s


(base) [yf3005@ga007 search-and-learn]$ python scripts/calculate_tokens_per_second.py data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_temp1.0_exp_20251210_141452_completions.jsonl
Line 1: LLM Gen - 167.22 tokens/s
Line 1: PRM Score - 437.62 tokens/s
Line 2: LLM Gen - 56.49 tokens/s
Line 2: PRM Score - 76.03 tokens/s
Line 3: LLM Gen - 199.34 tokens/s
Line 3: PRM Score - 280.00 tokens/s
Line 4: LLM Gen - 172.63 tokens/s
Line 4: PRM Score - 509.62 tokens/s
'''