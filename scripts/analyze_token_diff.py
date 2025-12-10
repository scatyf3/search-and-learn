#!/usr/bin/env python
"""
详细分析 completion_tokens 的计算差异

检查：
1. token_ids 是否包含 stop token
2. 文本编码和 token_ids 的差异
3. special tokens 的影响
"""

import json
from transformers import AutoTokenizer

def analyze_first_completion():
    jsonl_path = "data/meta-llama/Llama-3.2-1B-Instruct/beam_search_dynamic_n4_temp0.8_exp_20251208_225253_completions.jsonl"
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(jsonl_path, 'r') as f:
        data = json.loads(f.readline())
    
    completion = data['completions'][0]
    recorded_tokens = data['completion_tokens'][0]
    
    print("="*70)
    print("📝 第一个 completion 分析")
    print("="*70)
    print(f"\n文本长度: {len(completion)} 字符")
    print(f"记录的 token 数: {recorded_tokens}")
    
    # 方法1: 不加 special tokens
    tokens_no_special = tokenizer.encode(completion, add_special_tokens=False)
    print(f"\n方法1 - encode(add_special_tokens=False): {len(tokens_no_special)}")
    
    # 方法2: 加 special tokens  
    tokens_with_special = tokenizer.encode(completion, add_special_tokens=True)
    print(f"方法2 - encode(add_special_tokens=True): {len(tokens_with_special)}")
    
    # 方法3: tokenize
    tokens = tokenizer.tokenize(completion)
    print(f"方法3 - tokenize(): {len(tokens)}")
    
    # 查看前几个和后几个 token
    print(f"\n前 5 个 tokens (no special):")
    for i, tid in enumerate(tokens_no_special[:5]):
        print(f"  [{i}] {tid}: '{tokenizer.decode([tid])}'")
    
    print(f"\n后 5 个 tokens (no special):")
    for i, tid in enumerate(tokens_no_special[-5:], len(tokens_no_special)-5):
        print(f"  [{i}] {tid}: '{tokenizer.decode([tid])}'")
    
    # 检查是否有特殊字符
    if completion.endswith('\n\n'):
        print(f"\n⚠️  文本以 \\n\\n 结尾")
        without_stop = completion[:-2]
        tokens_without_stop = tokenizer.encode(without_stop, add_special_tokens=False)
        print(f"   去除 \\n\\n 后的 token 数: {len(tokens_without_stop)}")
        
        # 单独编码 \n\n
        stop_tokens = tokenizer.encode('\n\n', add_special_tokens=False)
        print(f"   \\n\\n 的 token 数: {len(stop_tokens)}")
        print(f"   \\n\\n 的 token IDs: {stop_tokens}")
        
    # 差异分析
    diff = recorded_tokens - len(tokens_no_special)
    print(f"\n📊 差异: {diff} tokens")
    
    if diff == 1:
        print("   可能原因: vLLM 的 token_ids 可能包含了一个额外的 token")
        print("   - 可能是 BOS/EOS token")
        print("   - 可能是 stop token 的计数方式不同")
    
    # 检查特殊 tokens
    print(f"\n🔍 特殊 tokens:")
    print(f"   BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_first_completion()
