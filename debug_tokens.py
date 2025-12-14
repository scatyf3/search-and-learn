#!/usr/bin/env python
"""调试token数据结构"""

import json
import sys

if len(sys.argv) < 2:
    print("用法: python debug_tokens.py <jsonl_file>")
    sys.exit(1)

jsonl_file = sys.argv[1]

with open(jsonl_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        if line.startswith('#') or not line.strip():
            continue
        
        data = json.loads(line)
        
        print(f"\n=== 记录 {line_num} ===")
        
        ct = data.get('completion_tokens', [])
        act = data.get('all_completion_tokens', [])
        pt = data.get('pruned_tokens', [])
        
        print(f"completion_tokens: {len(ct)} 个beam")
        print(f"  值: {ct}")
        print(f"  总和: {sum(ct) if isinstance(ct, list) else 'N/A'}")
        
        print(f"\nall_completion_tokens: {len(act)} 个beam")
        if isinstance(act, list) and len(act) > 0:
            if isinstance(act[0], list):
                print(f"  结构: 嵌套列表")
                print(f"  第一个子列表: {act[0]}")
                total = sum(sum(x) if isinstance(x, list) else x for x in act)
            else:
                print(f"  结构: 扁平列表")
                print(f"  值: {act}")
                total = sum(act)
            print(f"  总和: {total}")
        
        print(f"\npruned_tokens: {len(pt)} 个beam")
        print(f"  值: {pt}")
        print(f"  总和: {sum(pt) if isinstance(pt, list) else 'N/A'}")
        
        # 验证逻辑
        ct_sum = sum(ct) if isinstance(ct, list) else 0
        pt_sum = sum(pt) if isinstance(pt, list) else 0
        
        if isinstance(act, list):
            if isinstance(act[0], list) if len(act) > 0 else False:
                act_sum = sum(sum(x) if isinstance(x, list) else x for x in act)
            else:
                act_sum = sum(act)
        else:
            act_sum = 0
        
        print(f"\n验证:")
        print(f"  completion_tokens总和: {ct_sum}")
        print(f"  pruned_tokens总和: {pt_sum}")
        print(f"  all_completion_tokens总和: {act_sum}")
        print(f"  completion + pruned = {ct_sum + pt_sum}")
        print(f"  差异: {(ct_sum + pt_sum) - act_sum}")
        
        if line_num >= 3:  # 只看前3条
            break
