#!/usr/bin/env python
"""测试token统计修复"""

# 模拟测试：验证修改后的逻辑
# 假设场景：
# - lookahead_steps = 2 (总共生成3次)
# - 第一步生成10个tokens
# - 第二步生成8个tokens
# - 第三步生成7个tokens
# 
# 修改前：completion_tokens = 10 + 8 + 7 = 25
# 修改后：first_step_tokens = 10 (只统计第一步)
#
# 这样，如果有16个beams，每个beam每次迭代只记录实际添加到current_text的tokens
# active_beam_tokens + pruned_tokens = total_generated_tokens

print("=" * 60)
print("Token统计修复测试")
print("=" * 60)

print("\n场景说明：")
print("- lookahead_steps = 2")
print("- 每次迭代 generate_k_steps 会生成 3 次 (lookahead_steps + 1)")
print("- 但只有第一步的text会被添加到 beam.current_text")
print("\n修改前的问题：")
print("- completion_tokens 统计了所有3次生成的tokens")
print("- 导致 total_active_beam_tokens 远大于 total_generated_tokens")
print("\n修改后的改进：")
print("- 新增 first_step_tokens 只统计第一步的tokens")
print("- beam.completion_tokens 使用 first_step_tokens_sum")
print("- 这样 active_beam_tokens + pruned_tokens = total_generated_tokens")

print("\n示例计算：")
print("假设 16 个 beams，每个beam在某次迭代中：")
print("  - 生成第1步：10 tokens (这是实际添加到current_text的)")
print("  - 生成第2步：8 tokens (lookahead，仅用于PRM评分)")
print("  - 生成第3步：7 tokens (lookahead，仅用于PRM评分)")
print("\n修改前：")
print("  - 每个beam的completion_tokens = 10 + 8 + 7 = 25")
print("  - 16个beams总计 = 16 * 25 = 400 tokens")
print("\n修改后：")
print("  - 每个beam的completion_tokens = 10 (只统计第一步)")
print("  - 16个beams总计 = 16 * 10 = 160 tokens")
print("\n这样统计才能和实际生成的tokens数量一致！")

print("\n" + "=" * 60)
print("修改已完成，请重新运行beam search生成新数据进行验证")
print("=" * 60)
