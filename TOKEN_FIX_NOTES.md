# Token统计修复说明

## 问题描述

在使用 `check_pruned_tokens.py` 检查生成的 JSONL 文件时，发现：
```
total_active_beam_tokens + total_pruned_tokens >> total_generated_tokens
```

例如：
```
total_active_beam_tokens: 2643
total_pruned_tokens: 681
active + pruned = 3324
total_generated_tokens: 907
差异: 2417
```

## 根本原因

在 `generate_k_steps` 函数中，使用了 lookahead 机制：
1. 每次调用 `generate_k_steps(templated_convs, lookahead=2, ...)`
2. 实际会生成 `lookahead_steps + 1 = 3` 次
3. 但只有第一步的文本会被添加到 `beam.current_text`
4. **问题**：原来的 `completion_tokens` 统计了所有3次生成的tokens，包括lookahead的tokens

这导致：
- `beam.completion_tokens` 包含了lookahead的tokens
- `total_active_beam_tokens` 统计的是包含lookahead的总数
- `total_generated_tokens` 统计的是实际保留在文本中的tokens
- 因此两者不匹配

## 解决方案

### 修改文件
- `/home/yf3005/search-and-learn/src/sal/search/utils.py`

### 具体修改

1. **在 `GenResult` dataclass 中添加新字段**：
```python
@dataclass
class GenResult:
    ...
    completion_tokens: int = 0  # 包含lookahead的总token数
    first_step_tokens: int = 0  # 只包含第一步的token数（不含lookahead）
```

2. **在生成过程中记录第一步的tokens**：
```python
for gen_result, output in zip(current_gen, llm_outputs):
    gen_text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    gen_result.completion_tokens += num_tokens
    
    if i == 0:
        # 只在第一步记录token数
        gen_result.first_step_tokens = num_tokens
        ...
```

3. **在返回Beam时使用 `first_step_tokens`**：
```python
first_step_tokens_sum = 0  # 只统计第一步的tokens
for j in range(beam_width):
    gen_result = gen_results[counter]
    first_step_tokens_sum += gen_result.first_step_tokens  # 不再使用completion_tokens
    ...

beam_result = Beam(
    ...
    completion_tokens=first_step_tokens_sum,  # 使用first_step_tokens_sum
)
```

## 预期效果

修改后，token统计应该满足：
```
total_active_beam_tokens + total_pruned_tokens = total_generated_tokens
```

这样可以正确验证token统计的一致性。

## 验证方法

1. 重新运行beam search生成新的JSONL文件
2. 使用 `check_pruned_tokens.py` 检查新文件
3. 确认所有记录的token数量都通过校验

## 示例计算

假设 lookahead=2，16个beams：

**修改前**（每个beam每次迭代）：
- 生成第1步：10 tokens → 添加到 current_text
- 生成第2步：8 tokens → lookahead（不添加到current_text）
- 生成第3步：7 tokens → lookahead（不添加到current_text）
- beam.completion_tokens = 10 + 8 + 7 = 25
- 16个beams总计 = 400 tokens ❌

**修改后**（每个beam每次迭代）：
- 生成第1步：10 tokens → 添加到 current_text
- 生成第2步：8 tokens → lookahead（不统计）
- 生成第3步：7 tokens → lookahead（不统计）
- beam.completion_tokens = 10
- 16个beams总计 = 160 tokens ✅

修改后的统计才能正确反映实际生成并保留的tokens数量。
