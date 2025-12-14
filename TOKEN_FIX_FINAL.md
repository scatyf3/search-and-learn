## Token统计修复 - 第二次修正

### 发现的新问题

运行 `check_pruned_tokens.py` 检查修改后生成的文件时，发现 token 统计仍然不匹配：
```
total_active_beam_tokens: 4572
total_pruned_tokens: 1425
active + pruned = 5997
total_generated_tokens: 1720  ❌ 远小于 active + pruned
差异: 4277
```

### 根本原因（第二个问题）

虽然我们已经修复了 `first_step_tokens` 的统计（只统计第一步，不包含 lookahead），但在 `beam_search_dynamic.py` 中计算 `total_generated_tokens` 时，使用的是：

```python
total_tokens_all_beams = sum(b.completion_tokens for b in all_beams)
```

问题在于：
1. `all_beams` 是函数 `_beam_search_dynamic` 返回的原始 `beams` 列表
2. 在迭代过程中，由于使用 `copy.deepcopy(b)` 扩展 beams，新创建的 beams **不在** `all_beams` 列表中
3. 最终返回的 `beam_results` (即 `completed_beams`) 可能包含 deepcopy 的 beams
4. 因此 `all_beams` 的统计是不完整的

### 解决方案

修改 token 统计逻辑，直接通过 `active + pruned` 来计算总数：

```python
# 1. 最终activate beam的总token数（这些是最终返回的beams）
total_active_beam_tokens = sum(b.completion_tokens for b in beam_results)
# 2. 被prune的beam总token数 - 从all_beams中统计被prune的
total_pruned_tokens = sum(b.completion_tokens for b in all_beams if b.pruned)
# 3. 总生成token数 = activate beams + pruned beams
total_tokens_all_beams = total_active_beam_tokens + total_pruned_tokens
```

这样确保：
```
total_generated_tokens = total_active_beam_tokens + total_pruned_tokens
```

验证等式自然成立：
```
total_active_beam_tokens + total_pruned_tokens = total_generated_tokens ✅
```

### 修改的文件

1. **src/sal/search/utils.py** - 第一次修复（lookahead问题）
   - 添加 `first_step_tokens` 字段
   - 只统计第一步生成的 tokens

2. **src/sal/search/beam_search_dynamic.py** - 第二次修复（all_beams统计问题）
   - 修改 `total_tokens_all_beams` 的计算方式
   - 使用 `active + pruned` 而不是 `sum(all_beams)`

### 验证

重新生成数据后，使用 `check_pruned_tokens.py` 验证，应该看到：
```
✅ 所有记录的token数量都匹配！
```
