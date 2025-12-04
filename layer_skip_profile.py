import time
import torch
import copy
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 配置 ---
checkpoint = "facebook/layerskip-llama3.2-1B"
EXIT_LAYER = 4
NUM_SPEC_TOKENS = 5  # 每次猜5个
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在分析模型: {checkpoint}")
print(f"投机配置: Draft Layer={EXIT_LAYER}, K={NUM_SPEC_TOKENS}")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# 固定 Prompt 以便公平对比
prompt = "Alice and Bob are playing a game of chess"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
gen_kwargs = {
    "do_sample": False,
    "max_new_tokens": 100, # 跑长一点，让数据更稳
    "pad_token_id": tokenizer.eos_token_id,
    "use_cache": True
}

# 辅助函数：测速
def measure_speed(name, func, warmup=2):
    print(f"\n>>> 测试: {name}...")
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    # Timing
    start = time.perf_counter()
    output = func()
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    
    # Calculate TPS (Tokens Per Second)
    num_tokens = len(output[0]) - inputs['input_ids'].shape[1]
    tps = num_tokens / duration
    print(f"    耗时: {duration:.4f}s | 生成: {num_tokens} tokens | TPS: {tps:.2f} token/s")
    return tps, duration

# =======================================================
# 实验 1: 完整模型 (基准线)
# =======================================================
def run_full():
    return model.generate(**inputs, **gen_kwargs)

tps_full, time_full = measure_speed("完整模型 (Full 16 Layers)", run_full)

# =======================================================
# 实验 2: 纯草稿模型 (理论极限)
# =======================================================
# 我们制造一个物理截断的模型，看看如果不需要验证，它能跑多快
# 这代表了“如果命中率100%且零开销”时的速度上限
model_draft = copy.deepcopy(model)
model_draft.model.layers = model_draft.model.layers[:EXIT_LAYER]
model_draft.config.num_hidden_layers = EXIT_LAYER

def run_draft():
    return model_draft.generate(**inputs, **gen_kwargs)

tps_draft, time_draft = measure_speed(f"纯 Draft 模型 (Only {EXIT_LAYER} Layers)", run_draft)

del model_draft # 清理显存

# =======================================================
# 实验 3: 官方投机模式 (assistant_early_exit)
# =======================================================
def run_spec():
    return model.generate(
        **inputs, 
        **gen_kwargs,
        assistant_early_exit=EXIT_LAYER,
        num_assistant_tokens=NUM_SPEC_TOKENS
    )

tps_spec, time_spec = measure_speed("投机模式 (Speculative Decoding)", run_spec)

# =======================================================
# 📊 最终分析报告
# =======================================================
print("\n" + "="*60)
print("🔍 性能瓶颈分析报告")
print("="*60)

# 1. 计算理论上的层数比例
layer_ratio = 16 / EXIT_LAYER
print(f"1. 物理层数差异: 完整版是 Draft版的 {layer_ratio:.1f}倍")

# 2. 计算实际纯算力差异
raw_speedup = tps_draft / tps_full
print(f"2. 实际算力差距: Draft版 比 完整版 快 {raw_speedup:.2f}倍")
print(f"   (这意味着如果完全没有开销，你最高能加速 {raw_speedup:.2f}倍)")

# 3. 计算投机采样的实际表现
real_speedup = tps_spec / tps_full
print(f"3. 实际投机加速: {real_speedup:.2f}倍")

# 4. 损耗分析
efficiency = (real_speedup - 1) / (raw_speedup - 1) if raw_speedup > 1 else 0
print("-" * 60)
print(f"💡 结论验证:")

if raw_speedup > 2.0 and real_speedup < 1.3:
    print("✅ 验证成功：你的猜测是对的。")
    print("   现象：Draft 模型跑得飞快 (远超 Full)，但投机模式没跟上。")
    print("   原因：这说明大量的性能被消耗在 'Python 调度逻辑' 和 '验证失败的回退' 上。")
    print("        对于 Batch=1 的小模型，verify step 的开销占比太高了。")
elif real_speedup > 1.5:
    print("🎉 还可以：投机采样生效了，虽然没达到理论极限，但比基准快不少。")
else:
    print("❌ 异常：Draft 模型本身就不快（可能是 LM Head 瓶颈），所以投机也没法快。")

print("="*60)