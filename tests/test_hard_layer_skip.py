import time
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置模型和环境
checkpoint = "facebook/layerskip-llama3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device.upper()}")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token # Batching 必须设置

# 2. 加载完整模型
print(">>> 加载完整模型...")
model_full = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# 3. 制造物理截断模型 (Layer Skip)
print(">>> 制造截断版模型 (只保留前4层)...")
model_skip = copy.deepcopy(model_full)
EXIT_LAYER = 4
# 物理删除第4层之后的所有层，这是实现 Batching 加速的唯一办法
model_skip.model.layers = model_skip.model.layers[:EXIT_LAYER]
model_skip.config.num_hidden_layers = EXIT_LAYER

# 4. 准备 Batch 数据 (8条不同的提示词)
prompts = [
    "Alice and Bob are playing",
    "The capital of France is",
    "Python is a programming language",
    "The quick brown fox jumps",
    "Machine learning is fascinating",
    "To be or not to be",
    "I like to eat pizza",
    "Winter is coming"
]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

gen_kwargs = {
    "do_sample": False,
    "max_new_tokens": 30, # 稍微短一点，方便阅读
    "pad_token_id": tokenizer.eos_token_id
}

print("-" * 60)
print(f"Batch Size: {len(prompts)} | Layer Skip: 保留前 {EXIT_LAYER} 层")
print("-" * 60)

# --- 运行完整模型 ---
print(">>> 正在生成: Full Model (16 Layers)...")
torch.cuda.synchronize()
start = time.perf_counter()
outputs_full = model_full.generate(**inputs, **gen_kwargs)
torch.cuda.synchronize()
time_full = time.perf_counter() - start

# --- 运行截断模型 ---
print(f">>> 正在生成: Truncated Model ({EXIT_LAYER} Layers)...")
torch.cuda.synchronize()
start = time.perf_counter()
outputs_skip = model_skip.generate(**inputs, **gen_kwargs)
torch.cuda.synchronize()
time_skip = time.perf_counter() - start

# --- 解码文本 ---
text_full = tokenizer.batch_decode(outputs_full, skip_special_tokens=True)
text_skip = tokenizer.batch_decode(outputs_skip, skip_special_tokens=True)

# --- 打印详细对比结果 ---
print("\n" + "=" * 80)
print(" " * 30 + "生成内容对比")
print("=" * 80)

for i in range(len(prompts)):
    print(f"\n[Sample {i+1}]: Input = '{prompts[i]}'")
    print("-" * 40)
    print(f"🔴 Full (16层): {text_full[i].replace(prompts[i], '...').strip()}")
    print(f"🟢 Skip ( 4层): {text_skip[i].replace(prompts[i], '...').strip()}")

print("\n" + "=" * 80)
print(f"Full Batch 耗时: {time_full:.4f}s")
print(f"Skip Batch 耗时: {time_skip:.4f}s")
speedup = time_full / time_skip
print(f"🚀 真实加速倍率: {speedup:.2f}x")
print("=" * 80)