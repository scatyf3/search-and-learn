import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "facebook/layerskip-llama3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# 【关键约束】官方的 assistant_early_exit 目前不支持 Batching，必须单条跑
prompt = "Alice and Bob are playing a game of"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 基础参数
common_kwargs = {
    "do_sample": False,
    "max_new_tokens": 60, # 稍微长一点，体现投机优势
    "pad_token_id": tokenizer.eos_token_id,
}

print("-" * 60)
print(f"Device: {device}")
print("Mode: Speculative Decoding within single model (Self-Speculation)")
print("-" * 60)

# --- 1. 基准：完整模型 (Baseline) ---
print(">>> Running Standard Full Model...")
# 预热
model.generate(**inputs, max_new_tokens=2)
torch.cuda.synchronize()

start = time.perf_counter()
output_base = model.generate(**inputs, **common_kwargs)
torch.cuda.synchronize()
time_base = time.perf_counter() - start

# --- 2. 官方参数：assistant_early_exit ---
EXIT_LAYER = 4
NUM_SPECULATIVE_TOKENS = 5  # 【核心参数】每次投机 5 个 Token

print(f">>> Running with assistant_early_exit={EXIT_LAYER} (K={NUM_SPECULATIVE_TOKENS})...")

# 预热
model.generate(
    **inputs, 
    max_new_tokens=2, 
    assistant_early_exit=EXIT_LAYER, 
    num_assistant_tokens=NUM_SPECULATIVE_TOKENS
)
torch.cuda.synchronize()

start = time.perf_counter()

# 🔥 这里就是你要的官方用法
output_spec = model.generate(
    **inputs, 
    **common_kwargs,
    # 告诉模型：用第4层当草稿
    assistant_early_exit=EXIT_LAYER, 
    # 告诉模型：每次草稿生成 5 个，然后让 16 层一次性验证
    num_assistant_tokens=NUM_SPECULATIVE_TOKENS 
)
torch.cuda.synchronize()
time_spec = time.perf_counter() - start

# --- 结果对比 ---
print("-" * 60)
print(f"Standard Time: {time_base:.4f}s")
print(f"LayerSkip Time: {time_spec:.4f}s")
speedup = time_base / time_spec
print(f"🚀 Speedup: {speedup:.2f}x")

# 验证内容
text_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
text_spec = tokenizer.decode(output_spec[0], skip_special_tokens=True)

print("-" * 60)
if text_base == text_spec:
    print("✅ 内容完全一致 (投机采样验证成功)")
else:
    print("❌ 内容不一致 (逻辑异常)")

print(f"Text: {text_spec[:100]}...")