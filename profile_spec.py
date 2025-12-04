import torch
import time
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置模型路径 (这里使用 HuggingFace 官方 ID，你可以替换为本地路径)
# 注意：Llama 3 没有 3B 版本，这里假设你指的是 Llama 3.2 3B 和 1B
TARGET_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    print(f"正在加载目标模型: {TARGET_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    
    # 加载大模型 (Target Model)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"正在加载草稿模型: {DRAFT_MODEL_ID} ...")
    # 加载小模型 (Draft/Assistant Model)
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    return tokenizer, target_model, draft_model

def load_math_questions(num_samples=10):
    """加载 MATH-500 数据集并随机采样"""
    print(f"\n正在加载 HuggingFaceH4/MATH-500 数据集...")
    try:
        # 加载测试集
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = dataset["problem"]
        
        # 随机采样
        if len(problems) < num_samples:
            sampled_problems = problems
        else:
            sampled_problems = random.sample(problems, num_samples)
            
        print(f"✅ 成功采样 {len(sampled_problems)} 个数学问题。")
        return sampled_problems
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        print("⚠️ 将使用备用生成的简单数学问题进行测试。")
        # 备用问题
        return [f"Solve the following math problem: Calculate the integral of x^{i} + {i}x from 0 to 10." for i in range(1, num_samples + 1)]

def run_inference(name, model, tokenizer, inputs, assistant_model=None):
    # print(f"运行: {name}") 
    # 减少刷屏，只保留关键信息
    
    start_time = time.time()
    
    # 核心代码：如果有 assistant_model，transformers 会自动启用 Speculative Decoding
    output = model.generate(
        **inputs,
        assistant_model=assistant_model, # 关键参数
        max_new_tokens=200,              # 生成长度
        do_sample=True,                  # 采样模式
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 计算速度
    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    speed = generated_tokens / total_time
    
    # decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"  -> 生成 {generated_tokens} tokens, 耗时 {total_time:.2f}s, 速度 {speed:.2f} t/s")
    
    return speed, generated_tokens

def main():
    tokenizer, target_model, draft_model = load_models()
    
    # 采样 10 个问题
    N_SAMPLES = 10
    questions = load_math_questions(num_samples=N_SAMPLES)
    
    std_speeds = []
    spec_speeds = []
    
    print(f"\n🚀 开始 {N_SAMPLES} 轮 Math500 测试对比...\n")
    print(f"{'Sample':<8} | {'Standard (t/s)':<15} | {'Speculative (t/s)':<18} | {'Speedup':<10}")
    print("-" * 60)

    for i, prompt in enumerate(questions):
        # 构造 prompt，这里简单加上 Instruct 格式（如果模型需要 chat template 更好，这里简化处理）
        # Llama 3 通常建议用 chat template，这里直接由 tokenizer 处理 inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 1. 标准推理 (Standard Inference)
        speed_std, _ = run_inference(
            "Standard", target_model, tokenizer, inputs, assistant_model=None
        )
        std_speeds.append(speed_std)
        
        # 2. 投机采样 (Speculative Decoding)
        speed_spec, _ = run_inference(
            "Speculative", target_model, tokenizer, inputs, assistant_model=draft_model
        )
        spec_speeds.append(speed_spec)
        
        # 计算单次加速比
        ratio = speed_spec / speed_std
        print(f"{i+1:<8} | {speed_std:<15.2f} | {speed_spec:<18.2f} | {ratio:<10.2f}x")

    # 统计结果
    avg_std = np.mean(std_speeds)
    avg_spec = np.mean(spec_speeds)
    avg_ratio = avg_spec / avg_std # 总体加速比 (也可以计算 ratio 的平均值 np.mean(spec_speeds / std_speeds))
    mean_ratio = np.mean(np.array(spec_speeds) / np.array(std_speeds))

    print("\n" + "=" * 60)
    print("📊 最终统计结果")
    print("=" * 60)
    print(f"平均标准速度: {avg_std:.2f} tokens/sec")
    print(f"平均投机速度: {avg_spec:.2f} tokens/sec")
    print(f"平均加速比:   {mean_ratio:.2f}x")
    
    if mean_ratio > 1.0:
        print("\n✅ 投机采样在数学问题上带来了加速！")
    else:
        print("\n⚠️ 投机采样未带来加速。可能原因：")
        print("1. 数学推理逻辑性强，小模型难以准确预测大模型的复杂推理步骤（接受率低）。")
        print("2. 显卡负载或模型大小差异不足以抵消验证开销。")

if __name__ == "__main__":
    main()