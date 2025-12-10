#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

"""
使用 SGLang 进行推理性能测试
"""

import time
import sglang as sgl
from datasets import load_dataset

# Hardcoded parameters
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
TEMPERATURE = 0.8
TOP_P = 1.0
MAX_TOKENS = 2048
N_SAMPLES = 4
NUM_PROBLEMS = 10


def test_sglang_inference(prompts: list[str]):
    """使用 sglang 进行推理"""
    print("\n=== Testing SGLang ===")
    
    # 启动 sglang runtime
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.7,  # 减少静态显存占用
        max_running_requests=16,  # 限制并发请求数
        disable_cuda_graph=False,  # 保持CUDA graph以获得更好性能
    )
    sgl.set_default_backend(runtime)
    
    # 定义生成函数
    @sgl.function
    def generate_response(s, prompt):
        s += prompt
        s += sgl.gen(
            "response",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
    
    start_time = time.time()
    
    # 批量推理
    all_outputs = []
    for prompt in prompts:
        for _ in range(N_SAMPLES):
            state = generate_response.run(prompt=prompt)
            all_outputs.append(state["response"])
    
    end_time = time.time()
    
    # 估算 token 数（简单用字符数/4估算）
    total_tokens = sum(len(output) // 4 for output in all_outputs)
    elapsed = end_time - start_time
    
    print(f"SGLang - Time: {elapsed:.2f}s")
    print(f"SGLang - Total tokens (estimated): {total_tokens}")
    print(f"SGLang - Throughput: {total_tokens/elapsed:.2f} tokens/s")
    
    # 打印第一个结果示例
    print(f"\nExample output (first prompt, first sample):")
    print(all_outputs[0][:200] + "...")
    
    runtime.shutdown()
    
    return elapsed, total_tokens


def main():
    # 从 MATH-500 数据集采样10个问题
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.select(range(NUM_PROBLEMS))
    
    system_prompt = "Solve the following math problem step by step."
    test_prompts = [system_prompt + "\n" + p for p in dataset["problem"]]
    
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: HuggingFaceH4/MATH-500")
    print(f"Number of prompts: {len(test_prompts)}")
    print(f"Samples per prompt: {N_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max tokens: {MAX_TOKENS}")
    
    # 测试 SGLang
    sglang_time, sglang_tokens = test_sglang_inference(test_prompts)
    
    # 保存结果
    with open("sglang_benchmark_result.txt", "w") as f:
        f.write(f"SGLang Benchmark Results\n")
        f.write(f"Time: {sglang_time:.2f}s\n")
        f.write(f"Total tokens (estimated): {sglang_tokens}\n")
        f.write(f"Throughput: {sglang_tokens/sglang_time:.2f} tokens/s\n")
    
    print("\nResults saved to sglang_benchmark_result.txt")

if __name__ == "__main__":
    main()
