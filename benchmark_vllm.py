#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

"""
使用 vLLM 进行推理性能测试
"""

import time
import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

from vllm import LLM, SamplingParams
from datasets import load_dataset

# Hardcoded parameters
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
TEMPERATURE = 0.8
TOP_P = 1.0
MAX_TOKENS = 2048
N_SAMPLES = 4
SEED = 42
NUM_PROBLEMS = 10


def test_vllm_inference(prompts: list[str]):
    """使用 vllm 进行推理"""
    print("\n=== Testing VLLM ===")
    
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        dtype="half",
        )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=N_SAMPLES,
        seed=SEED,
    )
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    total_tokens = sum(len(output.outputs) * len(output.outputs[0].token_ids) for output in outputs)
    elapsed = end_time - start_time
    
    print(f"VLLM - Time: {elapsed:.2f}s")
    print(f"VLLM - Total tokens: {total_tokens}")
    print(f"VLLM - Throughput: {total_tokens/elapsed:.2f} tokens/s")
    
    # 打印第一个结果示例
    print(f"\nExample output (first prompt, first sample):")
    print(outputs[0].outputs[0].text[:200] + "...")
    
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
    
    # 测试 VLLM
    vllm_time, vllm_tokens = test_vllm_inference(test_prompts)
    
    # 保存结果
    with open("vllm_benchmark_result.txt", "w") as f:
        f.write(f"VLLM Benchmark Results\n")
        f.write(f"Time: {vllm_time:.2f}s\n")
        f.write(f"Total tokens: {vllm_tokens}\n")
        f.write(f"Throughput: {vllm_tokens/vllm_time:.2f} tokens/s\n")
    
    print("\nResults saved to vllm_benchmark_result.txt")

if __name__ == "__main__":
    main()
