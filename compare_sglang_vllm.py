#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

"""
简单对比 sglang 和 vllm 推理性能的脚本
"""

import time
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
import sglang as sgl

from sal.config import Config
from sal.utils.parser import H4ArgumentParser
from sal.utils.data import get_dataset


def test_vllm_inference(config: Config, prompts: list[str], n_samples: int = 4):
    """使用 vllm 进行推理"""
    print("\n=== Testing VLLM ===")
    
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        n=n_samples,
        seed=config.seed,
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


def test_sglang_inference(config: Config, prompts: list[str], n_samples: int = 4):
    """使用 sglang 进行推理"""
    print("\n=== Testing SGLang ===")
    
    # 启动 sglang runtime
    runtime = sgl.Runtime(
        model_path=config.model_path,
        tp_size=1,
    )
    sgl.set_default_backend(runtime)
    
    # 定义生成函数
    @sgl.function
    def generate_response(s, prompt):
        s += prompt
        s += sgl.gen(
            "response",
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
    
    start_time = time.time()
    
    # 批量推理
    all_outputs = []
    for prompt in prompts:
        for _ in range(n_samples):
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
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    
    # 从 MATH-500 数据集采样10个问题
    config.num_samples = 10
    dataset = get_dataset(config)
    test_prompts = [config.system_prompt + "\n" + p for p in dataset["problem"]]
    
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Number of prompts: {len(test_prompts)}")
    print(f"Samples per prompt: {config.n}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    
    # 测试 VLLM
    vllm_time, vllm_tokens = test_vllm_inference(config, test_prompts, config.n)
    
    # 测试 SGLang
    sglang_time, sglang_tokens = test_sglang_inference(config, test_prompts, config.n)
    
    # 对比结果
    print("\n=== Comparison Summary ===")
    print(f"VLLM time: {vllm_time:.2f}s")
    print(f"SGLang time: {sglang_time:.2f}s")
    print(f"Speedup: {vllm_time/sglang_time:.2f}x")
    

if __name__ == "__main__":
    main()
