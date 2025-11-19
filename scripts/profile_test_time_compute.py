#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performance profiling script for test-time compute approaches.
This script runs inference on a single sample and records detailed performance metrics.

However, trace.json is too large to be opened in Chrome Tracing... maybe use naive way
"""

import logging
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = load_prm(config)

    dataset = get_dataset(config)
    
    # Select only the first sample for profiling
    logger.info(f"Original dataset size: {len(dataset)}")
    dataset = dataset.select([0])
    logger.info("Selected only the first sample for profiling")
    logger.info(f"Sample problem: {dataset[0]['problem'][:100]}...")

    # Setup profiler output directory
    profiler_output_dir = Path(config.output_dir) / "profiler_results" if config.output_dir else Path("./profiler_results")
    profiler_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Profiler results will be saved to: {profiler_output_dir}")
    
    # Run inference with torch profiler
    logger.info("Starting performance profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            logger.info("Phase 1: Running model inference...")
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search with profiler",
                load_from_cache_file=False,
            )
        
        with record_function("scoring"):
            logger.info("Phase 2: Running scoring...")
            dataset = score(dataset, config)

    # Save profiler results
    profiler_trace_path = profiler_output_dir / f"{config.approach}_trace.json"
    prof.export_chrome_trace(str(profiler_trace_path))
    logger.info(f"Chrome trace saved to: {profiler_trace_path}")
    logger.info("You can view this trace at chrome://tracing/")
    
    # Print profiler summary to console
    logger.info("=== Performance Summary ===")
    print("\n🔥 Top 10 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n💾 Top 10 operations by memory usage:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    
    # Save detailed profiler summary to file
    profiler_summary_path = profiler_output_dir / f"{config.approach}_summary.txt"
    with open(profiler_summary_path, 'w') as f:
        f.write(f"Performance Analysis for {config.approach.upper()}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  - Model: {config.model_path}\n")
        f.write(f"  - Approach: {config.approach}\n")
        f.write(f"  - Batch size: {config.search_batch_size}\n")
        f.write(f"  - Number of samples (n): {config.n}\n")
        f.write(f"  - GPU memory utilization: {config.gpu_memory_utilization}\n")
        f.write(f"  - Number of GPUs: {num_gpus}\n\n")
        
        f.write("Top 20 operations by CUDA time:\n")
        f.write("-" * 40 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        f.write("\n\nTop 20 operations by CPU time:\n")
        f.write("-" * 40 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        
        f.write("\n\nTop 15 operations by memory usage:\n")
        f.write("-" * 40 + "\n")
        f.write(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=15))
        
        f.write("\n\nTop 15 operations by FLOPS:\n")
        f.write("-" * 40 + "\n")
        try:
            f.write(prof.key_averages().table(sort_by="flops", row_limit=15))
        except:
            f.write("FLOPS information not available\n")
    
    logger.info(f"Detailed summary saved to: {profiler_summary_path}")
    
    # Save the processed dataset
    save_dataset(dataset, config)
    
    # Print final statistics
    if torch.cuda.is_available():
        logger.info(f"GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    logger.info("🎉 Profiling completed successfully!")
    logger.info(f"📊 Results saved in: {profiler_output_dir}")
    logger.info(f"📈 Chrome trace: {profiler_trace_path}")
    logger.info(f"📋 Summary: {profiler_summary_path}")


if __name__ == "__main__":
    main()
