import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import dvts, best_of_n, beam_search
from sal.search.best_of_n_speculative import best_of_n_speculative
from sal.search.best_of_n_transformers import best_of_n_transformers
from sal.search.beam_search_adaptive import adaptive_beam_search
from sal.search.best_of_n_transformers_layerskip import best_of_n_transformers_layerskip
from sal.search.best_of_n_transformers_layerskip_hard import best_of_n_transformers_layerskip_hard

from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "beam_search_adaptive": adaptive_beam_search,
    "best_of_n_speculative": best_of_n_speculative,
    "best_of_n_transformers": best_of_n_transformers,
    "best_of_n_transformers_layerskip": best_of_n_transformers_layerskip,
    "best_of_n_transformers_layerskip_hard": best_of_n_transformers_layerskip_hard
}


def load_hf_model(model_path):
    """Helper to load HuggingFace model"""
    logger.info(f"Loading HF Model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def main():
    # 1. Setup & Initialization
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    logger.info(f"Starting execution with approach: {config.approach}")

    # 2. Load PRM
    logger.info(f"Loading real PRM from {config.prm_path}...")
    prm = load_prm(config)

    # 3. Initialize Function Arguments (基础参数)
    fn_kwargs = {
        "config": config,
        "prm": prm
    }

    # 4. Model Loading Strategy (从源代码迁移过来的核心逻辑)
    # =====================================================
    approach_fn = None

    # Case A: Speculative Decoding (HuggingFace Backend)
    if config.approach == "best_of_n_speculative":
        logger.info(f"Loading models for Speculative Decoding (HF)...")
        target_model, tokenizer = load_hf_model(config.model_path)
        draft_model, draft_tokenizer = load_hf_model(config.draft_model_path)
        
        fn_kwargs.update({
            "llm": target_model,
            "tokenizer": tokenizer,
            "draft_model": draft_model,
            "draft_tokenizer": draft_tokenizer
        })
        approach_fn = best_of_n_speculative

    # Case B: Adaptive Beam Search (vLLM Backend - One Big, One Small)
    elif config.approach == "beam_search_adaptive":
        logger.info(f"Loading dual models for Adaptive Beam Search (vLLM)...")
        
        # 注意：源代码这里用了 AutoModelForCausalLM，如果你的 adaptive_beam_search 需要 vLLM 对象，请确认此处
        # 根据源代码逻辑保持一致：
        llm_small = AutoModelForCausalLM.from_pretrained(config.model_path, device_map="cuda")
        llm_large = AutoModelForCausalLM.from_pretrained(config.draft_model_path, device_map="cuda")

        fn_kwargs.update({
            "llm_large": llm_large,
            "llm_small": llm_small
        })
        approach_fn = adaptive_beam_search

    # Case C: Standard Transformers Backend
    elif config.llm_backend == "transformers":
        logger.info(f"Loading single model (HuggingFace mode)...")
        model, tokenizer = load_hf_model(config.model_path)
        fn_kwargs.update({
            "llm": model,
            "tokenizer": tokenizer
        })
        
        if config.approach == "best_of_n":
            approach_fn = best_of_n_transformers
        elif config.approach == "best_of_n_transformers_layerskip":
            approach_fn = best_of_n_transformers_layerskip
        elif config.approach == "best_of_n_transformers_layerskip_hard":
            approach_fn = best_of_n_transformers_layerskip_hard
        elif config.approach == "best_of_n_transformers":
            from sal.search.best_of_n_transformers_wo_batching import best_of_n_transformers as bn_tf
            approach_fn = bn_tf
        else:
            raise ValueError(f"Approach {config.approach} not supported for transformers backend yet.")

    # Case D: Standard vLLM Backend (Default)
    else:
        logger.info(f"Initializing vLLM for {config.approach}...")
        num_gpus = torch.cuda.device_count()
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
        )
        fn_kwargs.update({"llm": llm})
        
        if config.approach == "beam_search":
            approach_fn = beam_search
        elif config.approach == "best_of_n":
            approach_fn = best_of_n
        else:
            raise ValueError(f"Unknown approach for vLLM: {config.approach}")

    # 5. Dataset Setup for Profiling
    # =====================================================
    dataset = get_dataset(config)
    
    # [Profiling] Select only the first sample
    logger.info(f"Original dataset size: {len(dataset)}")
    dataset = dataset.select([0])
    logger.info("Selected only the first sample for profiling")
    logger.info(f"Sample problem: {dataset[0]['problem'][:100]}...")

    # [Profiling] Setup profiler output directory
    profiler_output_dir = Path(config.output_dir) / "profiler_results" if config.output_dir else Path("./profiler_results")
    profiler_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Profiler results will be saved to: {profiler_output_dir}")

    run_batch_size = config.search_batch_size

    # 6. Run Inference with Profiler
    # =====================================================
    logger.info("Starting performance profiling...")
    
    # 启用 CPU 和 CUDA 的 profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        
        # Phase 1: Search / Inference
        with record_function("model_inference"):
            logger.info("Phase 1: Running model inference strategy...")
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=run_batch_size,
                fn_kwargs=fn_kwargs,  # 这里传入了上面动态构建的 fn_kwargs
                desc=f"Running search ({config.approach}) with profiler",
                load_from_cache_file=False,
            )
        
        # Phase 2: Scoring
        with record_function("scoring"):
            logger.info("Phase 2: Running scoring...")
            dataset = score(dataset, config)

    # 7. Save and Print Results
    # =====================================================
    
    # Save Chrome Trace
    profiler_trace_path = profiler_output_dir / f"{config.approach}_trace.json"
    prof.export_chrome_trace(str(profiler_trace_path))
    logger.info(f"Chrome trace saved to: {profiler_trace_path}")
    logger.info("You can view this trace at chrome://tracing/ or https://ui.perfetto.dev/")
    
    # Console Summary
    logger.info("=== Performance Summary ===")
    print("\n🔥 Top 10 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n💾 Top 10 operations by memory usage:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    
    # Detailed File Summary
    profiler_summary_path = profiler_output_dir / f"{config.approach}_summary.txt"
    num_gpus_available = torch.cuda.device_count()
    
    with open(profiler_summary_path, 'w') as f:
        f.write(f"Performance Analysis for {config.approach.upper()}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  - Model: {config.model_path}\n")
        f.write(f"  - Approach: {config.approach}\n")
        f.write(f"  - LLM Backend: {config.llm_backend}\n")
        f.write(f"  - Batch size: {config.search_batch_size}\n")
        f.write(f"  - Number of samples (n): {config.n}\n")
        f.write(f"  - GPUs Available: {num_gpus_available}\n\n")
        
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
    
    # Final Memory Stats
    if torch.cuda.is_available():
        logger.info(f"GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    logger.info("🎉 Profiling completed successfully!")

if __name__ == "__main__":
    main()
