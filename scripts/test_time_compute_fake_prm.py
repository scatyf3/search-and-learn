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

import logging
import os
import pickle
from datetime import datetime
import sys

import torch
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer  # 新增

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import dvts, best_of_n, beam_search
from sal.search.beam_search_dynamic import beam_search_dynamic
from sal.search.beam_search_dynamic_official import beam_search_dynamic_official
from sal.search.best_of_n_speculative import best_of_n_speculative
from sal.search.best_of_n_transformers import best_of_n_transformers
from sal.search.beam_search_adaptive import adaptive_beam_search
from sal.search.best_of_n_transformers_layerskip import best_of_n_transformers_layerskip
from sal.search.best_of_n_transformers_layerskip_hard import best_of_n_transformers_layerskip_hard
from sal.search.beam_search_dynamic_cosine import beam_search_dynamic_cosine

from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.models.fake_prm import FakePRM

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_recommended_dtype():
    """根据 GPU 计算能力自动推荐 dtype"""
    if not torch.cuda.is_available():
        return "auto"
    
    # 获取第一个 GPU 的计算能力
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    compute_cap = major + minor / 10
    
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"检测到 GPU: {gpu_name}")
    logger.info(f"计算能力: {major}.{minor}")
    
    # 计算能力 >= 8.0 支持 bfloat16 (A100, H100 等)
    # 计算能力 < 8.0 只支持 float16 (V100, RTX 8000 等)
    if compute_cap >= 8.0:
        logger.info("✅ 使用 bfloat16 (auto)")
        return "auto"  # 默认会使用 bfloat16
    else:
        logger.info("✅ 使用 float16 (half) - GPU 不支持 bfloat16")
        return "half"


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "beam_search_adaptive": adaptive_beam_search,
    "best_of_n_speculative": best_of_n_speculative,
    "best_of_n_transformers": best_of_n_transformers,
    "best_of_n_transformers_layerskip": best_of_n_transformers_layerskip,
    "best_of_n_transformers_layerskip_hard": best_of_n_transformers_layerskip_hard,
    "beam_search_dynamic": beam_search_dynamic,
    "beam_search_dynamic_official": beam_search_dynamic_official,
    "beam_search_dynamic_cosine": beam_search_dynamic_cosine    
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
    logger.info(f"Starting main execution with approach: {config.approach}")

    approach_fn = None
    num_gpus = torch.cuda.device_count()
    # 2. Load PRM (Process Reward Model)
    #if config.fake_prm:
    # logger.info("Using FakePRM for debugging...")
    # prm = FakePRM()
    # print(config)
    prm = load_prm(config)
    # 为啥prm path 无了
    logger.info(f"Loading real PRM from {config.prm_path}...")
    #prm = load_prm(config.prm_path)


    
    # 4. Initialize Function Arguments (基础参数)
    fn_kwargs = {
        "config": config,
        "prm": prm
    }

    # 5. Model Loading Strategy (核心分发逻辑)
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
        num_gpus = torch.cuda.device_count()
                
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
        # 根据 config 选择具体的 HF 函数
        if config.approach == "best_of_n":
            approach_fn = best_of_n_transformers
        elif config.approach == "best_of_n_transformers_layerskip":
            from sal.search.best_of_n_transformers_layerskip import best_of_n_transformers_layerskip
            approach_fn = best_of_n_transformers_layerskip
        # [新增] LayerSkip Hard 分支 ========================================
        elif config.approach == "best_of_n_transformers_layerskip_hard":
            from sal.search.best_of_n_transformers_layerskip_hard import best_of_n_transformers_layerskip_hard
            approach_fn = best_of_n_transformers_layerskip_hard
        # ===================================================================
        elif config.approach == "best_of_n_transformers":
            from sal.search.best_of_n_transformers_wo_batching import best_of_n_transformers
            approach_fn = best_of_n_transformers
        else:
            raise ValueError(f"Approach {config.approach} not supported for transformers backend yet.")

    # Case D: Standard vLLM Backend (Default)
    else:
        logger.info(f"Initializing vLLM for {config.approach}...")
        num_gpus = torch.cuda.device_count()
        
        # 自动检测 GPU 并选择合适的 dtype
        recommended_dtype = get_recommended_dtype()
        
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
            dtype=recommended_dtype,  # 根据 GPU 自动选择
        )
        fn_kwargs.update({"llm": llm})
        # 映射 approach 字符串到函数
        if config.approach == "beam_search":
            approach_fn = beam_search
        elif config.approach == "best_of_n":
            approach_fn = best_of_n
        elif config.approach == "beam_search_dynamic":
            approach_fn = beam_search_dynamic
        elif config.approach == "beam_search_dynamic_official":
            approach_fn = beam_search_dynamic_official
        elif config.approach == "beam_search_dynamic_cosine":
            approach_fn = beam_search_dynamic_cosine
        else:
            raise ValueError(f"Unknown approach for vLLM: {config.approach}")
    
    logger.info(f"Approach function set to: {approach_fn.__name__}")
    dataset = get_dataset(config)
    run_batch_size = config.search_batch_size
    logger.info(f"Dataset size before search: {len(dataset)}")
    logger.info(f"Dataset columns before search: {dataset.column_names}")
    logger.info(f"Starting search with batch size: {run_batch_size}")
    
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=run_batch_size,
        fn_kwargs=fn_kwargs,
        desc=f"Running search ({config.approach})",
        load_from_cache_file=False,
    )
    
    # 调试：打印数据集列名
    logger.info(f"Dataset columns after search: {dataset.column_names}")
    logger.info(f"First example keys: {list(dataset[0].keys())}")

    # evaluate the results if specified
    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()

'''
python scripts/test_time_compute.py recipes/best_of_n_with_transformers.yaml
'''