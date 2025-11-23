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

# 1. 强制禁用 vLLM V1 引擎 (必须在 import vllm 之前)
os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer  # 新增

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.search.best_of_n_transformers import best_of_n_transformers
from sal.search.best_of_n_speculative import best_of_n_speculative
from sal.search.dynamic_model_scheduler import dynamic_model_scheduler
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
    "best_of_n_transformers": best_of_n_transformers,
    "best_of_n_speculative": best_of_n_speculative,
    "dynamic_model_scheduler": dynamic_model_scheduler,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]
    num_gpus = torch.cuda.device_count()
    
    # 基础 fn_kwargs，所有方法都用
    prm = load_prm(config)
    fn_kwargs = {"config": config, "prm": prm}
    
    # 默认 batch size
    run_batch_size = config.search_batch_size

    # --- 模型加载逻辑分支 ---
    
    if config.approach == "dynamic_model_scheduler":
        logger.info("🚀 Loading models for Dynamic Scheduler (HuggingFace mode)...")
        
        # 1. 加载 Draft Model (1B)
        logger.info(f"Loading Draft Model: {config.draft_model_path}")
        tokenizer_1b = AutoTokenizer.from_pretrained(config.draft_model_path)
        model_1b = AutoModelForCausalLM.from_pretrained(
            config.draft_model_path, 
            torch_dtype=torch.float16, 
            device_map="auto" # 自动分配显存
        )
        
        # 2. 加载 Target Model (3B)
        logger.info(f"Loading Target Model: {config.model_path}")
        tokenizer_3b = AutoTokenizer.from_pretrained(config.model_path)
        model_3b = AutoModelForCausalLM.from_pretrained(
            config.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

        # 3. 更新参数字典
        fn_kwargs.update({
            "llm": None, # 占位
            "model_1b": model_1b,
            "tokenizer_1b": tokenizer_1b,
            "model_3b": model_3b,
            "tokenizer_3b": tokenizer_3b
        })
        
        # ⚠️ 动态调度逻辑包含复杂的Python控制流，强制 batch_size=1 以避免 padding 和对齐问题
        run_batch_size = 1
        logger.info("⚠️ Forcing batch_size=1 for dynamic scheduling.")

    elif config.approach in ["best_of_n_transformers", "best_of_n_speculative"]:
        # 这些方法可能在函数内部自己加载，或者还未适配外部加载
        logger.info(f"Running {config.approach} without external vLLM init.")
        llm = None
        fn_kwargs["llm"] = llm
        
    else:
        # vLLM based approaches (best_of_n, beam_search, etc.)
        logger.info(f"Initializing vLLM for {config.approach}...")
        llm = LLM(
            model=config.model_path,
            # 如果需要 N-gram 投机，可以在这里加 speculative_model="[ngram]"
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
        )
        fn_kwargs["llm"] = llm

    # --- 数据处理 ---

    dataset = get_dataset(config)

    logger.info(f"Starting search with batch size: {run_batch_size}")
    
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=run_batch_size,
        fn_kwargs=fn_kwargs,
        desc=f"Running search ({config.approach})",
        load_from_cache_file=False,
        remove_columns=dataset.column_names 
    )

    # evaluate the results if specified
    dataset = score(dataset, config)
    print(dataset)

    # --- 结果保存 ---
    pkl_folder = "pkl_results"
    os.makedirs(pkl_folder, exist_ok=True)
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_path.split('/')[-1]
    approach_name = config.approach
    pickle_filename = os.path.join(pkl_folder, f"timing_{model_name}_{approach_name}_{time_str}.pkl")
    
    try:
        with open(pickle_filename, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Saved all timing results to {pickle_filename}")
    except Exception as e:
        logger.error(f"Failed to save timing results: {e}")

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()

'''
python scripts/test_time_compute.py recipes/best_of_n_with_transformers.yaml
'''