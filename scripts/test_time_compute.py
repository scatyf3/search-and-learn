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

import torch
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

    import pickle
    timing_results = []
    dataset = get_dataset(config)
    # run inference using dataset.map
    # defined in sal.utils.data type: datasets.Dataset
    def timing_collect_fn(batch, config, llm, prm):
        result = approach_fn(batch, config, llm, prm)
        # 支持batched，收集每个样本的timing_result
        if isinstance(result, dict) and "timing_result" in result:
            timing_results.append(result["timing_result"])
        elif isinstance(result, list):
            for r in result:
                if isinstance(r, dict) and "timing_result" in r:
                    timing_results.append(r["timing_result"])
        return result

    dataset = dataset.map(
        timing_collect_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )
    # evaluate the results if specified
    dataset = score(dataset, config)

    pickle_filename = "timing_results_all.pkl"
    try:
        with open(pickle_filename, "wb") as f:
            pickle.dump(timing_results, f)
        logger.info(f"Saved all timing results to {pickle_filename}")
    except Exception as e:
        logger.error(f"Failed to save timing results: {e}")

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
