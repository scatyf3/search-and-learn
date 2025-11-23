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

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores



def best_of_n_with_draft(x, config: Config, llm: LLM, prm: PRM):
    """
    先用 draft_llm 生成初步答案，再用主 PRM 验证评分。
    config.draft_model_path: draft模型路径（可选）
    """
    import time
    from vllm import LLM, SamplingParams
    from datetime import datetime

    # 1. 加载 draft_llm
    if hasattr(config, 'draft_model_path') and config.draft_model_path:
        draft_llm = LLM(model=config.draft_model_path,
                        gpu_memory_utilization=config.gpu_memory_utilization,
                        enable_prefix_caching=True,
                        seed=config.seed,
                        tensor_parallel_size=torch.cuda.device_count())
    else:
        draft_llm = llm  # 没有 draft_model_path 就用主模型

    # 2. 构造 prompt
    tokenizer = draft_llm.get_tokenizer()
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
    )

    # 3. draft_llm 生成
    t_draft_start = time.time()
    responses = draft_llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    t_draft_end = time.time()
    draft_gen_time = t_draft_end - t_draft_start

    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]
    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # 4. 主 PRM 验证 draft completions
    t_prm_start = time.time()
    scores = prm.score(x["problem"], completions)
    t_prm_end = time.time()
    prm_score_time = t_prm_end - t_prm_start

    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    x["draft_gen_time"] = [draft_gen_time] * len(x["problem"])
    x["prm_score_time"] = [prm_score_time] * len(x["problem"])
    x["timing_n"] = [config.n] * len(x["problem"])
    x["timing_batch_size"] = [config.search_batch_size] * len(x["problem"])
    x["timing_timestamp"] = [datetime.now().strftime("%Y%m%d_%H%M%S")] * len(x["problem"])
    return x
