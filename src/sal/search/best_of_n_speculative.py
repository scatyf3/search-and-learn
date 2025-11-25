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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores


def best_of_n_speculative(x, config: Config, llm, prm: PRM, tokenizer=None, draft_model=None, draft_tokenizer=None):
    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]

    import time
    t_llm_start = time.time()
    # 遍历每一个 Prompt
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        
        prompt_completions = []
        prompt_token_lens = []

        # === 核心修改：循环 config.n 次，每次生成 1 条 ===
        for _ in range(config.n):
            outputs = llm.generate(
                **inputs,
                assistant_model=draft_model,  # 开启 Speculative Decoding
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=1,       # <--- 必须强制为 1
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码当前这一条
            # outputs[0]包含了 input_ids，我们需要去掉 prompt 部分
            # 或者直接 decode 后处理
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 简单的后处理：去掉 Prompt (如果模型输出包含 Prompt)
            # 注意：HuggingFace generate 输出通常包含 prompt，需要切掉
            # 这里取决于你的模型和 tokenizer 行为，通常建议如下操作：
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_len:] 
            text_only_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            prompt_completions.append(text_only_completion)
            prompt_token_lens.append(len(generated_ids))

        completions[i] = prompt_completions
        completion_tokens[i] = prompt_token_lens

    t_llm_end = time.time()
    llm_gen_time = t_llm_end - t_llm_start

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # PRM验证
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
    x["llm_gen_time"] = [llm_gen_time] * len(x["problem"])
    x["prm_score_time"] = [prm_score_time] * len(x["problem"])
    return x
