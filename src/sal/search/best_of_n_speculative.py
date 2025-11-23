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

def best_of_n_speculative(x, config: Config, llm, prm: PRM):
    """
    Speculative decoding: draft model generates candidates, main model verifies and accepts/rejects tokens.
    Assumes config.draft_model_path is set.
    """
    # 加载主模型和draft模型
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path).to('cuda')
    draft_tokenizer = AutoTokenizer.from_pretrained(config.draft_model_path)
    draft_model = AutoModelForCausalLM.from_pretrained(config.draft_model_path).to('cuda')

    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]

    import time
    t_llm_start = time.time()
    for i, prompt in enumerate(prompts):
        # draft模型生成候选
        draft_inputs = draft_tokenizer(prompt, return_tensors='pt').to('cuda')
        draft_outputs = draft_model.generate(
            **draft_inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            num_return_sequences=config.n,
            do_sample=True
        )
        draft_candidates = [draft_tokenizer.decode(o, skip_special_tokens=True) for o in draft_outputs]
        # speculative: 主模型对draft结果逐token验证
        accepted = []
        for candidate in draft_candidates:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
            candidate_ids = tokenizer(candidate, return_tensors='pt').input_ids.to('cuda')
            # 只保留主模型概率最高的token序列（简化版）
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                # 逐token比对draft和主模型分布
                # 这里只做简单accept/reject，实际可用更复杂策略
                accepted_tokens = []
                for idx in candidate_ids[0]:
                    prob = torch.softmax(logits, dim=-1)[0, idx].item()
                    if prob > 0.01:  # 设定阈值
                        accepted_tokens.append(idx)
                    else:
                        break
                accepted_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
                accepted.append(accepted_text)
        completions[i] = accepted
        completion_tokens[i] = [len(tokenizer.encode(a)) for a in accepted]
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
