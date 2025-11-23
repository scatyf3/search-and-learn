#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

"""
基于分步PRM分数的动态模型调度：
- draft模型（1B）生成每一步
- PRM对每一步打分
- 若分数高于阈值，下一步继续用1B模型
- 若分数低于阈值，下一步切换为3B模型生成
- 支持 transformers 格式模型
"""

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

def dynamic_model_scheduler(x, config: Config, llm, prm: PRM):
    # 加载两个模型
    tokenizer_1b = AutoTokenizer.from_pretrained(config.draft_model_path)
    model_1b = AutoModelForCausalLM.from_pretrained(config.draft_model_path).to('cuda')
    tokenizer_3b = AutoTokenizer.from_pretrained(config.model_path)
    model_3b = AutoModelForCausalLM.from_pretrained(config.model_path).to('cuda')

    prompts = [config.system_prompt + "\n" + p for p in x["problem"]]
    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]
    step_scores = [[] for _ in range(len(prompts))]
    model_trace = [[] for _ in range(len(prompts))]

    max_steps = config.max_steps if hasattr(config, "max_steps") else 10
    score_threshold = getattr(config, "score_threshold", 0.8)

    for i, prompt in enumerate(prompts):
        cur_prompt = prompt
        cur_model = model_1b
        cur_tokenizer = tokenizer_1b
        for step in range(max_steps):
            # 生成一步
            inputs = cur_tokenizer(cur_prompt, return_tensors="pt").to(cur_model.device)
            with torch.no_grad():
                output = cur_model.generate(**inputs, max_new_tokens=32, do_sample=False)
            gen_text = cur_tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            completions[i].append(gen_text)
            model_trace[i].append("1b" if cur_model is model_1b else "3b")
            # PRM打分
            score_result = prm.score([x["problem"][i]], [[gen_text]])
            # score_result: list[list[float]], 取第一个分数
            score = score_result[0][0] if isinstance(score_result[0], list) else score_result[0]
            step_scores[i].append(score)
            step_scores[i].append(score)
            # 判断是否切换模型
            if score < score_threshold:
                cur_model = model_3b
                cur_tokenizer = tokenizer_3b
            else:
                cur_model = model_1b
                cur_tokenizer = tokenizer_1b
            # 更新prompt
            cur_prompt += "\n" + gen_text
    # 聚合分数
    agg_scores = [aggregate_scores(s, config.agg_strategy) for s in step_scores]
    pred = ["".join(c) for c in completions]
    x["completions"] = completions
    x["step_scores"] = step_scores
    x["model_trace"] = model_trace
    x["pred"] = pred
    x["agg_scores"] = agg_scores
    return x
