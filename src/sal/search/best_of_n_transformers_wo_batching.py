import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
import time

def best_of_n_transformers(x, config: Config, llm, tokenizer, prm: PRM):
    # 加载模型 (注意：这里直接使用传入的 llm 对象，不要重新加载)
    model = llm

    # 构造输入
    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]

    t_llm_start = time.time()
    
    # 外层循环：遍历每一个 Prompt
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        prompt_candidates = []
        prompt_lens = []

        # === 核心修改：串行循环 n 次 ===
        for _ in range(config.n):
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=1,  # <--- 强制每次只生成 1 条
                do_sample=True
            )
            
            # outputs shape 为 [1, seq_len]，取 outputs[0]
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            seq_len = outputs.shape[-1]
            
            prompt_candidates.append(decoded_text)
            prompt_lens.append(seq_len)

        # 将生成的 n 个串行结果存入 completions
        completions[i] = prompt_candidates
        completion_tokens[i] = prompt_lens

    t_llm_end = time.time()
    llm_gen_time = t_llm_end - t_llm_start

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # PRM验证 (保持不变)
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
