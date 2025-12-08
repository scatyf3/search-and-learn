import time
import torch
import numpy as np
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

def best_of_n_speculative(x, config: Config, llm, tokenizer, draft_model, draft_tokenizer, prm: PRM):
    """
    Speculative decoding with Best-of-N.
    Forces num_return_sequences=1 inside a loop to support assisted_generation.
    """
    print("Running best_of_n_speculative...")
    
    # 1. 准备 Prompts
    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    # 初始化容器 [Batch_Size, N]
    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]

    # 修正点 1: 初始化耗时列表
    llm_gen_times = []

    # 2. 遍历 Batch (生成阶段)
    for i, prompt in enumerate(prompts):
        
        # 修正点 2: 开始计时 (针对当前这个问题)
        t_problem_start = time.time()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        
        prompt_candidates = []
        prompt_lens = []

        # 核心循环：生成 N 个候选项
        for _ in range(config.n):
            outputs = llm.generate(
                **inputs,
                assistant_model=draft_model,  # 启用 Speculative Decoding
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=1,       # 必须为 1
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码并去除 prompt
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_len:] # 切掉 prompt
            text_only = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            prompt_candidates.append(text_only)
            prompt_lens.append(len(generated_ids))

        completions[i] = prompt_candidates
        completion_tokens[i] = prompt_lens
        
        # 修正点 3: 结束计时并记录
        t_problem_end = time.time()
        llm_gen_times.append(t_problem_end - t_problem_start)

    # 完整性检查
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # 3. PRM 评分
    t_prm_start = time.time()
    scores = prm.score(x["problem"], completions, batch_size=config.prm_batch_size)
    t_prm_end = time.time()
    prm_score_time = t_prm_end - t_prm_start


    # 4. 聚合分数与预测
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]
    

    # 选出最佳
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]
    
    # 算平均每个问题的 PRM 耗时 (PRM通常是Batch的，所以这里算平均比较合理)
    avg_prm_time = prm_score_time / len(x["problem"]) if len(x["problem"]) > 0 else 0

    # 5. 返回结果
    '''
    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    但是之前的字段有problem, solution, level等等，为啥这里🈚️了
    '''
    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    
    x["llm_gen_time"] = llm_gen_times 
    x["prm_score_time"] = [avg_prm_time] * len(x["problem"])
    
    return x