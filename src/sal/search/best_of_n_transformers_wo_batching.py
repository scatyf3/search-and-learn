import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
import time

def best_of_n_transformers(x, config: Config, llm, tokenizer, prm: PRM):
    # 加载模型
    model = llm
    prm_batch_size = getattr(config, 'search_batch_size', 25)
    # 构造输入
    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]
    
    # 修正点 1: 创建一个列表来存储每个问题的独立耗时
    llm_gen_times = [] 

    # 外层循环：遍历每一个 Prompt
    for i, prompt in enumerate(prompts):
        
        # 修正点 2: 计时开始 (针对当前这个问题)
        t_problem_start = time.time()
        
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        prompt_candidates = []
        prompt_lens = []

        # 核心修改：串行循环 n 次
        for _ in range(config.n):
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码
            # 注意：如果不需要 prompt 本身，建议在这里切片 outputs[0][inputs.shape[1]:]
            # 这里保持原样
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            seq_len = outputs.shape[-1]
            
            prompt_candidates.append(decoded_text)
            prompt_lens.append(seq_len)

        # 将生成的 n 个串行结果存入 completions
        completions[i] = prompt_candidates
        completion_tokens[i] = prompt_lens
        
        # 修正点 3: 计时结束 (针对当前这个问题)
        t_problem_end = time.time()
        
        # 记录当前这个问题生成 N 个答案的总耗时
        llm_gen_times.append(t_problem_end - t_problem_start)

    # Check
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # PRM验证 (PRM通常是Batch处理，这里计算总时间是合理的)
    # 如果你想算平均每个问题的PRM耗时，可以除以 len(prompts)
    t_prm_start = time.time()
    scores = prm.score(x["problem"], completions, batch_size=prm_batch_size)
    t_prm_end = time.time()
    prm_score_time = t_prm_end - t_prm_start
    
    # 计算平均每个问题的 PRM 耗时 (可选，或者直接存总时间)
    avg_prm_time = prm_score_time / len(x["problem"]) if len(x["problem"]) > 0 else 0

    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    
    # 修正点 4: 直接使用记录好的独立时间列表
    x["llm_gen_time"] = llm_gen_times 
    
    # PRM 时间通常还是统一分配 (因为是Batch inference)，或者你可以填平均时间
    x["prm_score_time"] = [avg_prm_time] * len(x["problem"])
    
    return x