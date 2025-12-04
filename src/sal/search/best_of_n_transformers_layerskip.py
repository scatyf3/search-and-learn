import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
import time

def best_of_n_transformers_layerskip(x, config: Config, llm, tokenizer, prm: PRM):
    exit_layer = 8
    num_assist_tokens = 5
    # 2. 构造 Prompt 输入
    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    # 初始化容器
    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]
    llm_gen_times = []

    # 3. 外层循环：遍历 Batch 中的每个 Prompt
    for i, prompt in enumerate(prompts):
        t_problem_start = time.time()
        
        inputs = tokenizer(prompt, return_tensors='pt').to(llm.device)
        
        prompt_candidates = []
        prompt_lens = []

        # === 核心：串行循环 N 次 ===
        for _ in range(config.n):
            
            gen_kwargs = {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_return_sequences": 1,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = llm.generate(**inputs, **gen_kwargs)
            
            # 解码
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_len:] # 切掉 prompt
            decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            prompt_candidates.append(decoded_text)
            prompt_lens.append(len(generated_ids))
        # 填入结果
        completions[i] = prompt_candidates
        completion_tokens[i] = prompt_lens

        t_problem_end = time.time()
        llm_gen_times.append(t_problem_end - t_problem_start)

    # 4. 完整性检查
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # 5. PRM 验证
    t_prm_start = time.time()
    scores = prm.score(x["problem"], completions, batch_size=config.prm_batch_size, batched=config.prm_batch)

    t_prm_end = time.time()
    prm_score_time = t_prm_end - t_prm_start
    
    avg_prm_time = prm_score_time / len(x["problem"]) if len(x["problem"]) > 0 else 0

    # 维度检查
    if len(scores) != len(completions):
        print(f"⚠️ Warning: Dimension mismatch! Completions: {len(completions)}, Scores: {len(scores)}")

    # 计算聚合分数
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]
    
    # 选出最佳答案
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    # 6. 返回结果
    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    x["llm_gen_time"] = llm_gen_times 
    x["prm_score_time"] = [avg_prm_time] * len(x["problem"])
    
    return x