import torch
import numpy as np
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

def dynamic_model_scheduler(x, config: Config, llm, prm: PRM, model_1b=None, tokenizer_1b=None, model_3b=None, tokenizer_3b=None):
    """
    llm: 这里作为占位符，实际使用传入的 model_1b 和 model_3b
    """
    # 1. 确保模型已传入，避免在循环中重复加载
    if model_1b is None or model_3b is None:
        raise ValueError("Please load models outside the loop and pass them to fn_kwargs!")

    prompts = [config.system_prompt + "\n" + p for p in x["problem"]]
    
    # 初始化结果容器
    completions = [[] for _ in range(len(prompts))]
    step_scores = [[] for _ in range(len(prompts))]
    model_trace = [[] for _ in range(len(prompts))]

    max_steps = getattr(config, "max_steps", 10)
    score_threshold = getattr(config, "score_threshold", 0.8)
    
    # 停止符 ID (假设是 Llama tokenizer，\n 通常是 13 或类似，建议动态获取)
    # 这里简单起见，我们在 decode 后处理字符串
    
    for i, prompt in enumerate(prompts):
        cur_prompt = prompt
        # 初始默认使用 1B 模型
        cur_model = model_1b
        cur_tokenizer = tokenizer_1b
        
        for step in range(max_steps):
            # --- A. 生成一步 ---
            inputs = cur_tokenizer(cur_prompt, return_tensors="pt").to(cur_model.device)
            
            with torch.no_grad():
                # 生成最多 64 tokens，遇到换行符停止（模拟 Reasoning Step）
                output_ids = cur_model.generate(
                    **inputs, 
                    max_new_tokens=64, 
                    do_sample=False,
                    pad_token_id=cur_tokenizer.eos_token_id
                    # 理想情况应该加 stopping_criteria 遇到 \n 停止
                )
            
            # 截取新生成的 tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            gen_text = cur_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 简单的后处理：如果生成内容包含换行，只取第一行作为这一步
            if "\n" in gen_text:
                gen_text = gen_text.split("\n")[0]
            
            # 如果生成为空（比如直接生成了 EOS），结束循环
            if not gen_text.strip():
                break

            completions[i].append(gen_text)
            model_trace[i].append("1b" if cur_model is model_1b else "3b")
            
            # --- B. PRM 打分 ---
            # 构造 PRM 输入：通常 PRM 需要完整上下文
            # 注意：根据你的 PRM 实现，可能只需要 last step，也可能需要 full context
            # 这里假设 prm.score 接受 (problem, completions_list)
            
            # 这里的 gen_text 是当前这一步
            # 如果 PRM 需要以前所有的步骤，你需要拼接
            history_context = "".join(completions[i]) 
            
            # 调用 PRM
            score_result = prm.score([x["problem"][i]], [[history_context]])
            
            # --- C. 修复分数提取 (关键修复点) ---
            # score_result 通常结构是 [[score]] 或 [[score_token1, score_token2...]]
            raw_score = score_result[0] 
            
            # 递归解包，直到拿到 float
            while isinstance(raw_score, list) or isinstance(raw_score, np.ndarray) or isinstance(raw_score, torch.Tensor):
                if len(raw_score) > 0:
                    # 如果有多个分数（例如每个token的分数），通常取最后一个作为 step score
                    # 或者取第一个，取决于你的 PRM 库定义。这里假设取最后一个
                    raw_score = raw_score[-1]
                else:
                    raw_score = 0.0 # 空列表保护
                    break
            
            try:
                score = float(raw_score)
            except:
                score = 0.0 # 转换失败保护

            step_scores[i].append(score)
            
            # --- D. 动态调度逻辑 ---
            if score < score_threshold:
                # 分数低，下一步切换到强模型
                cur_model = model_3b
                cur_tokenizer = tokenizer_3b
            else:
                # 分数高，下一步切回（或保持）弱模型
                cur_model = model_1b
                cur_tokenizer = tokenizer_1b
            
            # 更新 Prompt 进入下一步
            cur_prompt += "\n" + gen_text

    # 聚合结果
    x["completions"] = completions
    x["step_scores"] = step_scores
    x["model_trace"] = model_trace
    # 将 list of steps 拼成完整答案
    x["pred"] = ["\n".join(c) for c in completions]
    
    # 计算聚合分数 (Mean/Min/Last)
    try:
        # 防止空列表报错
        x["agg_scores"] = [aggregate_scores(s, config.agg_strategy) if s else 0.0 for s in step_scores]
    except:
        x["agg_scores"] = [0.0] * len(step_scores)
        
    return x