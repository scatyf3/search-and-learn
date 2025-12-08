import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
import time
import copy

def best_of_n_transformers_layerskip_hard(x, config: Config, llm, tokenizer, prm: PRM):
    exit_layer = 8  # 设定截断层数（只跑前8层）
    
    # 注意：硬截断模式下不需要 num_assist_tokens，因为我们不进行验证/纠错
    # num_assist_tokens = 5 
    
    # 2. 构造 Prompt 输入
    prompts = []
    for prompt in x["problem"]:
        full_prompt = config.system_prompt + "\n" + prompt
        prompts.append(full_prompt)

    # 初始化容器
    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]
    llm_gen_times = []

    # === [关键修改 1] 保存原始模型结构 ===
    # 我们假设模型架构是 Llama/Qwen/Mistral 风格，层存储在 llm.model.layers
    # 如果是其他架构（如 GPT-2/BLOOM），属性路径可能不同（例如 llm.transformer.h）
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        original_layers = llm.model.layers
        original_num_layers = llm.config.num_hidden_layers
    else:
        # 兼容性 fallback，根据实际模型结构调整
        raise ValueError("Current code only supports Llama-like architectures (llm.model.layers)")

    # 3. 外层循环：遍历 Batch 中的每个 Prompt
    for i, prompt in enumerate(prompts):
        t_problem_start = time.time()
        
        inputs = tokenizer(prompt, return_tensors='pt').to(llm.device)
        
        prompt_candidates = []
        prompt_lens = []

        # === [关键修改 2] 应用硬截断 ===
        # 将模型的 layers 切片，只保留前 exit_layer 层
        # 使用 try-finally 确保即使报错也能恢复模型
        try:
            # 1. 物理切片 layers
            llm.model.layers = original_layers[:exit_layer]
            # 2. 修改 config (部分 RoPE 计算依赖 config 中的层数)
            llm.config.num_hidden_layers = exit_layer
            
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
                    # 此时 llm.generate 只会跑前 exit_layer 层
                    # 然后直接进入 Norm 和 LM_Head
                    outputs = llm.generate(**inputs, **gen_kwargs)
                
                # 解码
                input_len = inputs.input_ids.shape[1]
                generated_ids = outputs[0][input_len:] # 切掉 prompt
                decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                prompt_candidates.append(decoded_text)
                prompt_lens.append(len(generated_ids))
                
        except Exception as e:
            print(f"Error during generation: {e}")
            raise e
            
        finally:
            # === [关键修改 3] 恢复模型结构 ===
            # 无论是否成功，必须将模型恢复原样，以免影响后续评估或下一次循环
            llm.model.layers = original_layers
            llm.config.num_hidden_layers = original_num_layers

        # 填入结果
        completions[i] = prompt_candidates
        completion_tokens[i] = prompt_lens

        t_problem_end = time.time()
        llm_gen_times.append(t_problem_end - t_problem_start)

    # 4. 完整性检查
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # 5. PRM 验证 (通常使用完整模型或专门的 Reward Model)
    # 注意：这里生成的文本质量取决于 exit_layer 的输出是否能直接被 LM Head 理解
    t_prm_start = time.time()
    scores = prm.score(x["problem"], completions, batch_size=config.prm_batch_size)

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