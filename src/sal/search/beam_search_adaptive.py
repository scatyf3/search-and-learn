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
import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config



from .utils import Beam, build_conv, generate_k_steps, last

logger = logging.getLogger()
from sal.utils.score import aggregate_scores


ADAPTIVE_THRESHOLD = 0.6 

def _beam_search(batch_of_prompts, config: Config, llm_large: LLM, llm_small: LLM, prm) -> list[Beam]:
    # Initial sampling parameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )
    
    # 默认起手用小模型
    current_llm = llm_small 

    # --- Beam Initialization ---
    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    completed_beams: list[Beam] = []
    
    # For Debug
    config.num_iterations = 3 
    
    # --- Main Loop ---
    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        
        # 1. Filter Active Beams
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # 2. Duplicate if needed (Ensure size config.n)
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams

        # 3. Last Iteration Adjustment
        if i == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        # 4. Build Conversation & Template (PRE-GENERATION)
        # 这一步构建的是生成前的 Prompt 状态，Fallback 时可以直接复用这个 templated_convs
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        # Tokenizer using current LLM (Assuming tokenizer is compatible/shared or swappable)
        # 注意：如果大小模型 tokenizer 不同，切换模型时可能需要重新 tokenize，这里假设通用
        tokenizer = current_llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
            
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead

        # ============================================================
        # Adaptive Compute Block Start
        # ============================================================
        
        # [Step A] Snapshot: 记录生成前的文本长度，以便后续回滚
        # 此时 beam.current_text 还是干净的
        previous_text_lengths = [len(b.current_text) for b in active_beams]
        
        # [Step B] First Pass Generation (Default: Small Model)
        # 即使上一轮切到了 Large，理论上每一轮开始都应该尝试切回 Small 以节省算力
        current_llm = llm_small 
        
        gen_results = generate_k_steps(
            templated_convs, lookahead, current_llm, sampling_params, 1
        )

        # [Step C] Tentative Update (尝试更新 Beam)
        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            # 暂存这次生成的内容，方便 debug
            beam.last_new_text = gen_result.next_texts[0]
            
            # 更新状态
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0] # <--- 文本变长了
            beam.history.append(beam.next_texts[0]) # <--- History 变长了
            
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        # [Step D] Scoring (初次评分)
        scores = prm.score(prompts, completions)
        
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]
        
        # 展平分数以获取本轮最佳分 (Batch Level)
        # 注意 agg_scores 结构是 [[scalar], [scalar]...]
        flat_agg_scores = [s[0] for s in agg_scores]
        current_best_score = max(flat_agg_scores) if flat_agg_scores else 0.0

        print(f"[DEBUG][iter {i}] Model: SMALL | Best Score: {current_best_score:.4f}")

        # [Step E] Fallback Decision (核心决策)
        # 只有当分数不达标，且当前确实是用的小模型时，才触发回滚
        if current_best_score < ADAPTIVE_THRESHOLD:
            print(f"⚠️ [Fallback Triggered] Score {current_best_score:.4f} < {ADAPTIVE_THRESHOLD}. Rolling back to LARGE model...")
            
            # --- 1. Rollback Beams ---
            for idx, beam in enumerate(active_beams):
                # 恢复文本长度
                beam.current_text = beam.current_text[:previous_text_lengths[idx]]
                # 恢复 history (弹出刚刚加进去的那一项)
                if beam.history:
                    beam.history.pop()
                # 恢复 token 计数 (简单减去刚刚生成的 token 数)
                beam.completion_tokens -= gen_results[idx].completion_tokens
            
            # --- 2. Switch Model ---
            current_llm = llm_large
            
            # --- 3. Re-generate ---
            # 直接复用 templated_convs，因为那是生成前的输入，没变过
            gen_results = generate_k_steps(
                templated_convs, lookahead, current_llm, sampling_params, 1
            )
            
            # --- 4. Re-update Beams (使用大模型结果) ---
            prompts, completions = [], []
            for beam, gen_result in zip(active_beams, gen_results, strict=True):
                # 覆盖刚才的记录
                beam.last_new_text = gen_result.next_texts[0]
                
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                beam.completion_tokens += gen_result.completion_tokens
                beam.current_text += beam.next_texts[0]
                beam.history.append(beam.next_texts[0])
                
                prompts.append(beam.prompt)
                completions.append([beam.current_text])
                
                print(f"[DEBUG][iter {i}][RE-GEN] New content: {beam.next_texts[0]}")

            # --- 5. Re-score ---
            scores = prm.score(prompts, completions)
            agg_scores = [
                [aggregate_scores(s, config.agg_strategy) for s in score]
                for score in scores
            ]
            
            flat_agg_scores = [s[0] for s in agg_scores]
            current_best_score = max(flat_agg_scores) if flat_agg_scores else 0.0
            print(f"[DEBUG][iter {i}] Model: LARGE | New Best Score: {current_best_score:.4f}")

        # ============================================================
        # Adaptive Compute Block End
        # ============================================================

        # 5. Check for Completions (EOS / Length)
        # 这部分逻辑保持不变，因为 beam 状态已经是最终确定版（无论来自小还是大模型）
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
             if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)

        # 6. Save Scores to Beams
        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0] # 保存 Raw Score

        # 7. Filter agg_scores for active beams only
        # 找出未完成的分支对应的分数，用于后续排序剪枝
        agg_scores = [
            agg_scores[k] for k, b in enumerate(active_beams) if not b.completed
        ]
        
        # Debug Print
        for idx, score in enumerate(scores):
            if idx < len(active_beams) and not active_beams[idx].completed:
                print(f"[DEBUG][iter {i}] Beam index: {active_beams[idx].index}, PRM score: {score}")
        print(f"[DEBUG][iter {i}] Agg scores (active only): {agg_scores}")

        # 8. Update Active Beams List
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping
        if len(active_beams) == 0:
            break

        # 9. Deduplication
        if config.filter_duplicates:
            unique_beam_dict = {}
            for k, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = k
            active_beams = [active_beams[k] for k in unique_beam_dict.values()]
            agg_scores = [agg_scores[k] for k in unique_beam_dict.values()]

        # 10. Pruning (Top-K Selection)
        # Flatten agg_scores for argsort: [[0.8], [0.5]] -> [0.8, 0.5]
        flat_active_scores = np.array([s[0] for s in agg_scores])
        
        # Calculate how many to keep
        k_to_keep = max(1, config.n // config.beam_width)
        
        # Get indices of top scores
        if len(flat_active_scores) > k_to_keep:
            top_indices = np.argsort(flat_active_scores)[-k_to_keep:]
        else:
            top_indices = range(len(flat_active_scores))

        # Mark pruned
        for k, beam in enumerate(active_beams):
            if k not in top_indices:
                beam.pruned = True
                
        # (Loop continues to next iteration...)

    # --- Post Processing ---
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    # Fill up if needed
    if len(completed_beams) != config.n:
        repeats = (config.n // len(completed_beams)) + 1
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams


# 为啥还有warpper...
def adaptive_beam_search(examples, config: Config, llm: LLM, prm):
    problems = examples["problem"]
    beam_results = _beam_search(problems, config, llm, prm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
