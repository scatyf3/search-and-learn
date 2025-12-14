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
import time

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

# 全局统计字典，记录每个prompt的token和beam数量
tokens_per_prompt = defaultdict(int)  # key: prompt(str), value: total tokens
beam_number_per_prompt = defaultdict(int)  # key: prompt(str), value: number of beams


def exp_decay_beam_width(iteration: int, config: Config, min_beam_width: int) -> int:
    """
    指数/幂函数衰减策略
    - temperature = 1.0 -> 线性衰减
    - temperature < 1.0 -> 凸函数 (前期衰减慢)
    - temperature > 1.0 -> 凹函数 (前期衰减快)
    """
    warmup_steps = getattr(config, 'beam_warmup_steps', 10)
    temperature = getattr(config, 'beam_decay_temperature', 1.0)
    
    if iteration < warmup_steps:
        return config.n
    
    progress = (iteration - warmup_steps) / (config.num_iterations - warmup_steps)
    decay_factor = progress ** (1 / temperature)
    reduction_amount = (config.n - min_beam_width) * decay_factor
    width = config.n - reduction_amount
    return max(min_beam_width, int(width))


def cosine_decay_beam_width(iteration: int, config: Config, min_beam_width: int) -> int:
    """
    余弦衰减策略
    公式: η_t = η_min + 1/2 * (η_max - η_min) * (1 + cos(T_cur / T_total * π))
    """
    warmup_steps = getattr(config, 'beam_warmup_steps', 10)
    
    if iteration < warmup_steps:
        return config.n
    
    eta_max = config.n
    eta_min = min_beam_width
    t_total = config.num_iterations - warmup_steps
    t_cur = iteration - warmup_steps
    
    if t_total <= 0:
        return min_beam_width
    
    cosine_factor = 1 + np.cos((t_cur / t_total) * np.pi)
    width = eta_min + 0.5 * (eta_max - eta_min) * cosine_factor
    return max(eta_min, int(width))


def _beam_search_dynamic(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> tuple[list[Beam], list[Beam]]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
        logprobs=1,
    )

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
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0
                )
            )

    completed_beams: list[Beam] = []
    all_beams = copy.deepcopy(beams)  # 保存所有beams的副本用于统计

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        min_beam_width = 2
        
        # 根据配置选择衰减策略，动态调整beam width
        strategy = getattr(config, 'beam_decay_strategy', 'exp')
        
        if strategy == "cosine":
            current_beam_width = cosine_decay_beam_width(i, config, min_beam_width)
        else:
            current_beam_width = exp_decay_beam_width(i, config, min_beam_width)
        
        # 同时动态调整n（每次采样的候选数）
        # 使用相同的衰减比例来调整n
        min_n = 1
        if strategy == "cosine":
            current_n = cosine_decay_beam_width(i, config, min_n)
        else:
            current_n = exp_decay_beam_width(i, config, min_n)
        # 确保current_n至少为1
        current_n = max(1, current_n)
        
        logger.info(f"Iteration {i}: Setting beam width to {current_beam_width}, n to {current_n}")
        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != current_beam_width:
            repeats = (current_beam_width // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {current_beam_width}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: current_beam_width]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != current_beam_width:
                raise ValueError(
                    f"Expected {current_beam_width} active beams, but got {len(active_beams)}"
                )

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=current_n,  # 使用动态调整的n
            )
        else:
            # 更新sampling_params使用动态的n
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stop=["\n\n"],
                include_stop_str_in_output=True,
                n=current_n,  # 使用动态调整的n
                logprobs=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, current_n  # 使用动态的n
        )

        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            
            # 统计每个prompt的token数和beam数量
            tokens_per_prompt[beam.prompt] += gen_result.completion_tokens
            beam_number_per_prompt[beam.prompt] += 1

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        scores = prm.score(prompts, completions)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(current_beam_width // config.beam_width):
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: current_beam_width]
    else:
        completed_beams = completed_beams[: current_beam_width]

    if len(completed_beams) != current_beam_width:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (current_beam_width // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {current_beam_width}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: current_beam_width]
        ]
        completed_beams = extended_completed_beams

    # 返回完成的beams和所有beams（用于统计pruned信息）
    return completed_beams, beams


def beam_search_dynamic(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]

    start_time = time.perf_counter()
    beam_results, all_beams = _beam_search_dynamic(problems, config, llm, prm)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    completions = []
    pred = []
    completion_tokens = []
    scores = []
    beam_counts_total = []
    num_problems = len(problems)
    time_per_problem = total_time / num_problems
    
    # Token统计
    total_tokens_all_beams = sum(tokens_per_prompt.values())
    total_active_beam_tokens = sum(b.completion_tokens for b in beam_results)
    total_pruned_tokens = sum(b.completion_tokens for b in all_beams if b.pruned)
    
    for p in problems:
        beams = grouped_results[p]
        completions_i = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred_i = completions_i[np.argmax(agg_scores)]
        completions.append(completions_i)
        scores.append([b.all_scores for b in beams])
        pred.append(pred_i)
        completion_tokens.append(int(tokens_per_prompt[p]))
        beam_counts_total.append(int(beam_number_per_prompt[p]))
        logger.info(f"Total tokens for problem is {tokens_per_prompt[p]}")
        logger.info(f"Number of beams for problem is {beam_number_per_prompt[p]}")

    # 修改 examples 字典
    examples["completions"] = completions
    examples["scores"] = scores
    examples["pred"] = pred
    examples["completion_tokens"] = completion_tokens
    examples["beam_counts_total"] = beam_counts_total
    examples["total_time_beam_search"] = [time_per_problem] * num_problems
    
    # Token统计信息
    examples["total_generated_tokens"] = [total_tokens_all_beams] * num_problems
    examples["total_active_beam_tokens"] = [total_active_beam_tokens] * num_problems
    examples["total_pruned_tokens"] = [total_pruned_tokens] * num_problems
    
    # 打印统计信息
    logger.info(f"\n=== Token Statistics ===")
    logger.info(f"总生成token数: {total_tokens_all_beams}")
    logger.info(f"最终activate beam总token数: {total_active_beam_tokens}")
    logger.info(f"被prune的beam总token数: {total_pruned_tokens}")
    logger.info(f"平均每个问题的token数: {total_tokens_all_beams / num_problems:.2f}")
    logger.info(f"======================\n")

    return examples