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


def _beam_search_dynamic(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> list[Beam]:
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

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        min_beam_width = 2
        
        # 使用温度参数控制非线性衰减
        # temperature < 1: 更激进的衰减（前期快速减少）
        # temperature = 1: 线性衰减
        # temperature > 1: 更保守的衰减（后期才快速减少）
        temperature = getattr(config, 'beam_decay_temperature', 1.0)
        warmup_steps = getattr(config, 'beam_warmup_steps', 10)
        
        if i < warmup_steps:
            # 预热阶段，保持最大 beam width
            current_beam_width = config.n
        else:
            # === Cosine Decay 策略开始 ===
            # T_cur: 当前处于衰减期的第几步
            t_cur = i - warmup_steps
            
            # T_total: 衰减期的总步数
            # 注意：减 1 是为了让最后一次迭代 (i == num_iterations - 1) 时进度正好为 100%，从而达到 min_beam_width
            t_total = config.num_iterations - 1 - warmup_steps
            
            # 安全检查：防止 warmup 设置过大导致 t_total <= 0
            if t_total <= 0:
                current_beam_width = min_beam_width
            else:
                # 对应公式参数
                eta_max = config.n          # 最大宽度
                eta_min = min_beam_width    # 最小宽度
                
                # 公式: η_t = η_min + 1/2 * (η_max - η_min) * (1 + cos(T_cur / T_total * π))
                # 引用自图片公式
                cosine_factor = 1 + np.cos((t_cur / t_total) * np.pi)
                current_beam_width = eta_min + 0.5 * (eta_max - eta_min) * cosine_factor
            
            # 取整并确保下限
            current_beam_width = int(current_beam_width)
            current_beam_width = max(min_beam_width, current_beam_width)
            # === Cosine Decay 策略结束 ===
        
        logger.info(f"Iteration {i}: Setting beam width to {current_beam_width}")
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
                n=1,
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
            templated_convs, lookahead, llm, sampling_params, 1
        )

        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

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

    return completed_beams


def beam_search_dynamic_cosine(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]

    start_time = time.perf_counter()
    beam_results = _beam_search_dynamic(problems, config, llm, prm)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [],"total_time_beam_search": []}
    num_problems = len(problems)
    time_per_problem = total_time / num_problems
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

        # Add average time per problem
        results["total_time_beam_search"].append(time_per_problem)

    return results