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


def _beam_search(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> tuple[list[Beam], dict]:
    # Initial sampling parameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )
    
    # 初始化计时器
    total_llm_time = 0.0
    total_prm_time = 0.0

    # beam search configuration
    # init with original prompts
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
                    completion_tokens=0,
                )
            )

    completed_beams: list[Beam] = []
    # beam search 迭代若干回合
    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        # 用属性标记是否被剪枝
        # 第一个iter都算
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        # 复制active beams直到数量达到config.n
        # 这里的意思就是prepare 容器 作为prompt和output
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )
        # 如果是最后一次迭代，则修改SamplingParams，不在\n\n处停止，而是生成到EOS
        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )
        # Build conversations for active beams
        # 按照chat模板构建
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0 # 如果不是第一个，继续
        add_generation_prompt = i == 0 #  如果是第一个，添加生成提示

        # tokenizer init
        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        # apply chat template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        # 何意味，还能提前看吗
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        # 生成k步？但好像和beam没有关系
        
        # 计时 LLM 生成
        t_llm_start = time.time()
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, 1
        )
        t_llm_end = time.time()
        total_llm_time += (t_llm_end - t_llm_start)

        # 遍历beam 
        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            # copy result的结果到beam
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

        # prm评分
        # 计时 PRM 评分
        t_prm_start = time.time()
        scores = prm.score(prompts, completions)
        t_prm_end = time.time()
        total_prm_time += (t_prm_end - t_prm_start)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        # 找出未完成的分支
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        # 若全部分支都完成，early stop
        if len(active_beams) == 0:
            break

        # Filter duplicate active beams
        # 去重，诶如何定义重复，这不是和deepprune的一样吗，不过可能是简单去重
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
        # 从n里选beam_width 个继续生成
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True


    # 分割线，退出loop
    # 这个设计里n的作用是啥，哦，总共的trace
    # 然后num_iter是一个额外控制扩展多少次的超参数
    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    timing_info = {
        'llm_time': total_llm_time,
        'prm_time': total_prm_time
    }
    return completed_beams, timing_info


# 为啥还有warpper...
def beam_search(examples, config: Config, llm: LLM, prm: PRM):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"beam_search called with {len(examples['problem'])} problems")
    logger.info(f"Input keys: {list(examples.keys())}")
    
    problems = examples["problem"]
    beam_results, timing_info = _beam_search(problems, config, llm, prm)
    
    # 提取计时信息
    llm_gen_time = timing_info['llm_time']
    prm_score_time = timing_info['prm_time']

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    completions = []
    pred = []
    completion_tokens = []
    scores = []
    num_problems = len(problems)

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
        completion_tokens.append([b.completion_tokens for b in beams])

    # 修改 examples 字典，而不是创建新字典
    examples["completions"] = completions
    examples["scores"] = scores
    examples["pred"] = pred
    examples["completion_tokens"] = completion_tokens
    examples["llm_gen_time"] = [llm_gen_time / num_problems] * num_problems
    examples["prm_score_time"] = [prm_score_time / num_problems] * num_problems
    
    logger.info(f"beam_search returning with keys: {list(examples.keys())}")
    logger.info(f"completions length: {len(examples['completions'])}")

    return examples
