import copy
import logging
from collections import defaultdict
import time
import math # Added for potential math operations

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

# Assuming these exist in your utils
from .utils import Beam, build_conv, generate_k_steps, last
from sal.utils.score import aggregate_scores

logger = logging.getLogger(__name__)

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
                    completed=False,
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                    logprobs=[]
                )
            )

    completed_beams: list[Beam] = []
    
    # -----------------------------------------------------------------------
    # [NEW] 获取衰减温度参数 (默认为 1.0，即线性衰减)
    # 建议在 Config 中添加这个字段: decay_temperature
    # -----------------------------------------------------------------------
    decay_temperature = getattr(config, 'decay_temperature', 1.0)
    logger.info(f"Using Decay Temperature: {decay_temperature}")

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        min_beam_width = 2
        
        # -----------------------------------------------------------------------
        # [NEW] 基于温度的非线性动态收敛逻辑
        # -----------------------------------------------------------------------
        # 1. 计算线性进度 (0.0 -> 1.0)
        linear_progress = i / config.num_iterations
        
        # 2. 应用温度控制收敛曲线
        # 公式: contraction_factor = progress ^ temperature
        # - Temp > 1.0: 曲线下凹，前期收敛慢 (保持宽 Beam)，后期快。 (更注重探索)
        # - Temp < 1.0: 曲线上凸，前期收敛快 (快速缩小 Beam)，后期慢。 (更注重速度)
        # - Temp = 1.0: 线性。
        contraction_factor = linear_progress ** decay_temperature
        
        # 3. 计算当前 Beam Width
        reduction_amount = (config.n - min_beam_width) * contraction_factor
        
        # 保护逻辑：前几步通常不收敛，除非温度设得极低
        if i < 5: 
            reduction_amount = 0

        current_beam_width = int(config.n - reduction_amount)
        current_beam_width = max(min_beam_width, current_beam_width)
        
        logger.info(f"Iteration {i}: Progress={linear_progress:.2f}, Factor={contraction_factor:.2f} -> Beam Width {current_beam_width}")
        # -----------------------------------------------------------------------

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != current_beam_width:
            repeats = (current_beam_width // len(active_beams)) + 1
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: current_beam_width]
            ]
            active_beams = extended_active_beams

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

        # Filter completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

        if config.filter_duplicates:
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = i
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # PRUNING using current_beam_width
        if len(agg_scores) > 0:
            # We want to keep 'current_beam_width' number of beams, 
            # but we also need to respect the division by config.beam_width logic from your original code
            # Assuming you want to keep 'current_beam_width' total candidates:
            keep_count = min(len(active_beams), current_beam_width)
            
            # Sort indices by score (ascending)
            sorted_indices = np.argsort(np.array(agg_scores).flatten())
            
            # Keep the top 'keep_count' indices
            top_indices = sorted_indices[-keep_count:]

            for idx, beam in enumerate(active_beams):
                if idx not in top_indices:
                    beam.pruned = True
        else:
             # Edge case: all beams pruned or completed in this step
             pass

    # Sort completed beams
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n] # Original logic used current_beam_width here, but usually we want top N overall
    else:
        completed_beams = completed_beams[: config.n]

    # Fill up if missing
    if len(completed_beams) < config.n: # Use config.n as target for final output
        target_size = config.n 
        if len(completed_beams) > 0:
            repeats = (target_size // len(completed_beams)) + 1
            extended_completed_beams = [
                copy.deepcopy(b) for b in (completed_beams * repeats)[: target_size]
            ]
            completed_beams = extended_completed_beams
        else:
             logger.warning("No completed beams found!")

    return completed_beams


def beam_search_dynamic(examples, config: Config, llm: LLM, prm: PRM):
    """
    Main entry point compatible with your profiler script
    """
    problems = examples["problem"]

    start_time = time.perf_counter()
    beam_results = _beam_search_dynamic(problems, config, llm, prm)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [],"total_time_beam_search": []}
    num_problems = len(problems)
    # Avoid division by zero
    time_per_problem = total_time / num_problems if num_problems > 0 else 0
    
    for p in problems:
        beams = grouped_results[p]
        if not beams:
             # Handle case with no results
             results["completions"].append([])
             results["scores"].append([])
             results["pred"].append("")
             results["completion_tokens"].append([])
             results["total_time_beam_search"].append(time_per_problem)
             continue
             
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        
        # Best prediction
        pred = completions[np.argmax(agg_scores)] if agg_scores else ""
        
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["total_time_beam_search"].append(time_per_problem)

    return results