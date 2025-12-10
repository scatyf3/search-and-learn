#!/usr/bin/env python
"""
批量评估所有生成的 completions.jsonl 文件（简化版）
"""

import os
import subprocess
import glob
import json
import re
from datetime import datetime

# ================= 配置区域 =================
CONDA_ENV_PATH = "/scratch/yf3005/qwen-math"
EVAL_SCRIPT = "/home/yf3005/Qwen2.5-Math/evaluation/evaluate.py"
RESULTS_DIR = "/home/yf3005/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_LOG = "/home/yf3005/search-and-learn/evaluation_results.txt"
OUTPUT_TSV = "/home/yf3005/search-and-learn/evaluation_summary.tsv"
# ===========================================

def get_jsonl_files(directory):
    """获取目录下所有的 completions.jsonl 文件"""
    pattern = os.path.join(directory, "*_completions.jsonl")
    files = glob.glob(pattern)
    files.sort()
    return files

def extract_config_from_file(file_path):
    """从文件第一行提取配置信息"""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('# CONFIG:'):
                config_json = first_line.replace('# CONFIG:', '').strip()
                config = json.loads(config_json)
                return config
    except Exception as e:
        print(f"   ⚠️ 无法读取配置: {e}")
    return {}

def extract_generation_times_and_tokens(file_path):
    """从数据文件中提取生成时间、PRM评分时间和token统计"""
    llm_times = []
    prm_times = []
    total_times = []
    completion_tokens_list = []
    total_completions = 0
    estimated_tokens = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    data = json.loads(line)
                    # 提取 llm_gen_time (新格式直接是数值)
                    if 'llm_gen_time' in data:
                        llm_time = data['llm_gen_time']
                        if isinstance(llm_time, list) and len(llm_time) > 0:
                            llm_times.append(llm_time[0])
                        elif isinstance(llm_time, (int, float)):
                            llm_times.append(llm_time)
                    
                    # 提取 prm_score_time (新格式直接是数值)
                    if 'prm_score_time' in data:
                        prm_time = data['prm_score_time']
                        if isinstance(prm_time, list) and len(prm_time) > 0:
                            prm_times.append(prm_time[0])
                        elif isinstance(prm_time, (int, float)):
                            prm_times.append(prm_time)
                    
                    # 提取 total_time_beam_search (新格式字段)
                    if 'total_time_beam_search' in data:
                        total_time = data['total_time_beam_search']
                        if isinstance(total_time, (int, float)):
                            total_times.append(total_time)
                    
                    # 统计token信息
                    if 'completion_tokens' in data:
                        tokens = data['completion_tokens']
                        if isinstance(tokens, list):
                            # 如果tokens都是0，使用字符长度估算 (约4字符=1token)
                            if all(t == 0 for t in tokens):
                                if 'completions' in data:
                                    completions = data['completions']
                                    total_completions += len(completions)
                                    # 估算tokens: 平均4个字符约等于1个token
                                    estimated = sum(len(c) // 4 for c in completions)
                                    estimated_tokens += estimated
                                    completion_tokens_list.append(estimated)
                            else:
                                total_completions += len(tokens)
                                token_sum = sum(tokens)
                                estimated_tokens += token_sum
                                completion_tokens_list.append(token_sum)
                        elif isinstance(tokens, (int, float)):
                            completion_tokens_list.append(tokens)
                            estimated_tokens += tokens
                            total_completions += 1
                    elif 'completions' in data:
                        # 如果没有completion_tokens字段，直接从completions估算
                        completions = data['completions']
                        total_completions += len(completions)
                        estimated = sum(len(c) // 4 for c in completions)
                        estimated_tokens += estimated
                        completion_tokens_list.append(estimated)
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"   ⚠️ 无法读取时间信息: {e}")
    
    result = {}
    if llm_times:
        result['avg_llm_time'] = sum(llm_times) / len(llm_times)
        result['total_llm_time'] = sum(llm_times)
        result['num_samples'] = len(llm_times)
    
    if prm_times:
        result['avg_prm_time'] = sum(prm_times) / len(prm_times)
        result['total_prm_time'] = sum(prm_times)
    
    # 优先使用 total_time_beam_search，如果没有则计算
    if total_times:
        result['avg_total_time'] = sum(total_times) / len(total_times)
        result['total_time'] = sum(total_times)
    elif llm_times and prm_times:
        result['avg_total_time'] = result['avg_llm_time'] + result['avg_prm_time']
        result['total_time'] = result['total_llm_time'] + result['total_prm_time']
    
    # 添加token统计
    if completion_tokens_list:
        result['total_completions'] = total_completions
        result['total_tokens_estimated'] = estimated_tokens
        result['avg_tokens_per_sample'] = estimated_tokens / len(completion_tokens_list) if completion_tokens_list else 0
        result['avg_tokens_per_completion'] = estimated_tokens / total_completions if total_completions > 0 else 0
    
    return result if result else None

def extract_params_from_filename(filename):
    """从文件名提取参数"""
    params = {}
    
    # 提取 n 值: beam_search_dynamic_n16_...
    n_match = re.search(r'_n(\d+)_', filename)
    if n_match:
        params['n'] = int(n_match.group(1))
    
    # 提取 temperature: temp0.5 或 temp0_5
    temp_match = re.search(r'_temp([0-9.]+)', filename)
    if temp_match:
        params['temperature'] = float(temp_match.group(1))
    
    # 提取 strategy: cosine, exp 等
    if '_cosine_' in filename:
        params['strategy'] = 'cosine'
    elif '_exp_' in filename:
        params['strategy'] = 'exp'
    elif '_linear_' in filename:
        params['strategy'] = 'linear'
    
    # 提取时间戳: ..._20251206_030435_...
    timestamp_match = re.search(r'_(\d{8}_\d{6})_', filename)
    if timestamp_match:
        params['timestamp'] = timestamp_match.group(1)
    
    # 提取 approach
    if 'beam_search_dynamic' in filename:
        params['approach'] = 'beam_search_dynamic'
    elif 'best_of_n' in filename:
        params['approach'] = 'best_of_n'
    elif 'beam_search' in filename:
        params['approach'] = 'beam_search'
    
    return params

def infer_params_from_sweep_order(n, timestamp, all_files):
    """根据扫参顺序推断超参数
    
    扫参顺序: product(N_VALUES, TEMP_VALUES, STRATEGY_VALUES)
    N_VALUES = [4, 16]
    TEMP_VALUES = [0.5, 0.8, 1.0, 2.0]
    STRATEGY_VALUES = ["exp", "cosine"]
    """
    TEMP_VALUES = [0.5, 0.8, 1.0, 2.0]
    STRATEGY_VALUES = ["exp", "cosine"]
    
    # 获取相同 n 值的所有文件，按时间戳排序
    same_n_files = [(f, extract_params_from_filename(os.path.basename(f))) 
                    for f in all_files 
                    if extract_params_from_filename(os.path.basename(f)).get('n') == n]
    same_n_files.sort(key=lambda x: x[1].get('timestamp', ''))
    
    # 找到当前文件在列表中的索引
    current_idx = None
    for idx, (f, params) in enumerate(same_n_files):
        if params.get('timestamp') == timestamp:
            current_idx = idx
            break
    
    if current_idx is None:
        return None, None
    
    # 根据索引计算超参数
    # 顺序: (n, temp, strategy) 其中 temp 和 strategy 循环
    num_combinations = len(TEMP_VALUES) * len(STRATEGY_VALUES)
    
    if current_idx < num_combinations:
        temp_idx = current_idx // len(STRATEGY_VALUES)
        strategy_idx = current_idx % len(STRATEGY_VALUES)
        
        return TEMP_VALUES[temp_idx], STRATEGY_VALUES[strategy_idx]
    
    return None, None

def format_config_info(config, filename_params, all_files=None):
    """格式化配置信息用于显示"""
    info_lines = []
    
    # 优先从文件名中提取参数，因为新格式没有CONFIG注释
    n = filename_params.get('n') or config.get('n', 'N/A')
    temp = filename_params.get('temperature') or config.get('beam_decay_temperature') or config.get('temperature')
    strategy = filename_params.get('strategy') or config.get('beam_decay_strategy')
    approach = filename_params.get('approach') or config.get('approach', 'N/A')
    timestamp = filename_params.get('timestamp') or config.get('timestamp', 'N/A')
    
    # 如果配置中没有 temp 和 strategy，尝试从扫参顺序推断
    if (temp is None or strategy is None) and all_files and n != 'N/A' and timestamp != 'N/A':
        inferred_temp, inferred_strategy = infer_params_from_sweep_order(n, timestamp, all_files)
        if temp is None and inferred_temp is not None:
            temp = inferred_temp
        if strategy is None and inferred_strategy is not None:
            strategy = inferred_strategy
    
    if temp is None:
        temp = 'N/A'
    if strategy is None:
        strategy = 'N/A'
    
    info_lines.append(f"   Approach: {approach}")
    info_lines.append(f"   N: {n}")
    info_lines.append(f"   Temperature: {temp}")
    info_lines.append(f"   Strategy: {strategy}")
    info_lines.append(f"   Timestamp: {timestamp}")
    
    return '\n'.join(info_lines), {
        'approach': approach,
        'n': n,
        'temperature': temp,
        'strategy': strategy,
        'timestamp': timestamp
    }

def run_evaluation(file_path, output_file, all_files=None):
    """运行单个文件的评估"""
    filename = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f"📊 正在评估: {filename}")
    print(f"{'='*80}")
    
    # 提取配置信息
    config = extract_config_from_file(file_path)
    filename_params = extract_params_from_filename(filename)
    config_info, params_dict = format_config_info(config, filename_params, all_files)
    
    # 提取生成时间和token统计
    time_info = extract_generation_times_and_tokens(file_path)
    print(config_info)
    if time_info:
        if 'avg_llm_time' in time_info:
            print(f"   Avg LLM Time: {time_info['avg_llm_time']:.2f}s")
            params_dict['avg_llm_time'] = round(time_info['avg_llm_time'], 2)
            params_dict['total_llm_time'] = round(time_info['total_llm_time'], 2)
        
        if 'avg_prm_time' in time_info:
            print(f"   Avg PRM Time: {time_info['avg_prm_time']:.2f}s")
            params_dict['avg_prm_time'] = round(time_info['avg_prm_time'], 2)
            params_dict['total_prm_time'] = round(time_info['total_prm_time'], 2)
        
        if 'avg_total_time' in time_info:
            print(f"   Avg Total Time: {time_info['avg_total_time']:.2f}s")
            params_dict['avg_total_time'] = round(time_info['avg_total_time'], 2)
            params_dict['total_time'] = round(time_info['total_time'], 2)
        
        if 'num_samples' in time_info:
            print(f"   Num Samples: {time_info['num_samples']}")
            params_dict['num_samples'] = time_info['num_samples']
        
        # 添加token统计信息
        if 'total_tokens_estimated' in time_info:
            print(f"   Total Tokens (est): {time_info['total_tokens_estimated']:,}")
            print(f"   Avg Tokens/Sample: {time_info['avg_tokens_per_sample']:.1f}")
            print(f"   Avg Tokens/Completion: {time_info['avg_tokens_per_completion']:.1f}")
            print(f"   Total Completions: {time_info['total_completions']}")
            params_dict['total_tokens_estimated'] = time_info['total_tokens_estimated']
            params_dict['avg_tokens_per_sample'] = round(time_info['avg_tokens_per_sample'], 1)
            params_dict['avg_tokens_per_completion'] = round(time_info['avg_tokens_per_completion'], 1)
            params_dict['total_completions'] = time_info['total_completions']
    
    # 直接使用 conda 环境中的 python
    python_path = os.path.join(CONDA_ENV_PATH, "bin", "python")
    
    cmd = [
        python_path,
        EVAL_SCRIPT,
        "--file_path",
        file_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10分钟超时
        )
        
        # 提取准确率
        accuracy = None
        output = result.stdout
        
        # 尝试多种格式提取准确率
        # 格式1: 'acc': 56.0
        match = re.search(r"'acc'\s*:\s*(\d+\.?\d*)", output)
        if match:
            accuracy = float(match.group(1))
        else:
            # 格式2: accuracy: 56.0% 或 accuracy: 56.0
            for line in output.split('\n'):
                if 'accuracy' in line.lower() or 'acc' in line.lower():
                    # 尝试提取百分比
                    match = re.search(r'(\d+\.?\d*)\s*%', line)
                    if match:
                        accuracy = float(match.group(1))
                        break
                    # 尝试提取数字
                    match = re.search(r':\s*(\d+\.?\d*)', line)
                    if match:
                        accuracy = float(match.group(1))
                        break
        
        # 写入日志
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"文件: {filename}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(config_info + "\n")
            f.write(f"{'='*80}\n")
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n--- STDERR ---\n{result.stderr}\n")
            f.write("\n")
        
        print(f"✅ 完成: {filename}")
        
        # 打印准确率
        if accuracy is not None:
            print(f"   📈 Accuracy: {accuracy}%")
            params_dict['accuracy'] = accuracy
        else:
            print(f"   ⚠️ 未能提取准确率")
            for line in result.stdout.split('\n'):
                if 'accuracy' in line.lower() or 'correct' in line.lower():
                    print(f"   {line.strip()}")
        
        return True, params_dict
        
    except subprocess.TimeoutExpired:
        error_msg = f"❌ 超时: {filename}"
        print(error_msg)
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"文件: {filename}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{error_msg}\n\n")
        return False, {}
        
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ 失败: {filename}\n错误: {e}"
        print(error_msg)
        if e.stderr:
            print(f"   {e.stderr[:200]}")
        
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"文件: {filename}\n")
            f.write(f"{'='*80}\n")
            f.write(f"❌ 评估失败\n{error_msg}\n")
            if e.stderr:
                f.write(f"STDERR:\n{e.stderr}\n")
            f.write("\n")
        return False, {}

def main():
    # 清空或创建输出日志文件
    with open(OUTPUT_LOG, 'w') as f:
        f.write(f"批量评估结果\n")
        f.write(f"结果目录: {RESULTS_DIR}\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    # 获取所有文件
    jsonl_files = get_jsonl_files(RESULTS_DIR)
    
    if not jsonl_files:
        print(f"❌ 在 {RESULTS_DIR} 中未找到任何 completions.jsonl 文件")
        return
    
    print(f"🚀 找到 {len(jsonl_files)} 个文件待评估")
    print(f"📂 结果目录: {RESULTS_DIR}")
    print(f"📝 日志输出: {OUTPUT_LOG}")
    print(f"📊 TSV汇总: {OUTPUT_TSV}")
    print(f"🐍 Python: {CONDA_ENV_PATH}/bin/python\n")
    
    # 统计
    success_count = 0
    failed_count = 0
    results = []
    
    # 逐个评估
    for idx, file_path in enumerate(jsonl_files, 1):
        print(f"\n进度: [{idx}/{len(jsonl_files)}]")
        
        success, params = run_evaluation(file_path, OUTPUT_LOG, jsonl_files)
        if success:
            success_count += 1
            if params:
                params['filename'] = os.path.basename(file_path)
                results.append(params)
        else:
            failed_count += 1
    
    # 生成 TSV 汇总表
    if results:
        import csv
        with open(OUTPUT_TSV, 'w', newline='') as f:
            fieldnames = ['filename', 'approach', 'n', 'temperature', 'strategy', 'timestamp', 'accuracy', 
                         'avg_llm_time', 'avg_prm_time', 'avg_total_time', 
                         'total_llm_time', 'total_prm_time', 'total_time', 'num_samples']
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n📊 TSV汇总表已生成: {OUTPUT_TSV}")
        
        # 按超参数分组显示结果
        print(f"\n{'='*80}")
        print(f"📈 结果汇总（按超参数分组）")
        print(f"{'='*80}")
        
        # 按 n, temperature, strategy 分组
        grouped = {}
        for r in results:
            key = (r.get('n'), r.get('temperature'), r.get('strategy'))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        
        for key, items in sorted(grouped.items()):
            n, temp, strategy = key
            print(f"\nN={n}, Temperature={temp}, Strategy={strategy}")
            for item in items:
                acc = item.get('accuracy', 'N/A')
                ts = item.get('timestamp', 'N/A')
                avg_total = item.get('avg_total_time', item.get('avg_llm_time', 'N/A'))
                avg_llm = item.get('avg_llm_time', 'N/A')
                avg_prm = item.get('avg_prm_time', 'N/A')
                print(f"  - {ts}: Accuracy={acc}%, LLM={avg_llm}s, PRM={avg_prm}s, Total={avg_total}s")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"🎉 评估完成！")
    print(f"{'='*80}")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"📊 总计: {len(jsonl_files)}")
    print(f"📝 详细结果已保存到: {OUTPUT_LOG}")
    print(f"📊 TSV汇总已保存到: {OUTPUT_TSV}")
    
    # 写入总结
    with open(OUTPUT_LOG, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"评估总结\n")
        f.write(f"{'='*80}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"总计: {len(jsonl_files)}\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
