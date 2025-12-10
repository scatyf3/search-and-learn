#!/usr/bin/env python
"""
批量评估所有生成的 completions.jsonl 文件
"""

import os
import subprocess
import glob
from pathlib import Path

# ================= 配置区域 =================
CONDA_ENV = "/scratch/yf3005/qwen-math"
EVAL_SCRIPT = "/home/yf3005/Qwen2.5-Math/evaluation/evaluate.py"
RESULTS_DIR = "/home/yf3005/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_LOG = "/home/yf3005/search-and-learn/evaluation_results.txt"
# ===========================================

def get_jsonl_files(directory):
    """获取目录下所有的 completions.jsonl 文件"""
    pattern = os.path.join(directory, "*_completions.jsonl")
    files = glob.glob(pattern)
    files.sort()  # 按文件名排序
    return files

def run_evaluation(file_path, output_file):
    """运行单个文件的评估"""
    filename = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f"📊 正在评估: {filename}")
    print(f"{'='*80}")
    
    # 构建命令
    cmd = [
        "conda", "run", "-n", CONDA_ENV.split('/')[-1], 
        "--no-capture-output",
        "python", EVAL_SCRIPT,
        "--file_path", file_path
    ]
    
    try:
        # 运行评估
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 写入日志
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"文件: {filename}\n")
            f.write(f"{'='*80}\n")
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n--- STDERR ---\n{result.stderr}\n")
            f.write("\n")
        
        print(f"✅ 完成: {filename}")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ 失败: {filename}\n错误: {e}\n{e.stderr}"
        print(error_msg)
        
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"文件: {filename}\n")
            f.write(f"{'='*80}\n")
            f.write(f"❌ 评估失败\n{error_msg}\n\n")
        
        return False

def main():
    # 清空或创建输出日志文件
    with open(OUTPUT_LOG, 'w') as f:
        f.write(f"批量评估结果\n")
        f.write(f"结果目录: {RESULTS_DIR}\n")
        f.write(f"评估时间: {subprocess.check_output(['date']).decode().strip()}\n")
        f.write(f"{'='*80}\n\n")
    
    # 获取所有文件
    jsonl_files = get_jsonl_files(RESULTS_DIR)
    
    if not jsonl_files:
        print(f"❌ 在 {RESULTS_DIR} 中未找到任何 completions.jsonl 文件")
        return
    
    print(f"🚀 找到 {len(jsonl_files)} 个文件待评估")
    print(f"📂 结果目录: {RESULTS_DIR}")
    print(f"📝 日志输出: {OUTPUT_LOG}\n")
    
    # 统计
    success_count = 0
    failed_count = 0
    
    # 逐个评估
    for idx, file_path in enumerate(jsonl_files, 1):
        print(f"\n进度: [{idx}/{len(jsonl_files)}]")
        
        if run_evaluation(file_path, OUTPUT_LOG):
            success_count += 1
        else:
            failed_count += 1
    
    # 总结
    print(f"\n{'='*80}")
    print(f"🎉 评估完成！")
    print(f"{'='*80}")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"📊 总计: {len(jsonl_files)}")
    print(f"📝 详细结果已保存到: {OUTPUT_LOG}")
    
    # 写入总结
    with open(OUTPUT_LOG, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"评估总结\n")
        f.write(f"{'='*80}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"总计: {len(jsonl_files)}\n")

if __name__ == "__main__":
    main()
