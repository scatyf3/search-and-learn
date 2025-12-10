#!/bin/bash
# 为了测试test time compute的计算开销，从math500 取第一个数据集，用torch profiler计算结果

# --- Slurm 资源配置 ---
#SBATCH --job-name=test_compute       # 任务名称
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --ntasks-per-node=1           # 每个节点运行1个任务
#SBATCH --cpus-per-task=4             # 为该任务申请4个CPU核心
#SBATCH --mem=128G                     # 申请32GB内存
#SBATCH --time=1:00:00               # 任务最长运行时间 (1小时)
#SBATCH --account=pr_289_general      # (!!!) 你的账户 (来自 sshare)
#SBATCH --partition=rtx8000           # rtx8000分区有加速效果
#SBATCH --gres=gpu:1           
# --- 日志路径 (!!!) ---
# (确保日志也保存在 /scratch 目录下)
# %x 是任务名, %j 是任务ID
# (请确保 /scratch/yf3005/slurm_logs/ 目录存在)
#SBATCH --output=/scratch/yf3005/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/yf3005/slurm_logs/%x-%j.err


# 对比 vLLM 和 SGLang 推理性能的脚本

set -e

echo "========================================="
echo "Benchmarking vLLM vs SGLang"
echo "========================================="

module load anaconda3/2020.07          # 加载集群的 Anaconda
eval "$(conda shell.bash hook)"        # 启用 'conda activate'
export PYTHONNOUSERSITE=True           # 隔离环境

echo ""

# ============ 1. 测试 vLLM ============
echo "Step 1: Testing vLLM..."
echo "Activating vLLM environment..."

# 激活 vLLM 环境
conda activate /scratch/yf3005/sal_project/sal

echo "Running vLLM benchmark..."
python benchmark_vllm.py

echo "vLLM benchmark completed."
echo ""

# ============ 2. 测试 SGLang ============
echo "Step 2: Testing SGLang..."
echo "Switching to SGLang environment..."

# 切换到 SGLang 环境
conda deactivate
deactivate || true
conda activate /scratch/yf3005/sal_project/sglang_env

echo "Running SGLang benchmark..."
python benchmark_sglang.py

echo "SGLang benchmark completed."
echo ""

# ============ 3. 对比结果 ============
echo "========================================="
echo "Comparison Results"
echo "========================================="

echo ""
echo "--- vLLM Results ---"
cat vllm_benchmark_result.txt

echo ""
echo "--- SGLang Results ---"
cat sglang_benchmark_result.txt

echo ""
echo "========================================="
echo "Benchmark completed!"
echo "Results saved to vllm_benchmark_result.txt and sglang_benchmark_result.txt"
echo "========================================="
