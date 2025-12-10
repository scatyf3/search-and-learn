#!/bin/bash

# --- Slurm 资源配置 ---
#SBATCH --job-name=test_dynamic_beam       # 任务名称
#SBATCH --nodes=1                     # 申请1个节点
#SBATCH --ntasks-per-node=1           # 每个节点运行1个任务
#SBATCH --cpus-per-task=4             # 为该任务申请4个CPU核心
#SBATCH --mem=32G                     # 申请32GB内存
#SBATCH --time=24:00:00               # 任务最长运行时间 (1小时)
#SBATCH --account=pr_289_general      # (!!!) 你的账户 (来自 sshare)
#SBATCH --partition=a100_1            # (!!!) 申请A100分区 (来自 sinfo)
#SBATCH --gres=gpu:a100:1             # (!!!) 明确申请1块A100 GPU

# --- 日志路径 (!!!) ---
# (确保日志也保存在 /scratch 目录下)
# %x 是任务名, %j 是任务ID
# (请确保 /scratch/yf3005/slurm_logs/ 目录存在)
#SBATCH --output=/scratch/yf3005/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/yf3005/slurm_logs/%x-%j.err

# --- 运行你的程序 ---

# 1. 创建日志目录 (如果不存在)
mkdir -p /scratch/yf3005/slurm_logs

# 2. 打印任务信息
echo "Job starting on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# 3. (!!!) 进入你的项目根目录
#    (我假设你的'search-and-learn'目录在 /scratch/yf3005/ 下)
cd /home/yf3005/search-and-learn

# 4. (!!!) 激活 Conda 环境 (HPC 的正确方式)
module purge
module load anaconda3/2020.07          # 1. 加载集群的 Anaconda
eval "$(conda shell.bash hook)"        # 2. 启用 'conda activate' 功能
conda activate /scratch/yf3005/sal_project/sal # 3. 激活你 /scratch 上的环境
export PYTHONNOUSERSITE=True           # 4. 隔离环境

# 5. 打印 Python 路径 (用于调试)
echo "Using Python at: $(which python)"

# 6. 运行你的 Python 脚本

# 排查有无问题
# python scripts/test_time_compute_fake_prm.py recipes/Llama-3.2-1B-Instruct/beam_search_n4_debug.yaml

python scripts/test_time_compute_fake_prm.py recipes/Llama-3.2-1B-Instruct/beam_search_n4.yaml
python scripts/test_time_compute_fake_prm.py recipes/Llama-3.2-1B-Instruct/beam_search_n16.yaml
# 7. 任务结束
echo "Job finished."