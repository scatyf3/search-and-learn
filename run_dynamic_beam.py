import os
import yaml
import subprocess
import time
from itertools import product

# ================= ⚙️ 配置区域 =================

# 1. 原始模板文件路径 (你的基础配置)
TEMPLATE_CONFIG_PATH = "recipes/Llama-3.2-1B-Instruct/beam_search_dynamic.yaml"

# 2. 你的运行脚本路径
SCRIPT_PATH = "scripts/test_time_compute_fake_prm.py"

# 3. 临时配置文件存放目录 (会自动创建)
SWEEP_CONFIG_DIR = "recipes/sweeps"

# 4. 要枚举的参数网格
STRATEGY_VALUES = ["exp", "cosine"]
N_VALUES = [4, 16, 64]
TEMP_VALUES = [0.5, 0.8, 1.0, 2.0]

# 5. Debug模式 (如果为True，只采样1个问题)
DEBUG_MODE = False  # 设为 False 关闭调试

# ==============================================

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def run_sweep():
    # 确保存放生成配置的目录存在
    os.makedirs(SWEEP_CONFIG_DIR, exist_ok=True)

    # 读取原始配置作为模板
    if not os.path.exists(TEMPLATE_CONFIG_PATH):
        print(f"❌ 错误: 找不到模板文件 {TEMPLATE_CONFIG_PATH}")
        return

    base_config = load_yaml(TEMPLATE_CONFIG_PATH)
    
    # 生成参数组合
    combinations = list(product(N_VALUES, TEMP_VALUES, STRATEGY_VALUES))
    total_jobs = len(combinations)
    
    print(f"🚀 开始扫参任务 | 共 {total_jobs} 组实验")
    print(f"📂 模板配置: {TEMPLATE_CONFIG_PATH}")
    print(f"📝 生成配置目录: {SWEEP_CONFIG_DIR}\n")

    for idx, (n, temp, strategy) in enumerate(combinations, 1):
        print(f"--------------------------------------------------")
        print(f"▶️  [任务 {idx}/{total_jobs}] 正在配置: n={n}, temp={temp}, strategy={strategy}")

        # 1. 修改配置
        current_config = base_config.copy()
        current_config['n'] = n
        current_config['beam_decay_temperature'] = temp
        current_config['beam_decay_strategy'] = strategy
        
        # Debug模式: 只采样1个问题
        if DEBUG_MODE:
            current_config['num_samples'] = 1 

        # 2. 生成新的文件名
        # 例如: beam_dynamic_n16_t0.8.yaml
        config_filename = f"beam_dynamic_n{n}_t{temp}_{strategy}.yaml"
        new_config_path = os.path.join(SWEEP_CONFIG_DIR, config_filename)

        # 3. 保存新配置到磁盘
        save_yaml(current_config, new_config_path)

        # 4. 拼接运行命令
        # 你的命令格式: python script.py config_path
        cmd = ["python", SCRIPT_PATH, new_config_path]

        try:
            start_time = time.time()
            
            # 5. 执行命令
            # check=True 会在脚本报错(exit code != 0)时抛出异常
            subprocess.run(cmd, check=True)
            
            duration = time.time() - start_time
            print(f"✅ [任务 {idx}] 完成! 耗时: {duration:.2f}s")
            
        except subprocess.CalledProcessError:
            print(f"❌ [任务 {idx}] 失败! (n={n}, temp={temp}, strategy={strategy})")
            # 可以选择 continue 继续，或者 break 退出
            continue
        except KeyboardInterrupt:
            print("\n🛑 用户手动停止扫参。")
            break

    print("\n🎉 所有扫参任务结束。")

if __name__ == "__main__":
    run_sweep()
