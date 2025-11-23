# 基于 recipes/Llama-3.2-3B-Instruct/best_of_n.yaml
# 修改 n = 4, 16, 64, 256，分别运行，保存结果

import yaml
import subprocess
import shutil
import os

# 原始yaml路径和备份
yaml_path = 'recipes/Llama-3.2-3B-Instruct/best_of_n.yaml'
backup_path = yaml_path + '.bak'
shutil.copyfile(yaml_path, backup_path)

n_list = [4, 16, 64, 256]

for n in n_list:
    # 修改yaml中的n
    with open(backup_path, 'r') as f:
        config = yaml.safe_load(f)
    config['n'] = n
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Running n={n}...')
    # 运行推理脚本
    subprocess.run(['python', 'scripts/test_time_compute.py', yaml_path])
    print(f'Finished n={n}\n')

# 恢复原始yaml
shutil.move(backup_path, yaml_path)
print('All runs finished, yaml restored.')


