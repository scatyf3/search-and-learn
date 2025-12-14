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

import logging
import time
import json
from pathlib import Path
from datetime import datetime
import glob
import os

from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config) -> Dataset:
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))
    
    return dataset


def get_processed_indices(config: Config) -> set:
    """检查输出目录中已存在的文件，返回已处理的样本索引集合"""
    if config.output_dir is None:
        config.output_dir = f"data/{config.model_path}"
    
    if not os.path.exists(config.output_dir):
        return set()
    
    # 构建文件名模式，与 save_dataset 保持一致
    n_str = f"_n{config.n}" if hasattr(config, "n") and config.n is not None else ""
    
    # 添加 temperature 和 strategy 参数（与 save_dataset 逻辑一致）
    temp_str = ""
    strategy_str = ""
    if hasattr(config, "beam_decay_temperature") and config.beam_decay_temperature is not None:
        temp_str = f"_temp{config.beam_decay_temperature}"
    if hasattr(config, "beam_decay_strategy") and config.beam_decay_strategy is not None:
        strategy_str = f"_{config.beam_decay_strategy}"
    
    pattern = f"{config.output_dir}/{config.approach}{n_str}{temp_str}{strategy_str}_*_completions.jsonl"
    
    existing_files = glob.glob(pattern)
    if not existing_files:
        return set()
    
    # 使用最新的文件
    latest_file = max(existing_files, key=os.path.getmtime)
    logger.info(f"🔍 发现已存在文件: {latest_file}")
    
    processed_indices = set()
    try:
        with open(latest_file, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    data = json.loads(line)
                    # 假设数据集按顺序处理，使用行号作为索引（减去配置行）
                    processed_indices.add(len(processed_indices))
                except json.JSONDecodeError:
                    continue
        
        if processed_indices:
            logger.info(f"✅ 从 {latest_file} 加载了 {len(processed_indices)} 个已处理样本")
    except Exception as e:
        logger.warning(f"⚠️  读取已存在文件失败: {e}")
        return set()
    
    return processed_indices


def save_dataset(dataset, config):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            config.output_dir = f"data/{config.model_path}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 文件名加入n参数和其他参数
        n_str = f"_n{config.n}" if hasattr(config, "n") and config.n is not None else ""
        
        # 添加 temperature 和 strategy 参数（如果存在）
        temp_str = ""
        strategy_str = ""
        if hasattr(config, "beam_decay_temperature") and config.beam_decay_temperature is not None:
            temp_str = f"_temp{config.beam_decay_temperature}"
        if hasattr(config, "beam_decay_strategy") and config.beam_decay_strategy is not None:
            strategy_str = f"_{config.beam_decay_strategy}"
        
        params_str = f"{n_str}{temp_str}{strategy_str}"
        
        # 总是生成新文件：生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{config.output_dir}/{config.approach}{params_str}_{timestamp}_completions.jsonl"
        
        # 保存配置头
        config_dict = config.__dict__.copy()
        config_dict["timestamp"] = timestamp
        
        with open(out_path, 'w') as f:
            f.write(f"# CONFIG: {json.dumps(config_dict, ensure_ascii=False)}\n")
        
        logger.info(f"✨ 创建新文件: {out_path}")

        # 保存数据集内容（追加模式）
        dataset.to_json(out_path, lines=True, mode='a')
        logger.info(f"💾 已保存 {len(dataset)} 条记录到 {out_path}")
