from transformers import AutoTokenizer
import json

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# JSONL 文件路径
jsonl_file_path = "data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_temp1.0_exp_20251210_144841_completions.jsonl"

# 检查每个 completion 的 token 数量
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        data = json.loads(line)
        completions = data.get("completions", [])
        
        print(f"Line {line_number}:")
        for idx, completion in enumerate(completions):
            token_count = len(tokenizer.encode(completion, add_special_tokens=False))
            print(f"  Completion {idx + 1}: {token_count} tokens")