import random

# FakePRM: 用于 debug，不加载真实模型，score 返回 0~1 随机分数
class FakePRM:
    def score(self, prompts, completions):
        # 返回和 completions 结构一致的分数（每个 completion 一个分数）
        return [[[random.random()]] for _ in completions]