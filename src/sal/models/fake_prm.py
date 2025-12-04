import random

# FakePRM: 用于 debug，不加载真实模型，score 返回 0~1 随机分数
class FakePRM:
    def score(self, prompts, completions):
        """
        completions 结构: [Batch_Size, N] (例如 [[c1, c2, c3, c4]])
        我们需要返回: [Batch_Size, N, 1] 
        """
        return [
            # 外层循环遍历 Batch (c_list 是这一批的 N 个候选项)
            # 内层循环遍历 N (_ 是单个 completion 字符串)
            [ [random.random()] for _ in c_list ] 
            for c_list in completions
        ]