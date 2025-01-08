from abc import ABCMeta, abstractmethod


class Attack:
    __meta_class__ = ABCMeta

    def __init__(self):
        self.context = {}

    # 用于读取配置文件中的内容或者其它信息，算法如有需要额外知识可以调用，比如假设客户端可以串通
    def assign(self, context: dict):
        self.context.update(context)

    # 生成有毒梯度，返回的值是字典，类似于 model.state_dict() 的结构
    @abstractmethod
    def gen_gradient(self, context: dict):
        pass

class Strategic:
    __meta_class__ = ABCMeta

    def __init__(self):
        self.context = {}

    # 用途同上
    def assign(self, context: dict):
        self.context.update(context)

    # 检测有毒梯度，进行报告，返回数组，对应所有客户都梯度，有毒梯度置为 1，其它置为 0
    # gradients = context['gradients']
    @abstractmethod
    def examine(self, context: dict) -> list:
        pass

    # 设计聚合算法
    # updates = context['updates']
    @abstractmethod
    def aggregate(self, context: dict) -> list:
        pass
