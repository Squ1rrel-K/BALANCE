from impl import train_client_DP_reg, get_test_acc_reg, train_client_DP, get_test_acc, train_client, train_client_reg
from meta_define.fl import TaskType


class RegressionDP(TaskType):

    def __init__(self):
        self.noise_multiplier = 1.1
        self.max_grad_norm = 1.0

    def train(self, context: dict):
        return train_client_DP_reg(context["c_d"], context["model"], context["optim"], context["lr"],
                                   context["loss_func"],
                                   context["model_state"], context["device"], self.noise_multiplier,
                                   self.max_grad_norm)

    def eval(self, context: dict):
        return get_test_acc_reg(context["model"], context["data"], context["target"], context["loss_func"],
                                context["device"])


class ClassificationDP(TaskType):

    def __init__(self):
        self.noise_multiplier = 1.1
        self.max_grad_norm = 1.0

    def train(self, context: dict):
        return train_client_DP(context["c_d"], context["model"], context["optim"], context["lr"],
                               context["loss_func"],
                               context["model_state"], context["device"], self.noise_multiplier,
                               self.max_grad_norm)

    def eval(self, context: dict):
        return get_test_acc(context["model"], context["data"], context["target"], context["loss_func"],
                                context["device"])


class Classification(TaskType):

    def eval(self, context: dict):
        return get_test_acc(context["model"], context["data"], context["target"], context["loss_func"],
                                context["device"])

    def train(self, context: dict):
        return train_client(context["c_d"], context["model"], context["optim"], context["lr"], context["loss_func"],
                            context["model_state"], context["device"])


class Regression(TaskType):

    def eval(self, context: dict):
        return get_test_acc_reg(context["model"], context["data"], context["target"], context["loss_func"],
                                context["device"])

    def train(self, context: dict):
        return train_client_reg(context["c_d"], context["model"], context["optim"], context["lr"], context["loss_func"],
                                context["model_state"], context["device"])
