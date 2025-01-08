import torch
from torch.utils.data import DataLoader

from impl import train_client
from meta_define.attack import Strategic
import util.easy_fed as ef


class DefaultStrategic(Strategic):

    def examine(self, context: dict) -> list:
        gradients = context['gradients']
        n = len(gradients)
        return [0 for _ in range(n)]

    def aggregate(self, context: dict):
        return context['updates']


class Krum(Strategic):

    def __init__(self):
        self.scores = None
        self.f = 3
        self.n = 10
        self.k = self.n - self.f - 1

    def get_score(self, g, gradients):
        s = []
        for grad in gradients:
            w = ef.l2(g, grad)
            s.append(w)
        list.sort(s)
        return sum(s[0:self.n - self.f])

    def update_grad_score(self, gradients):
        scores = []
        for i in range(len(gradients)):
            score = self.get_score(gradients[i], gradients)
            scores.append([score, i])
        scores.sort(key=lambda x: x[0])
        self.scores = scores

    def examine(self, context: dict) -> list:
        gradients = context['gradients']
        result = [1 for _ in range(len(gradients))]
        for i in range(self.k):
            result[self.scores[i][1]] = 0
        return result

    def aggregate(self, context: dict):
        updates = context['updates']
        n = len(updates)
        self.update_grad_score(updates)
        result = []
        for i in range(self.k):
            result.append(ef.mul_constant(updates[self.scores[i][1]], n / self.k))
        return result


class FLtrust(Strategic):
    def __init__(self):
        super().__init__()
        self.scale = 1
        self.update = None
        self.root_data, self.model_state, self.optim, self.lr, self.loss_fun, self.model_type, self.device \
            = None, None, None, None, None, None, None
        self.epoch = 1

    def assign(self, context: dict):
        train_dataset, _ = context["dataset"]
        data_size = 300
        self.root_data = DataLoader(dataset=torch.utils.data.Subset(train_dataset, range(0, data_size)),
                                    batch_size=min(32, data_size),
                                    shuffle=True)
        self.model_type = context["model"]
        self.optim, self.lr, self.loss_fun, self.device = \
            (context["optim"], context["lr"], context["loss_fun"], context["device"])

    def train(self):
        if (self.root_data and self.model_state and self.optim and self.lr and self.loss_fun and self.model_type and
                self.device is not None):
            self.update = ef.copy(self.model_state)
            for i in range(self.epoch):
                update, sum_loss, true_predict, size = train_client(self.root_data, self.model_type, self.optim,
                                                                    self.lr,
                                                                    self.loss_fun, self.model_state, self.device)
                self.model_state = ef.sub(self.model_state, update)
            self.update = ef.mul_constant(ef.sub(self.update, self.model_state), self.scale)
            self.model_state = None

    def examine(self, context: dict) -> list:
        self.train()
        gradients = context['gradients']
        result = []
        for g in gradients:
            w, g_l2, go_l2 = ef.cos(g, self.update)
            if w <= 0:
                result.append(1)
            else:
                result.append(0)
        return result

    def aggregate(self, context: dict):
        model_state = context['global_model'].state_dict()
        self.model_state = ef.copy(model_state)
        self.train()
        updates = context['updates']
        result = []
        n = len(updates)
        sum_weight = 0
        for up in updates:
            w, g_l2, go_l2 = ef.cos(up, self.update)
            w = max(w, 0)
            sum_weight += w
            result.append(ef.mul_constant(up, w * go_l2 / g_l2))
        for i in range(len(result)):
            result[i] = ef.mul_constant(result[i], n)
        return result
