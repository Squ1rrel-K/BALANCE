import numpy as np
import torch
from torch.autograd import Variable

from impl import train_client
from meta_define.attack import Attack
import util.easy_fed as ef
from util import count_loss


class DefaultAttack(Attack):

    def gen_gradient(self, context: dict):
        c_d, model, optim, lr, loss_func, model_state, device = (context['c_d'], context['model'], context['optim'],
                                                                 context['lr'], context['loss_func'],
                                                                 context['model_state'],
                                                                 context['device'])
        client_update, sum_loss, true_predict, size = train_client(c_d, model, optim, lr, loss_func, model_state,
                                                                   device)
        return client_update

class ReversalGradient(Attack):

    def gen_gradient(self, context: dict):
        c_d, model, optim, lr, loss_fun, model_state, device = (context['client'], context['model'], context['optim'],
                                                                context['lr'], context['loss_fun'],
                                                                context['model_state'],
                                                                context['device'])
        client_update, sum_loss, true_predict, size = train_client(c_d, model, optim, lr, loss_fun,
                                                                       model_state, device)
        return ef.mul_constant(client_update, -1)


class LabelFlipping(Attack):

    def gen_gradient(self, context: dict):
        c_d, model, optim, lr, loss_fun, model_state, device = (context['c_d'], context['model'], context['optim'],
                                                                context['lr'], context['loss_func'],
                                                                context['model_state'],
                                                                context['device'])
        client_update, sum_loss, true_predict, size = LabelFlipping.LF_train(c_d, model, optim, lr, loss_fun,
                                                                             model_state, device)
        return client_update

    @staticmethod
    def LF_train(client, model, optim, lr, loss_fun, model_state, device):
        sum_loss = 0
        true_predict = 0
        size = 0
        local_model = model().to(device)
        local_model.load_state_dict(model_state)
        optimizer = optim(local_model.parameters(), lr=lr)
        for data in client:
            inputs, labels = data
            idx = torch.randperm(labels.nelement())
            labels = labels.view(-1)[idx].view(labels.size())
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = count_loss(loss_fun, outputs, labels)
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            true_predict += torch.sum(id == labels.data)
            size += len(labels)
        client_update = ef.sub(model_state, local_model.state_dict())
        return client_update, sum_loss, true_predict, size



class Sine(Attack):

    def __init__(self):
        super().__init__()
        self.cosine_scaling_factor = 2
        self.norm_scaling_factor = 100
        self.scale = None
        self.m = None
        self.update = None
        self.root_data, self.model_state, self.optim, self.lr, self.loss_fun, self.model_type, self.device \
            = None, None, None, None, None, None, None
        self.epoch = 1
        self.adv_num = 4
        self.imp_k_size = 10000

    def set_cl(self, root_data, optim, lr, loss_fun, model_type, device, scale=1):
        self.root_data, self.optim, self.lr, self.loss_fun, self.model_type, self.device \
            = root_data, optim, lr, loss_fun, model_type, device
        self.scale = scale

    def train(self):
        if (self.root_data and self.model_state and self.optim and self.lr and self.loss_fun and self.model_type and
                self.device is not None):
            self.update = ef.add(self.model_state, ef.sub(self.model_state, self.model_state))
            for i in range(self.epoch):
                update, sum_loss, true_predict, size = train_client(self.root_data, self.model_type, self.optim,
                                                                    self.lr,
                                                                    self.loss_fun, self.model_state, self.device)
                self.model_state = ef.sub(self.model_state, update)
            self.update = ef.sub(self.update, self.model_state)
            self.model_state = None

    def assign(self, context: dict):
        super().assign(context=context)
        client_data = context["client_data"]
        model_state = context["global_model"].state_dict()
        self.model_state = model_state
        self.train()
        avg_update = ef.zeros(model_state)
        for i in range(self.adv_num):
            client = client_data[i]
            update, sum_loss, true_predict, size = train_client(client, self.model_type, self.optim,
                                                                self.lr,
                                                                self.loss_fun, model_state, self.device)
            avg_update = ef.add(avg_update, ef.mul_constant(update, 1 / self.adv_num))
        self.m = self.gen_m(avg_update, self.update, device=self.device)

    @staticmethod
    def q_e_solve(a, b, c):
        delta = b * b - 4 * a * c
        if delta >= 0:
            sqrt_delta = torch.sqrt(delta)
            return (-b + sqrt_delta) / (2 * a), (-b - sqrt_delta) / (2 * a)
        return None

    @staticmethod
    def q_e_solve_np(a, b, c):
        delta = b * b - 4 * a * c
        print("d=", delta, ", b^2=", b * b, ", 4ac=", 4 * a * c)
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            return (-b + sqrt_delta) / (2 * a), (-b - sqrt_delta) / (2 * a)
        return None

    @staticmethod
    def cosine_sim(a, b):
        return np.sum(a * b) / (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))

    @staticmethod
    def similar_cosine_vector(m, cl, cosine_scaling_factor, k):
        m = torch.clone(m).cpu().numpy()
        cl = torch.clone(cl).cpu().numpy()
        alpha = np.sum(m * cl)
        beta = np.sum(m * m)
        a = alpha - m[k] * cl[k]
        b = beta - m[k] * m[k]
        alpha = 1 / (cosine_scaling_factor * cosine_scaling_factor) * alpha * alpha
        x = Sine.q_e_solve_np(beta * cl[k] * cl[k] - alpha, 2 * a * cl[k] * beta, beta * a * a - alpha * b)
        if x is not None:
            result1, result2 = np.copy(m), np.copy(m)
            result1[k] = x[0]
            result2[k] = x[1]
            return torch.from_numpy(result1), torch.from_numpy(result2)
        return None

    @staticmethod
    def update_param(alpha_pre, beta, m, cl, k, x):
        alpha_pre = alpha_pre - m[k] * cl[k] + x * cl[k]
        beta = beta - m[k] * m[k] + x * x
        return alpha_pre, beta

    @staticmethod
    def similar_cosine_vector_x(alpha_pre, beta, m, cl, cosine_scaling_factor, k):
        a = alpha_pre - m[k] * cl[k]
        b = beta - m[k] * m[k]
        alpha = 1 / (cosine_scaling_factor * cosine_scaling_factor) * alpha_pre * alpha_pre
        return Sine.q_e_solve(beta * cl[k] * cl[k] - alpha, 2 * a * cl[k] * beta, beta * a * a - alpha * b)

    def gen_m(self, b, cl, device):
        b, cl = ef.flatten_all_gradient(b), ef.flatten_all_gradient(cl)
        cl_dist = torch.sqrt(torch.sum(cl * cl))
        cs_mb = 1
        m = torch.clone(b)
        imp_k = torch.sort(torch.abs(b - cl), descending=True).indices
        imp_k = imp_k[:self.imp_k_size]
        alpha_pre = torch.sum(m * cl)
        beta = torch.sum(m * m)
        for k in imp_k:
            # m_k_t = Sine.similar_cosine_vector(m, cl, cosine_scaling_factor, k)
            x = Sine.similar_cosine_vector_x(alpha_pre, beta, m, cl, self.cosine_scaling_factor, k)
            if x is None:
                continue
            self.cosine_scaling_factor = 1
            cs_1, cs_2 = ((cs_mb * beta - m[k] * b[k] + x[0] * b[k]) / (beta - m[k] * m[k] + x[0] * x[0]),
                          (cs_mb * beta - m[k] * b[k] + x[1] * b[k]) / (beta - m[k] * m[k] + x[1] * x[1]))
            if cs_1 < cs_2:
                cs_mbk = cs_1
                x = x[0]
            else:
                cs_mbk = cs_2
                x = x[1]
            if cs_mbk < cs_mb:
                alpha_pre_, beta_ = Sine.update_param(alpha_pre, beta, m, cl, k, x)
                if cl_dist / self.norm_scaling_factor <= torch.sqrt(beta_) <= cl_dist * self.norm_scaling_factor:
                    cs_mb = cs_mbk
                    alpha_pre, beta = alpha_pre_, beta_
                    m[k] = x
        return m

    def gen_gradient(self, context: dict):
        model_state = context["model_state"]
        perturbation = torch.randn(self.m.size()).to(self.device)
        perturbation *= 1e-2
        return ef.restore_gradient_by_vector(model_state, self.m + perturbation)
