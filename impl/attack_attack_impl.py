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
        c_d, model, optim, lr, loss_fun, model_state, device = (context['c_d'], context['model'], context['optim'],
                                                                context['lr'], context['loss_func'],
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

# Sine 攻击
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

# 噪声攻击具体实现
class NoiseAttack(Attack):

    def __init__(self, mean: float = 0.0, stddev: float = 1.0):
        super().__init__()
        self.mean = mean
        self.stddev = stddev

    def gen_gradient(self, context: dict):
        c_d, model, optim, lr, loss_fun, model_state, device = (context['c_d'], context['model'], context['optim'],
                                                                context['lr'], context['loss_func'],
                                                                context['model_state'],
                                                                context['device'])

        # 生成 honest gradient（诚实的梯度）
        client_update, sum_loss, true_predict, size = train_client(c_d, model, optim, lr, loss_fun,
                                                                   model_state, device)

        # 为每个梯度参数生成噪声
        perturbed_gradient = {}
        for param_name, param_value in client_update.items():
            # 为每个参数生成与其形状一致的噪声
            noise = torch.normal(self.mean, self.stddev, size=param_value.shape, device=device)

            # 将噪声加到该参数的梯度上
            perturbed_gradient[param_name] = param_value + noise

        return perturbed_gradient


# Min-Max 攻击
class MinMaxAttack(Attack):

    def __init__(self, gamma: float = 1.0, p_grad: str = "std"):
        """
        初始化 Min-Max 攻击的 gamma 和扰动类型
        gamma: 缩放系数
        p_grad: 默认为标准差 std，表示选择如何生成扰动
        """
        super().__init__()
        self.gamma = gamma
        self.p_grad = p_grad  # 可以是 'std' 或者其他选项，如 'mean'

    def gen_gradient(self, context: dict):
        """
        生成 Min-Max 攻击的恶意梯度
        """

        c_d, model, optim, lr, loss_fun, model_state, device = (context['c_d'], context['model'], context['optim'],
                                                                context['lr'], context['loss_func'],
                                                                context['model_state'], context['device'])

        # 生成 honest gradient（诚实的梯度）
        client_update, sum_loss, true_predict, size = train_client(c_d, model, optim, lr, loss_fun,
                                                                   model_state, device)

        # 获取所有客户端的梯度
        benign_gradients = client_update  # 假设这个梯度是来自多个正常客户端

        # 计算标准差作为扰动大小
        gradient_list = [value.flatten() for value in benign_gradients.values()]
        all_gradients = torch.cat(gradient_list, dim=0)

        # 根据指定的扰动类型计算扰动
        if self.p_grad == "std":
            # 使用标准差的倒数作为扰动
            perturbation = torch.std(all_gradients) ** -1
        else:
            # 其他处理方式（例如mean）
            perturbation = torch.mean(all_gradients) ** -1

        # 生成恶意梯度，依照 Min-Max 攻击策略生成扰动
        malicious_gradient = {}
        for param_name, param_value in benign_gradients.items():
            # 生成扰动后的梯度
            gradient_shape = param_value.shape
            perturbation_vector = perturbation * torch.normal(self.gamma, 1, size=gradient_shape, device=device)

            # 根据Min-Max攻击策略生成扰动
            malicious_gradient[param_name] = param_value + perturbation_vector

        return malicious_gradient


# Min-Sum 攻击
class MinSumAttack(Attack):

    def __init__(self, gamma: float = 1.0, p_grad: str = "std"):
        """
        初始化 Min-Sum 攻击的 gamma 和扰动类型
        gamma: 缩放系数
        p_grad: 默认为标准差 std，表示选择如何生成扰动
        """
        super().__init__()
        self.gamma = gamma
        self.p_grad = p_grad  # 可以是 'std' 或者其他选项，如 'mean'

    def gen_gradient(self, context: dict):
        """
        生成 Min-Sum 攻击的恶意梯度
        """

        c_d, model, optim, lr, loss_fun, model_state, device = (context['c_d'], context['model'], context['optim'],
                                                                context['lr'], context['loss_func'],
                                                                context['model_state'], context['device'])

        # 生成 honest gradient（诚实的梯度）
        client_update, sum_loss, true_predict, size = train_client(c_d, model, optim, lr, loss_fun,
                                                                   model_state, device)

        # 获取所有客户端的梯度
        benign_gradients = client_update  # 假设这个梯度是来自多个正常客户端

        # 计算标准差作为扰动大小
        gradient_list = [value.flatten() for value in benign_gradients.values()]
        all_gradients = torch.cat(gradient_list, dim=0)

        # 根据指定的扰动类型计算扰动
        if self.p_grad == "std":
            # 使用标准差的倒数作为扰动
            perturbation = torch.std(all_gradients) ** -1
        else:
            # 其他处理方式（例如mean）
            perturbation = torch.mean(all_gradients) ** -1

        # 生成恶意梯度，依照 Min-Sum 攻击策略生成扰动
        malicious_gradient = {}
        for param_name, param_value in benign_gradients.items():
            # 生成扰动后的梯度
            gradient_shape = param_value.shape
            perturbation_vector = perturbation * torch.normal(self.gamma, 1, size=gradient_shape, device=device)

            # 根据Min-Sum攻击策略生成扰动
            malicious_gradient[param_name] = param_value + perturbation_vector

        return malicious_gradient



# LIE 攻击具体实现
class LIEAttack(Attack):

    def gen_gradient(self, context: dict):
        """
        生成 LIE 攻击的恶意梯度。

        参数:
            context (dict): 包含客户端数据、模型、优化器等信息的字典。

        返回:
            dict: 恶意梯度，与 model.state_dict() 结构一致。
        """
        # 从 context 中提取所需信息
        c_d, model, optim, lr, loss_fun, model_state, device = (
            context['c_d'],  # 客户端数据
            context['model'],  # 模型类
            context['optim'],  # 优化器类
            context['lr'],  # 学习率
            context['loss_func'],  # 损失函数
            context['model_state'],  # 全局模型状态
            context['device']  # 设备（CPU/GPU）
        )

        # 模拟诚实客户端训练，获取本地梯度更新
        client_update, sum_loss, true_predict, size = self.train_client(
            c_d, model, optim, lr, loss_fun, model_state, device
        )

        # 估计梯度的均值和标准差
        mean_grad, std_grad = self.estimate_grad_stats(client_update)

        # 获取恶意因子 z，若未提供则使用默认值 1.0  可以自己设置调整 参考论文设置0.3
        z = context.get('attack_factor', 1.0)

        # 生成恶意梯度
        malicious_update = self.apply_lie_attack(mean_grad, std_grad, z,client_update,device)

        return malicious_update

    def train_client(self, client_data, model, optim, lr, loss_fun, model_state, device):
        """
        模拟诚实客户端的训练过程，计算梯度更新。
        """
        # 初始化本地模型并加载全局模型状态
        local_model = model().to(device)
        local_model.load_state_dict(model_state)
        optimizer = optim(local_model.parameters(), lr=lr)

        sum_loss = 0
        true_predict = 0
        size = 0

        # 遍历客户端数据进行训练
        for data in client_data:
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            _, predicted = torch.max(outputs.data, 1)
            sum_loss += loss.data
            true_predict += torch.sum(predicted == labels.data)
            size += len(labels)

        # 计算梯度更新（全局模型状态 - 本地模型状态）
        client_update = self.sub(model_state, local_model.state_dict())
        return client_update, sum_loss, true_predict, size

    def estimate_grad_stats(self, client_update):
        """
        估计梯度更新的均值和标准差。

        参数:
            client_update (dict): 本地梯度更新

        返回:
            tuple: (均值字典, 标准差字典)
        """
        mean_grad = {}
        std_grad = {}
        for key in client_update.keys():
            grad_tensor = client_update[key]
            mean_grad[key] = torch.mean(grad_tensor)
            std_grad[key] = torch.std(grad_tensor) if torch.std(grad_tensor) != 0 else torch.tensor(1e-5)
        return mean_grad, std_grad

    def apply_lie_attack(self, mean_grad, std_grad, z, client_update,device):
        malicious_update = {}
        for key in client_update.keys():
            shape = client_update[key].shape  # 获取原始形状
            # 确保 client_update[key] 在 GPU 上
            client_update[key] = client_update[key].to(device)
            # 计算 value，确保 mean_grad[key] 和 std_grad[key] 在同一设备上
            value = mean_grad[key] - z * std_grad[key]
            # 创建与 value 相同设备的 ones 张量
            ones_tensor = torch.ones(shape, device=device)
            malicious_update[key] = value * ones_tensor
        return malicious_update

    def sub(self, state_dict1, state_dict2):
        """
        计算两个状态字典的差。
        """
        return {k: state_dict1[k] - state_dict2[k] for k in state_dict1.keys()}
