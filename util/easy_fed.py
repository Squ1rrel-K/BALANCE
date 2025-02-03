import math
from collections import OrderedDict

import torch
from matplotlib.font_manager import weight_dict
from torch import nn
import numpy as np
from experiment.models import Model_CIFAR10,ResNet18


def hadmard_product(state_dict_1, state_dict_2):
    result = OrderedDict()
    for name in state_dict_1:
        result[name] = state_dict_1[name] * state_dict_2[name]
    return result

def sum_all(state_dict):
    result = 0
    for name in state_dict:
        result += state_dict[name].sum()
    return result

def mul_constant(state_dict, c):
    result = OrderedDict()
    for name in state_dict:
        result[name] = state_dict[name] * c
    return result

def transfer(state_dict, rule):
    result = OrderedDict()
    for name in state_dict:
        result[rule(name)] = state_dict[name]
    return result
def zeros(state_dict):
    return sub(state_dict, state_dict)
def sub(state_dict_1, state_dict_2):
    result = OrderedDict()
    for name in state_dict_1:
        result[name] = state_dict_1[name] - state_dict_2[name]
    return result

def copy(state_dict):
    result = OrderedDict()
    for name in state_dict:
        result[name] = state_dict[name].clone()
    return result

def add(state_dict_1, state_dict_2):
    result = OrderedDict()
    for name in state_dict_1:
        result[name] = state_dict_1[name] + state_dict_2[name]
    return result


def update_by_state_dict(model: torch.nn.Module, update, lr):
    sd = model.state_dict()
    model.load_state_dict(sub(sd, mul_constant(update, lr)))


def cos(state_dict_1, state_dict_2):
    v = 0
    l1 = 0
    l2 = 0
    for name in state_dict_1:
        v += torch.sum(state_dict_1[name] * state_dict_2[name])
        l1 += torch.sum(state_dict_1[name] * state_dict_1[name])
        l2 += torch.sum(state_dict_2[name] * state_dict_2[name])
    return v / math.sqrt(l1 * l2), math.sqrt(l1), math.sqrt(l2)


def l2(state_dict_1, state_dict_2):
    result = 0
    for name in state_dict_1:
        mid = state_dict_1[name] - state_dict_2[name]
        result += torch.sum(mid * mid)
    return math.sqrt(result)

def l2_self(state_dict):
    result = 0
    for name in state_dict:
        mid = state_dict[name]
        result += torch.sum(mid * mid)
    return math.sqrt(result)


def restore_gradient_by_vector(state_dict, vector):
    result = OrderedDict()
    start_index = 0
    for name in state_dict:
        bias = state_dict[name].numel()
        result[name] = torch.reshape(vector[start_index: start_index+bias], state_dict[name].size())
        start_index += bias
    return result

def flatten_all_gradient(state_dict):
    result = []
    for name in state_dict:
        result.append(torch.reshape(state_dict[name], (-1,)))
    return torch.cat(result, 0)

def flatten_seg_gradient(state_dict):
    result = []
    for name in state_dict:
        result.append(torch.reshape(state_dict[name], (-1,)))
    return result

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def model_size(state_dict):
    result = 0
    for name in state_dict:
        result += state_dict[name].numel()
    return result

# 模型参数选择 是 weights 还是 bias
def model_select(state_dict, select_word):
    result = copy(state_dict)
    for name in state_dict:
        if select_word not in name:
            del result[name]
    return result


def dict_to_list(state_dict):
    result = []
    for name in state_dict:
        result.append(state_dict[name])
    return result

# model.state_dict()   key 代表每一层 value 是tensor张量

def set_client_params(num_client):
    result = []
    for i in range(num_client):
        model = ResNet18()
        result.append(model.state_dict())
    return result


# 分层权重reshape
def reshape_weights(weights):
    """
    通用方法将每个客户端的模型参数展平，适应多种形状的参数。
    如果参数为 4D（通常是卷积层权重），则展平每个卷积核；如果是 2D 则保持原始形状。
    """
    reshaped_weights = []
    for client_id, client_weights in enumerate(weights):
        reshaped_client = {}
        for layer_name, param in client_weights.items():
            if param.ndimension() == 4:  # 如果是 4D 卷积层权重
                # 展平每个卷积核，将形状 (out_channels, in_channels, height, width) 转换为 (out_channels, in_channels * height * width)
                reshaped_client[layer_name] = param.view(param.size(0), -1)
            elif param.ndimension() == 2:  # 如果是 2D 全连接层权重
                # 保持原始形状
                reshaped_client[layer_name] = param
            else:
                # 如果参数维度不符合预期，保留原始形状（或记录异常）
                reshaped_client[layer_name] = param
        reshaped_weights.append(reshaped_client)

    return reshaped_weights



# 分层矩阵聚合
def matrix_dnc_aggregation(client_matrices, num_clients, c, n_iters):
    good_clients = set(range(num_clients))
    for _ in range(n_iters):
        # 计算所有客户端矩阵的均值矩阵
        mean_matrix = sum(client_matrices) / len(client_matrices)

        # 计算中心化矩阵
        centered_matrices = [matrix - mean_matrix for matrix in client_matrices]

        # 叠加中心化矩阵
        stacked_matrix = sum(centered_matrices)

        # 计算叠加矩阵的 SVD
        U, S, Vh = torch.linalg.svd(stacked_matrix, full_matrices=False)
        top_singular_vector = Vh[0, :]  # 提取第一右奇异向量

        # 计算每个矩阵的异常分数
        outlier_scores = [
            torch.norm(torch.mm(centered_matrix, top_singular_vector.unsqueeze(1)))**2
            for centered_matrix in centered_matrices
        ]

        # 根据异常分数筛选正常客户端
        num_to_keep = int((1 - c) * num_clients)
        top_indices = np.argsort(outlier_scores)[:num_to_keep]
        good_clients = good_clients.intersection(set(top_indices))

    return good_clients

# 分层向量聚合
def vector_dnc_aggregation(client_vectors, num_clients, c, n_iters):
    good_clients = set(range(num_clients))

    for _ in range(n_iters):
        # 1. 拼接所有客户端的向量为一个矩阵，每个客户端的向量作为矩阵的一行
        stacked_matrix = torch.stack(client_vectors)

        # 2. 计算均值向量并中心化每个客户端的向量
        mean_vector = torch.mean(stacked_matrix, dim=0)
        centered_vectors = [vector - mean_vector for vector in client_vectors]

        # 3. 叠加中心化向量
        stacked_centered_vectors = torch.stack(centered_vectors)

        # 4. 计算叠加矩阵的 SVD
        U, S, Vh = torch.linalg.svd(stacked_centered_vectors, full_matrices=False)
        top_singular_vector = Vh[0, :]  # 提取第一右奇异向量

        # 5. 计算每个向量的异常分数
        outlier_scores = [
            torch.norm(torch.mm(centered_vector.unsqueeze(0), top_singular_vector.unsqueeze(1))) ** 2
            for centered_vector in stacked_centered_vectors
        ]

        # 6. 根据异常分数筛选正常客户端
        num_to_keep = int((1 - c) * num_clients)
        top_indices = np.argsort(outlier_scores)[:num_to_keep]
        good_clients = good_clients.intersection(set(top_indices))

    return good_clients


if __name__ == '__main__':
    num_clients = 10
    # 包含所有客户端的模型参数
    updates = set_client_params(num_clients)
    weights = []
    for client in updates:
        weights.append(model_select(client, "weight"))
    # weights 包含 所有客户端的梯度  是OrderedDict 形状
    num = 1
    weights = reshape_weights(weights)
    # 现在每一层梯度都变成矩阵的形式
    for weight in weights:
        print("第"+ str(num) +"客户端参数形状")
        num += 1
        for g in dict_to_list(weight):
            print(g.shape)
    # 现在模拟实现分层的 Dnc 聚合
        # 聚合参数
        c = 0.2  # 恶意客户端比例
        n_iters = 5  # 迭代次数
        # 分层 DnC 聚合
        for layer_idx in weights[0].keys():  # 遍历每一层
            # 提取每个客户端该层的参数矩阵
            client_matrices = [client[layer_idx] for client in weights]
            # 这里进行判断 是一维还是多维 一维用vector_dnc_aggregation  多维用matrix_dnc_aggregation
            if client_matrices[0].dim() == 1:  # 1D vector (e.g., a weight vector)
                # Perform DNC aggregation for vectors
                good_indices = vector_dnc_aggregation(client_matrices, num_clients, c, n_iters)
            else:  # Multi-dimensional matrix (e.g., weight matrices for layers)
                # Perform DNC aggregation for matrices
                good_indices = matrix_dnc_aggregation(client_matrices, num_clients, c, n_iters)
            # 获取良性客户端索引
            #good_indices = matrix_dnc_aggregation(client_matrices, num_client, c, n_iters)

            # 将异常客户端的该层参数置为 0
            for i in range(num_clients):
                if i not in good_indices:
                    weights[i][layer_idx] = torch.zeros_like(weights[i][layer_idx])

        # 输出结果
        for client_id, client_weights in enumerate(weights):
            print(f"第{client_id + 1}客户端最终参数形状:")
            for layer_name, param in client_weights.items():
                print(f"{layer_name}: {param.shape}")

