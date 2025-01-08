import math
from collections import OrderedDict

import torch
from torch import nn



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


if __name__ == '__main__':
    from performance.models import Model_CIFAR10
    model = Model_CIFAR10()
    print(model_size(model.state_dict()))

    v = flatten_all_gradient(model.state_dict())
    print(v.size())

    g = restore_gradient_by_vector(model.state_dict(), v)
    print(g)

