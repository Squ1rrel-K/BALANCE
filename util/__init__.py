import numpy as np
import torch
import inspect
from torch.utils.data import DataLoader

from meta_define.setting import Setting


def cover(context: dict) -> dict:
    result = {}
    result.update(context)
    return result


def get_classes(module, class_type=None):
    classes = []
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if class_type is None or issubclass(obj, class_type):
                classes.append(obj)
    return classes


def get_class_dict(module):
    classes = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            classes[name] = obj
    return classes


def split_dataset_to_client(n, dataset):
    clients = []
    data_size = int(len(dataset) / n)
    for i in range(n):
        train_sub = torch.utils.data.Subset(dataset, range(i * data_size, (i + 1) * data_size))
        clients.append(DataLoader(dataset=train_sub, batch_size=32, shuffle=True))
    return clients


def auto_match(setting: Setting, args: dict, context: dict):
    path = setting.get_path()
    if path is not None:
        keys = path.split("/")
        root = args
        for i in range(len(keys)):
            key = keys[i]
            if isinstance(root, dict) and key in root.keys():
                if i == len(keys) - 1:
                    # try:
                        setting.interpret(args[key], context)
                    # except:
                    #     print("Failed to load", "'"+key+"'.", "Please check the configuration file and try again.")
                    #     exit(-1)
                    # pass
                else:
                    root = root[key]


def create_classify(id, size):
    result = []
    for i in range(size):
        if i == id:
            result.append(1)
        else:
            result.append(0)
    return result


def count_loss(loss_func, preds, targets, device="cuda", size=10):
    try:
        return loss_func(preds, targets)
    except:
        result = []
        for t in targets:
            result.append(create_classify(int(t), size=size))
        result = torch.from_numpy(np.array(result, dtype=np.float32)).to(device)
        return loss_func(preds, result)


def to_array(input_array, array_type=0):
    if array_type == 0:
        if isinstance(input_array, np.ndarray):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            return input_array.detach().numpy()
        return np.array(input_array)
    elif array_type == 1:
        if isinstance(input_array, np.ndarray):
            return torch.from_numpy(np.array(input_array))
        elif isinstance(input_array, torch.Tensor):
            return input_array
        elif isinstance(input_array, list):
            try:
                result = torch.from_numpy(np.array(input_array))
                return result
            except:
                try:
                    result = torch.tensor(input_array)
                    return result
                except:
                    return torch.from_numpy(np.array([item.cpu().detach().numpy() for item in input_array]))
        return torch.from_numpy(np.array(input_array))


def util(array_type=0):
    if array_type == 0:
        return np
    else:
        return torch


set_array_type = 1
array_function = lambda input_array: to_array(input_array, set_array_type)
array_util = util(set_array_type)
