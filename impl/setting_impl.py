from torchvision import transforms

from meta_define.setting import Setting
from util import get_class_dict

import impl.fl_framework_impl as framework
import impl.fl_task_impl as task
import impl.attack_attack_impl as attack
import impl.attack_strategic_impl as strategic
import torch.nn as nn
import torch.optim as optim
import experiment.models as model
import torchvision.datasets as dataset

from util.non_iid_dataset import NonIIDDataset

frameworks = get_class_dict(framework)
tasks = get_class_dict(task)
attacks = get_class_dict(attack)
strategics = get_class_dict(strategic)
nns = get_class_dict(nn)
optims = get_class_dict(optim)
models = get_class_dict(model)
datasets = get_class_dict(dataset)


class FrameworkSetting(Setting):
    order = 1

    def get_path(self) -> str:
        return "framework"

    def interpret(self, content, context: dict):
        context["framework"] = frameworks[content]()


class FLSetting(Setting):
    order = 2

    def get_path(self) -> str:
        return "fl_settings"

    def interpret(self, content, context: dict):
        root = context["fl_settings"]
        root["loss_func"] = nns[content["loss_func"]]()
        root["optim"] = optims[content["optim"]]
        root["task"] = tasks[content["task"]]()
        root["model"] = models[content["model"]]
        root["dataset"] = FLSetting.load_dataset(content)

    @staticmethod
    def load_dataset(content):
        data_root, dataset = content["data_root"], datasets[content["dataset"]]
        train_dataset = dataset(root=data_root, train=True,
                                transform=transforms.ToTensor(), download=True)
        test_dataset = dataset(root=data_root, train=False,
                               transform=transforms.ToTensor(), download=True)
        if "data_heterogeneity_settings" in content:
            n = content["client_size"]
            alpha = content["data_heterogeneity_settings"]["alpha"]
            train_dataset = NonIIDDataset(train_dataset)
            train_dataset.shuffle(alpha, n)
            test_dataset = NonIIDDataset(test_dataset)
            test_dataset.shuffle(alpha, n)
        return train_dataset, test_dataset


class ByzantineSetting(Setting):
    order = 3

    def get_path(self) -> str:
        return "byzantine_settings"

    def interpret(self, content, context: dict):
        root = context["byzantine_settings"]
        root["byzantine_attack"] = attacks[content["byzantine_attack"]]()
        root["byzantine_defend"] = strategics[content["byzantine_defend"]]()
        root["byzantine_defend"].assign(context["fl_settings"])
