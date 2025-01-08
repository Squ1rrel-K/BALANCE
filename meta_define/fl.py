from abc import ABCMeta, abstractmethod
import torch


class TaskType:
    __meta_class__ = ABCMeta

    @abstractmethod
    def train(self, context: dict):
        pass

    @abstractmethod
    def eval(self, context: dict):
        pass

class ClientTrain:
    __meta_class__ = ABCMeta

    def __init__(self):
        self.task_type: TaskType = None

    def set_task_type(self, task_type: TaskType):
        self.task_type = task_type

    @abstractmethod
    def train(self, context: dict):
        pass

class Framework:
    __meta_class__ = ABCMeta

    def __init__(self):
        self.model: torch.nn.Module = None
        self.device = None
        self.loss_func = None
        self.test_dataset = None

    @abstractmethod
    def train_clients(self, client_train: ClientTrain, context: dict):
        pass

    @abstractmethod
    def aggregate(self, context: dict) -> list:
        pass

    @abstractmethod
    def evaluate(self, context: dict) -> dict:
        pass

    @abstractmethod
    def update(self, context: dict):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_global_model(self):
        pass