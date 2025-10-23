from abc import ABCMeta, abstractmethod
import torch


class TaskType:
    """
    Abstract base class for defining task types in federated learning.
    Implementations should define training and evaluation procedures.
    """
    __meta_class__ = ABCMeta

    @abstractmethod
    def train(self, context: dict):
        """
        Abstract method for training procedure.
        
        Args:
            context (dict): Dictionary containing training context information
                            such as model, data, optimizer, etc.
        """
        pass

    @abstractmethod
    def eval(self, context: dict):
        """
        Abstract method for evaluation procedure.
        
        Args:
            context (dict): Dictionary containing evaluation context information
                            such as model, data, metrics, etc.
        """
        pass

class ClientTrain:
    """
    Abstract base class for client training in federated learning.
    Defines how clients perform local training.
    """
    __meta_class__ = ABCMeta

    def __init__(self):
        """Initialize client training with no task type."""
        self.task_type: TaskType = None

    def set_task_type(self, task_type: TaskType):
        """
        Set the task type for this client.
        
        Args:
            task_type (TaskType): The task type to be used for training
        """
        self.task_type = task_type

    @abstractmethod
    def train(self, context: dict):
        """
        Abstract method for client training procedure.
        
        Args:
            context (dict): Dictionary containing client training context
                            such as client data, model, optimizer, etc.
        """
        pass

class Framework:
    """
    Abstract base class for federated learning frameworks.
    Defines the core operations of a federated learning system.
    """
    __meta_class__ = ABCMeta

    def __init__(self):
        """Initialize framework with empty model and settings."""
        self.model: torch.nn.Module = None
        self.device = None
        self.loss_func = None
        self.test_dataset = None

    @abstractmethod
    def train_clients(self, client_train: ClientTrain, context: dict):
        """
        Abstract method for coordinating client training.
        
        Args:
            client_train (ClientTrain): Client training implementation
            context (dict): Dictionary containing training context
        """
        pass

    @abstractmethod
    def aggregate(self, context: dict) -> list:
        """
        Abstract method for aggregating client updates.
        
        Args:
            context (dict): Dictionary containing aggregation context
                            such as client updates, weights, etc.
        
        Returns:
            list: Aggregated updates
        """
        pass

    @abstractmethod
    def evaluate(self, context: dict) -> dict:
        """
        Abstract method for evaluating global model.
        
        Args:
            context (dict): Dictionary containing evaluation context
        
        Returns:
            dict: Evaluation metrics
        """
        pass

    @abstractmethod
    def update(self, context: dict):
        """
        Abstract method for updating global model with aggregated updates.
        
        Args:
            context (dict): Dictionary containing update context
                            such as aggregated updates
        """
        pass

    @abstractmethod
    def load(self):
        """
        Abstract method for loading global model.
        """
        pass

    @abstractmethod
    def get_global_model(self):
        """
        Abstract method for retrieving the global model.
        
        Returns:
            The global model
        """
        pass