from torch.autograd import Variable
from torch.utils.data import DataLoader

from meta_define.fl import Framework, ClientTrain
import util.easy_fed as ef


class FedAvg(Framework):
    """
    Implementation of the Federated Averaging (FedAvg) algorithm.
    This is the standard federated learning algorithm that aggregates client updates
    by weighted averaging.
    """

    def evaluate(self, context: dict) -> dict:
        """
        Evaluate the global model on the test dataset.
        
        Args:
            context (dict): Dictionary containing evaluation context with 'task'
            
        Returns:
            dict: Dictionary containing test accuracy and loss
        """
        sum_correct = 0
        sum_loss = 0
        sum_size = 0
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=100, shuffle=False)
        for data in test_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
            correct, loss, size = context["task"].eval({"model": self.global_model, "data": inputs, "target": labels,
                                                        "loss_func": self.loss_func, "device": self.device})
            sum_correct += correct
            sum_loss += loss
            sum_size += size
        return {"test_acc": 100 * sum_correct / sum_size, "test_loss": sum_loss / sum_size}

    def __init__(self):
        """Initialize FedAvg with empty global model."""
        super().__init__()
        self.global_model = None

    def load(self):
        """Load the global model to the specified device."""
        self.global_model = self.model().to(self.device)

    def get_global_model(self):
        """
        Get the current global model.
        
        Returns:
            The global model
        """
        return self.global_model

    def update(self, context: dict):
        """
        Update the global model with aggregated updates.
        
        Args:
            context (dict): Dictionary containing 'update' with aggregated model updates
        """
        ef.update_by_state_dict(self.global_model, context["update"], lr=1)

    def train_clients(self, client_train: ClientTrain, context: dict):
        """
        Coordinate client training and collect updates.
        
        Args:
            client_train (ClientTrain): Client training implementation
            context (dict): Dictionary containing training context
            
        Returns:
            list: List of client updates
        """
        updates = []
        client_data = context["client_data"]
        context["model_state"] = self.global_model.state_dict()
        n = len(client_data)
        for ci in range(n):
            c_d = client_data[ci]
            context["i"] = ci
            context["c_d"] = c_d
            client_update = client_train.train(context)
            updates.append(client_update)
        return updates

    def aggregate(self, context: dict) -> list:
        updates = context["updates"]
        model_state = self.global_model.state_dict()
        update = ef.sub(model_state, model_state)
        size_of_updates = len(updates)
        for i in range(size_of_updates):
            update = ef.add(update, ef.mul_constant(updates[i], 1 / size_of_updates))
        return update
