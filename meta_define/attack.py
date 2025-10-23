from abc import ABCMeta, abstractmethod


class Attack:
    """
    Abstract base class for Byzantine attack strategies in federated learning.
    Implementations should define how to generate poisoned gradients.
    """
    __meta_class__ = ABCMeta

    def __init__(self):
        """Initialize attack with empty context."""
        self.context = {}

    def assign(self, context: dict):
        """
        Assign additional context information to the attack.
        Used to read configuration settings or other information.
        For example, when clients can collude in an attack.
        
        Args:
            context (dict): Dictionary containing context information
        """
        self.context.update(context)

    @abstractmethod
    def gen_gradient(self, context: dict):
        """
        Generate poisoned gradients.
        
        Args:
            context (dict): Dictionary containing gradient generation context
                            such as model, data, optimizer, etc.
        
        Returns:
            dict: Poisoned gradients in a structure similar to model.state_dict()
        """
        pass

class Strategic:
    """
    Abstract base class for Byzantine defense strategies in federated learning.
    Implementations should define how to detect poisoned gradients and
    how to aggregate client updates securely.
    """
    __meta_class__ = ABCMeta

    def __init__(self):
        """Initialize defense strategy with empty context."""
        self.context = {}

    def assign(self, context: dict):
        """
        Assign additional context information to the defense strategy.
        
        Args:
            context (dict): Dictionary containing context information
        """
        self.context.update(context)

    @abstractmethod
    def examine(self, context: dict) -> list:
        """
        Examine gradients to detect poisoned ones.
        
        Args:
            context (dict): Dictionary containing examination context
                            with 'gradients' from all clients
        
        Returns:
            list: Binary list where 1 indicates poisoned gradient and 0 indicates honest gradient
        """
        pass

    @abstractmethod
    def aggregate(self, context: dict) -> list:
        """
        Design secure aggregation algorithm to mitigate Byzantine attacks.
        
        Args:
            context (dict): Dictionary containing aggregation context
                            with 'updates' from all clients
        
        Returns:
            list: Securely aggregated updates
        """
        pass
