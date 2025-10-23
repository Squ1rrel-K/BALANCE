# Meta Define Module

This directory contains the abstract base classes that define the core components of the FL_ExP framework. These abstract classes serve as interfaces that concrete implementations must follow.

## Overview

The meta_define module provides the foundation for the entire framework through well-defined interfaces. By inheriting from these abstract classes, you can create custom implementations of attacks, defenses, federated learning frameworks, and more.

## Files

### attack.py

Contains abstract classes for Byzantine attacks and defense strategies:

- `Attack`: Base class for implementing Byzantine attacks
  - Key method: `gen_gradient(context)` - Generates poisoned gradients
  
- `Strategic`: Base class for implementing defense strategies
  - Key methods: 
    - `examine(context)` - Detects poisoned gradients
    - `aggregate(context)` - Aggregates gradients after filtering out poisoned ones

### fl.py

Contains abstract classes for federated learning components:

- `TaskType`: Base class for defining task types
  - Key methods:
    - `train(context)` - Training logic for a specific task
    - `eval(context)` - Evaluation logic for a specific task

- `ClientTrain`: Base class for client training behavior
  - Key method: `train(context)` - Defines how clients train their local models

- `Framework`: Base class for federated learning frameworks
  - Key methods:
    - `train_clients(client_train, context)` - Coordinates client training
    - `aggregate(context)` - Aggregates client updates
    - `evaluate(context)` - Evaluates the global model
    - `update(context)` - Updates the global model
    - `load()` - Loads the model
    - `get_global_model()` - Returns the global model

### setting.py

Contains the abstract class for configuration settings:

- `Setting`: Base class for custom configuration file parsing

## How to Create Custom Components

To create a custom component, follow these steps:

1. Import the appropriate abstract class from this module
2. Create a new class that inherits from the abstract class
3. Implement all required abstract methods
4. Register your implementation in the appropriate location

### Example: Creating a Custom Attack

```python
from meta_define.attack import Attack

class MyCustomAttack(Attack):
    def __init__(self, param1=1.0):
        super().__init__()
        self.param1 = param1
    
    def gen_gradient(self, context: dict):
        # Implement your attack logic here
        # Access normal gradients and modify them
        # Return poisoned gradients
        pass
```

### Example: Creating a Custom Defense

```python
from meta_define.attack import Strategic

class MyCustomDefense(Strategic):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def examine(self, context: dict) -> list:
        # Implement your detection logic here
        # Return a list where 1 indicates poisoned gradient, 0 indicates benign
        pass
    
    def aggregate(self, context: dict) -> list:
        # Implement your aggregation logic here
        # Filter out poisoned gradients and aggregate the rest
        # Return the filtered list of gradients
        pass
```

## Integration with the Framework

After creating your custom components, you can use them in experiments by specifying them in the configuration files. The framework will automatically load and use your implementations.