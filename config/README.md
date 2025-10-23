# Configuration Directory

This directory contains JSON configuration files for FL_ExP experiments.

## Configuration Files

- `test.json`: Basic configuration for testing Byzantine-robust federated learning
- `multi_test.json`: Configuration for running multiple test experiments
- `multi_cen.json`: Configuration for centralized experiments with multiple parameters

## Configuration Structure

A typical configuration file has the following structure:

```json
{
    "framework": "FedAvg",                   // Federated learning framework
    "byzantine_settings": {                  // Byzantine attack and defense settings
        "byzantine_attack": "LabelFlipping", // Attack algorithm
        "byzantine_defend": "Krum",          // Defense algorithm
        "adv_rate": 0.3                      // Proportion of adversarial clients
    },
    "fl_settings": {                         // Federated learning settings
        "model": "Model_CIFAR10",            // Model architecture
        "task": "Classification",            // Task type
        "data_root": "../data",              // Data directory
        "dataset": "CIFAR10",                // Dataset name
        "data_heterogeneity_settings": {     // Non-IID settings
            "alpha": 0.1                     // Dirichlet distribution parameter
        },
        "loss_func": "CrossEntropyLoss",     // Loss function
        "optim": "Adam",                     // Optimizer
        "device": "cuda",                    // Device (cuda/cpu)
        "client_size": 10,                   // Number of clients
        "epochs": 50,                        // Number of training epochs
        "lr": 1e-3                           // Learning rate
    },
    "other_settings": {                      // Other settings
        "report_dir": "../logs",             // Directory for saving results
        "report_name": "test"                // Name of the result file
    }
}
```

## Creating Custom Configurations

To create a custom configuration:
1. Copy an existing configuration file
2. Modify the parameters according to your experiment needs
3. Save it with a descriptive name
4. Run the experiment using your new configuration file