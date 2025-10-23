# FL_ExP (Federated Learning Experiment Platform)

## Introduction

FL_ExP is a comprehensive experimental framework for federated learning research, with built-in implementations of various algorithms that can be customized and extended. This framework is designed to facilitate research in Byzantine-robust federated learning and other federated learning scenarios.

### Key Features

- Byzantine-robustness experiments (customizable attack and defense algorithms)
- Customizable federated learning framework implementations
- Non-i.i.d. data distribution settings
- Differential privacy support
- Modular and extensible architecture

### Built-in Implementations

- **Attacks**: LF (Label Flipping) attack, Sine attack, Noise attack, etc.
- **Defenses**: Krum, FLTrust, etc.
- **Frameworks**: FedAvg

## Project Structure

The project is organized into several key directories:

- `meta_define/`: Abstract class definitions for the core components
- `impl/`: Concrete implementations of the abstract classes
- `experiment/`: Experiment execution and model definitions
- `util/`: Utility functions and helper classes
- `config/`: Configuration files for experiments

See the README.md file in each directory for detailed information.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib (for visualization)

See `requirements.txt` for the complete list of dependencies.

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Experiment**:
   Edit the configuration files in the `config/` directory to set up your experiment.
   
   Example configuration (`config/test.json`):
   ```json
   {
       "framework": "FedAvg",
       "byzantine_settings": {
           "byzantine_attack": "LabelFlipping",
           "byzantine_defend": "Krum",
           "adv_rate": 0.3
       },
       "fl_settings": {
           "model": "Model_CIFAR10",
           "task": "Classification",
           "data_root": "../data",
           "dataset": "CIFAR10",
           "data_heterogeneity_settings": {
               "alpha": 0.1
           },
           "loss_func": "CrossEntropyLoss",
           "optim": "Adam",
           "device": "cuda",
           "client_size": 10,
           "epochs": 50,
           "lr": 1e-3
       },
       "other_settings": {
           "report_dir": "../logs",
           "report_name": "test"
       }
   }
   ```

3. **Run Experiment**:
   ```bash
   python -m experiment.simulation
   ```

4. **Run Multiple Experiments**:
   ```bash
   python -m experiment.multiple_simulation
   ```

## Customization Guide

### Meta Definitions

The `meta_define` package contains abstract classes that define the core components of the framework:

- `attack.py`:
  - `Attack`: Base class for implementing Byzantine attacks
  - `Strategic`: Base class for implementing defense strategies

- `fl.py`:
  - `TaskType`: Base class for defining task types (classification, regression, etc.)
  - `ClientTrain`: Base class for client training behavior (normal client, adversarial client)
  - `Framework`: Base class for federated learning frameworks (FedAvg, etc.)

- `setting.py`: Base class for custom configuration file parsing

### Creating Custom Components

To create custom components, inherit from the appropriate abstract class and implement the required methods. See the README files in each directory for detailed instructions.

## License

This project is open-source and available under the MIT License.

## Citation

If you use this framework in your research, please cite:

```
@software{FL_ExP,
  author = {FL_ExP Contributors},
  title = {FL_ExP: A Federated Learning Experiment Platform},
  year = {2023},
  url = {https://github.com/yourusername/FL_ExP}
}
```

## Contact

For questions or feedback, please open an issue on the GitHub repository.