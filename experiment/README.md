# Experiment Module

This directory contains the code for running federated learning experiments using the FL_ExP framework.

## Files

### models.py
Contains model definitions used in experiments, such as CNN models for CIFAR-10.

### simulation.py
The main entry point for running single experiments. It provides functions for:
- `train()`: Standard federated learning training
- `train_with_attack()`: Federated learning with Byzantine attacks and defenses
- `start()`: Entry point that parses configuration and starts the experiment

### multiple_simulation.py
Supports running multiple experiments with different parameters for comparative analysis.

## Usage

To run a single experiment:
```bash
python -m experiment.simulation
```

To run multiple experiments:
```bash
python -m experiment.multiple_simulation
```

## Configuration

Experiments are configured using JSON files in the `config/` directory. The configuration specifies:
- The federated learning framework
- Byzantine attack and defense settings
- Model and dataset settings
- Training parameters

See the `config/` directory README for more details on configuration options.