# FeRA CLI Tool Usage Guide

## Overview

The `fera_cli.py` script provides a convenient command-line interface for running FeRA defense experiments with BackFed. It includes support for start/end epochs, multi-layer detection, and comprehensive configuration options.

## Quick Start

### Basic Usage
```bash
# Basic pattern attack detection
python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101

# Multi-layer detection
python fera_cli.py --dataset cifar10 --attack pattern --multi-layer --layers layer2,layer3,penultimate

# Conservative detection
python fera_cli.py --dataset cifar10 --attack pattern --top-k 0.2 --spectral-weight 0.7 --delta-weight 0.3
```

## Command Line Options

### Required Arguments
- `--dataset, -d`: Dataset to use (`cifar10`, `emnist`, `femnist`, `tinyimagenet`, `reddit`, `sentiment140`)
- `--attack, -a`: Attack method (`pattern`, `pixel`, `badnets`, `blended`, `distributed`, `edge_case`, `a3fl`, `iba`, `neurotoxin`, `chameleon`, `anticipate`)

### Epoch Configuration
- `--start-epoch, -s`: Starting epoch for attack (default: 1001)
- `--end-epoch, -e`: Ending epoch for attack (default: 1101)

### FeRA Parameters
- `--spectral-weight`: Weight for spectral norm signal (default: 0.6)
- `--delta-weight`: Weight for delta norm signal (default: 0.4)
- `--top-k`: Top-K percent to flag as malicious (default: 0.5)
- `--root-size`: Size of root dataset for feature extraction (default: 64)

### Multi-Layer Options
- `--multi-layer`: Enable multi-layer feature extraction
- `--layers`: Comma-separated list of layers (`penultimate,layer2,layer3,layer4`)
- `--combine-method`: Method to combine multi-layer scores (`mean`, `max`, `vote`)

### Training Configuration
- `--num-rounds`: Total number of training rounds (default: 600)
- `--num-clients`: Total number of clients (default: 100)
- `--num-clients-per-round`: Number of clients selected per round (default: 10)
- `--training-mode`: Training mode (`parallel`, `sequential`)
- `--num-gpus`: GPU fraction per client (default: 1.0)
- `--num-cpus`: CPU cores per client (default: 12)

### Checkpoint and Logging
- `--checkpoint`: Path to checkpoint file or round number to resume from
- `--save-logging`: Logging method (`csv`, `wandb`, `both`)
- `--dir-tag`: Custom directory tag for organizing results

### Advanced Options
- `--debug`: Enable debug mode (use subset of data)
- `--debug-fraction`: Fraction of data to use in debug mode (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 123)
- `--alpha`: Dirichlet distribution parameter for non-IID data (default: 0.9)

### Output Options
- `--dry-run`: Print the command that would be executed without running it
- `--verbose, -v`: Enable verbose output

## Examples

### 1. Basic Pattern Attack Detection
```bash
python fera_cli.py \
    --dataset cifar10 \
    --attack pattern \
    --start-epoch 1001 \
    --end-epoch 1101
```

### 2. Multi-Layer Detection Against Neurotoxin
```bash
python fera_cli.py \
    --dataset cifar10 \
    --attack neurotoxin \
    --multi-layer \
    --layers layer2,layer3,penultimate \
    --combine-method max
```

### 3. Conservative Detection with Custom Parameters
```bash
python fera_cli.py \
    --dataset emnist \
    --attack pattern \
    --top-k 0.2 \
    --spectral-weight 0.7 \
    --delta-weight 0.3 \
    --root-size 128
```

### 4. Full Experiment with All Options
```bash
python fera_cli.py \
    --dataset cifar10 \
    --attack pattern \
    --start-epoch 1001 \
    --end-epoch 1601 \
    --multi-layer \
    --layers layer2,layer3,penultimate \
    --combine-method max \
    --top-k 0.3 \
    --spectral-weight 0.6 \
    --delta-weight 0.4 \
    --root-size 128 \
    --checkpoint checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth \
    --num-rounds 600 \
    --training-mode parallel \
    --num-gpus 1.0 \
    --save-logging csv
```

### 5. Debug Mode (Quick Test)
```bash
python fera_cli.py \
    --dataset cifar10 \
    --attack pattern \
    --start-epoch 1001 \
    --end-epoch 1005 \
    --debug \
    --debug-fraction 0.1 \
    --num-rounds 10
```

### 6. Dry Run (See Command Without Executing)
```bash
python fera_cli.py \
    --dataset cifar10 \
    --attack pattern \
    --start-epoch 1001 \
    --end-epoch 1101 \
    --dry-run \
    --verbose
```

## Generated Commands

The CLI automatically generates the appropriate `main.py` command. For example:

```bash
python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101
```

Generates:
```bash
python main.py -cn cifar10 aggregator=fera atk_config=cifar10_multishot atk_config.data_poison_method=pattern atk_config.poison_start_round=1001 atk_config.poison_end_round=1101 aggregator_config.fera.spectral_weight=0.6 aggregator_config.fera.delta_weight=0.4 aggregator_config.fera.top_k_percent=0.5 aggregator_config.fera.root_size=64 num_rounds=600 num_clients=100 num_clients_per_round=10 training_mode=parallel num_gpus=1.0 num_cpus=12 save_logging=csv dir_tag=cifar10_fera_data(pattern)_epochs_1001_1101 seed=123 alpha=0.9
```

## Auto-Generated Directory Tags

The CLI automatically generates descriptive directory tags:
- `cifar10_fera_data(pattern)_epochs_1001_1101`
- `emnist_fera_model(neurotoxin)_epochs_1001_1601`

## Validation

The CLI validates:
- Spectral + Delta weights sum to 1.0 (with warning if not)
- Start epoch < End epoch
- Valid layer names for multi-layer mode
- Existence of BackFed directory

## Error Handling

- **KeyboardInterrupt**: Graceful exit with code 130
- **CalledProcessError**: Shows return code and error message
- **ValidationError**: Clear error messages for invalid parameters
- **FileNotFoundError**: Helpful message if BackFed directory not found

## Integration with SLURM

You can use the CLI in SLURM batch scripts:

```bash
#!/bin/bash
#SBATCH --job-name=fera_pattern_test
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

cd /path/to/BackFed
python fera_cli.py \
    --dataset cifar10 \
    --attack pattern \
    --start-epoch 1001 \
    --end-epoch 1101 \
    --num-rounds 100
```

## Tips

1. **Use `--dry-run`** to verify commands before execution
2. **Use `--verbose`** to see the full generated command
3. **Use `--debug`** for quick testing with small data subsets
4. **Customize `--dir-tag`** for better result organization
5. **Adjust `--top-k`** based on expected attack rate (lower = more conservative)

## Help

```bash
python fera_cli.py --help
```

Shows all available options with detailed descriptions and examples.
