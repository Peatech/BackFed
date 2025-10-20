<div align="center">
  <h1> BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning </h1>
  <img src="backfed_logo.png" alt="BackFed project logo" width="50%"/>
  <br><br/>
  
  [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/get-started/locally/) [![Ray 2.10.0](https://img.shields.io/badge/Ray-2.10.0-blue.svg)](https://docs.ray.io/en/latest/installation.html) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2507.04903)
</div>

## About
BackFed is a comprehensive benchmark framework to efficiently and reliably evaluate backdoor attacks in Federated Learning (FL). This benchmark integrates Ray for parallel execution, Hydra for configuration management, and a modular architecture for easy extension of new attacks and defenses. Compared to existing codebases for backdoor attacks in FL, our framework could **achieve 2X - 10X speedup in training time.**

📄 **Paper**: [BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning](https://arxiv.org/abs/2507.04903)

## Updates
+ [05/09/2025]: Fix bugs for Flare and FLTrust defenses. 
+ [06/09/2025]: Add LocalDP client-side defense.
+ [07/09/2025]: Add Anticipate Malicious client.

## Features

- **Modular Architecture**: Each attack and defense belonges to a separate file, allowing for an easy extension with new attacks, defenses, models, and datasets.
- **Parallel Execution**: Support for both sequential and parallel training modes using Ray with timeout mechanism for client training.
- **Comprehensive Attack & Defense Library**: Implementation of various attacks and defenses in a standardized setting for a reliable benchmark.
- **Flexible Configuration**: Hydra-based configuration system for easy and extensible experiment setups.
- **Real-time Logging**: Support real-time loggings via WandB or CSV.
- **Resource Tracking**: Client training and server aggregation are monitored based on memory usage and computation time.

## Supported Datasets and Models

| **Dataset** | **Task** | **Models** | **Data Distribution** |
| :-- | :-- | :-- | :-- |
| CIFAR-10/100 | image classification | ResNet + VGG models | IID (Uniform) + Simulated Non-IID (Dirichlet) |
| EMNIST (ByClass) | handwritten recognition | MnistNet + ResNet models | IID (Uniform) + Dirichlet Non-IID (Dirichlet) |
| Tiny-Imagenet | image classification | Any Pytorch models | IID (Uniform) + Dirichlet Non-IID (Dirichlet) |
| FEMNIST (Federated version of EMNIST) | handwritten recognition | MnistNet + ResNet models | Natural Non-IID (split by writers) |
| Reddit | next-word-prediction | LSTM + Transformer | Natural Non-IID (split by authors) |
| Sentiment140 | sentiment analysis | LSTM + Transformer + ALBERT | Natural Non-IID (split by tweets) |

## Project Structure

```
BackFed/
├── config/                 # Configuration files
│   ├── base.yaml           # Base configuration
│   ├── cifar10.yaml        # CIFAR-10 specific config
│   ├── emnist.yaml         # EMNIST specific config
│   ├── femnist.yaml        # FEMNIST specific config
│   ├── reddit.yaml         # Reddit specific config
│   ├── sentiment140.yaml   # Sentiment140 specific config
│   └── atk_config/         # Attack configurations
├── backfed/                # Core framework
│   ├── clients/            # Client implementations
│   ├── servers/            # Server implementations
│   ├── poisons/            # Poisoning methods
│   ├── models/             # Model architectures
│   ├── datasets/           # Dataset handling
│   ├── utils/              # Utility functions
│   ├── client_manager.py   # Client management
│   ├── context_actor.py    # Ray context management
│   └── fl_dataloader.py    # Federated data loading
├── experiments/            # Example experiment scripts
├── data/                   # Raw datasets
├── data_splits/            # Pre-computed data partitions (for CIFAR10, EMNIST, Tiny-ImageNet)
├── checkpoints/            # Model checkpoints
├── csv_results/            # Experiment results
├── outputs/                # Hydra output logs
├── main.py                 # Main entry point
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites

- Python 3.11
- PyTorch 2.6.0
- Ray 2.10.0 (Most reliable version, later versions could lead to high RAM usage)
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
  ```bash
  git clone https://github.com/thinh-dao/BackFed.git
  cd BackFed
  ```

2. Install dependencies in your environment:
  ```bash
  pip install -r requirements.txt
  ```

3. Download pretrained models (Optional)
  Experiments in ```config``` folder required trained checkpoints, which are provided in [Google Drive](https://drive.google.com/drive/folders/1Pu6ZcDBNfvkXrXxky3Ek6wpWT60_wAdt?usp=drive_link). You can download and put inside checkpoints folder (Note that you must preserve folder structure). Alternatively, you can use the following command to automate the download process:
  ```bash
  chmod +x download_models.sh
  ./download_models.sh $dataset_name # Choose from cifar10, mnist, tinyimagenet, reddit, sentiment140
  ./download_models.sh all  # Download all models (skip if exists)
  ```
  Training settings for these pretrained weights are given in ```experiments/clean_training.sh```.

## Usage
By default, all configuration files in config folder (e.g., ```emnist.yaml```, ```cifar10.yaml```, etc.) will inherit from ```base.yaml``` and override parameters. Please see ```config/base.yaml``` for a full description of parameters.

#### Basic Usage
Choose a configuration file to run experiment:
```bash
python main.py --config-name cifar10  # Equivalent to config/cifar10.yaml
```

For fine-grained control over parameters, you can:

1. **Create custom configuration files** that override key parameters from `base.yaml`
2. **Use attack configuration files** that override parameters from `atk_config/base_attack.yaml`
3. **Override parameters directly** via command line (see [Customizing Experiments](#customizing-experiments-in-command-line))

#### Customizing Experiments in Command Line
Modify configuration parameters using Hydra's override syntax:
```bash
python main.py aggregator=unweighted_fedavg dataset=CIFAR10 model=ResNet18 num_rounds=600
```

To override attack configurations, use the following syntax:
```bash
python main.py atk_config=cifar10_multishot atk_config.model_poison_method=base atk_config.data_poison_method=pattern
```

Note that you can override specific parameters of an attack. For example, set data poison attack to IBA and change ```atk_eps``` of IBA to 0.1:
```bash
python main.py atk_config.data_poison_method=iba atk_config.data_poison_config.iba.atk_eps=0.1
```

Similarly, to override defense configurations, change ```aggregator``` and ```aggregator_config```. For example, to use Trimmed-Mean defense with 20% trimming ratio:
```bash
python main.py aggregator=trimmed_mean aggregator_config.trimmed_mean.trim_ratio=0.2
```

#### Controlling Parallel Training Resources

In `parallel` training mode, multiple Ray actors (spawned processes) are created for concurrent client training. The framework automatically determines the optimal number of parallel actors based on available hardware resources.

**Actor Calculation Formula:**

```
num_parallel_actors = min(
    num_available_gpus / num_gpus_per_client,
    num_available_cpus / num_cpus_per_client
)
```

Where:
- `num_available_gpus`: Number of GPUs specified in `cuda_visible_devices`
- `num_available_cpus`: Total CPU cores available on the machine
- `num_gpus_per_client`: GPU fraction per client (set via `num_gpus` parameter)
- `num_cpus_per_client`: CPU cores per client (set via `num_cpus` parameter, default: 1)

**Resource Configuration Examples**

```bash
# Example 1: 4 GPUs, 0.5 GPU per client = 8 parallel actors
python main.py cuda_visible_devices="0,1,2,3" num_gpus=0.5

# Example 2: 2 GPUs, 1.0 GPU per client = 2 parallel actors
python main.py cuda_visible_devices="0,1" num_gpus=1.0
```

**Performance Tuning**

The most effective way to control parallelism is by adjusting:
1. **`cuda_visible_devices`**: Controls the number of available GPUs
2. **`num_gpus`**: Controls GPU allocation per client

> **⚠️ IMPORTANT:** If the application freezes due to too many parallel Ray actors, try one of these solutions:
> - **Increase** `num_gpus` (allocate more GPU memory per client)
> - **Decrease** the number of GPUs in `cuda_visible_devices`
> - **Increase** `num_cpus` (allocate more CPU cores per client)
> 
> This will reduce the total number of parallel actors and prevent resource contention.

## Configuration

The framework uses Hydra for configuration management. Below are the key configuration parameters organized by category:

### Core Experiment Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregator` | str | `unweighted_fedavg` | Aggregation method for federated learning (see [Available Aggregation Methods](#available-aggregation-methods)) |
| `no_attack` | bool | `False` | Disable attacks (set to `True` for clean training) |
| `training_mode` | str | `parallel` | Training mode (`parallel` for Ray-based or `sequential` for single-threaded) |
| `num_rounds` | int | `600` | Number of federated learning rounds |
| `num_clients` | int | `100` | Total number of clients in the federation |
| `num_clients_per_round` | int | `10` | Number of clients selected per round |
| `federated_evaluation` | bool | `False` | Whether to perform federated evaluation (evaluate global model on client's validation set) |
| `federated_val_split` | float | `0.0` | Validation split for each client's dataset (should be > 0 if federated evaluation is True) |
| `seed` | int | `123456` | Random seed for reproducibility |
| `deterministic` | bool | `False` | Enable deterministic training (slower but fully reproducible) |

Note: If you use debuggers (e.g., ipdb or pdb), it is recommended to set `training_mode=sequential` and `progress_bar=False`.

### Dataset and Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | **Required** | Dataset name (CIFAR10, CIFAR100, MNIST, EMNIST, FEMNIST, TinyImageNet, Reddit, Sentiment140) |
| `model` | str | **Required** | Model architecture (ResNet18, MNISTNet, Simple, Transformer, WordModel) |
| `num_classes` | int | **Required** | Number of classes in the dataset |
| `datapath` | str | `data` | Path to dataset directory |
| `partitioner` | str | `dirichlet` | Data partitioning method (`uniform` or `dirichlet`) |
| `alpha` | float | `0.5` | Dirichlet distribution parameter (lower = more non-IID, higher = more IID) |
| `normalize` | bool | `True` | Whether to normalize the dataset |
| `test_batch_size` | int | `512` | Batch size for testing |
| `test_every` | int | `1` | Test global model every N rounds |

### Client Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client_config.local_epochs` | int | `2` | Number of local training epochs per client |
| `client_config.batch_size` | int | `64` | Training batch size for each client |
| `client_config.lr` | float | `0.1` | Learning rate for client training |
| `client_config.optimizer` | str | `sgd` | Optimizer type (sgd, adam, adamw) |
| `client_config.weight_decay` | float | `5e-4` | Weight decay for regularization |
| `client_config.momentum` | float | `0.9` | Momentum for SGD optimizer |
| `client_config.mixed_precision` | bool | `True` | Enable mixed precision training |
| `client_config.timeout` | float | `Null` | Timeout for client training (seconds) |
| `client_config.val_split` | float | `${federated_val_split}` | Validation split for each client (inherited from federated_val_split) |

### Resource Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda_visible_devices` | str | `"0,1,2,3,4"` | GPU devices to use (comma-separated) |
| `num_cpus` | int | `1` | CPU cores allocated per client |
| `num_gpus` | float | `0.5` | GPU fraction allocated per client |
| `num_workers` | int | `4` | Number of workers for data loading |
| `pin_memory` | bool | `True` | Enable pinned memory for faster GPU transfer |
| `debug` | bool | `False` | Enable debug mode (uses subset of data) |
| `debug_fraction_data` | float | `0.1` | Fraction of data to use in debug mode |

### Model Checkpointing and Resuming

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint` | str/int | `Null` | Resume from checkpoint (round number, file path, or "wandb") |
| `save_model` | bool | `False` | Save model to outputs directory |
| `save_checkpoint` | bool | `False` | Save model to checkpoints directory |
| `save_model_rounds` | list | `[200,400,600,800,1000]` | Specific rounds to save model |
| `pretrain_model_path` | str | `Null` | Path to pretrained weights or "IMAGENET1K_V2" |

### Logging and Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_logging` | str | `csv` | Logging method (`wandb`, `csv`, `both`, or `Null`) |
| `dir_tag` | str | Auto-generated | Directory tag for organizing results |
| `plot_data_distribution` | bool | `False` | Generate data distribution plots |
| `plot_client_selection` | bool | `False` | Generate client selection plots |
| `disable_progress_bar` | bool | `False` | Disable progress bars during training |

## Attacks and Defenses

### Implemented Attacks

| Method | Category | Description | Source |
|--------|----------|-------------|-----------------|
| **Neurotoxin** | Model Poisoning | Selective parameter poisoning targeting specific neurons | [Neurotoxin: Durable backdoors in federated learning (ICML 2022)](https://proceedings.mlr.press/v162/zhang22w/zhang22w.pdf)|
| **Chameleon** | Model Poisoning | Use contrastive learning to adapt the feature extractor better to attacks | [Chameleon: Adapting to peer images for planting durable backdoors in federated learning](https://proceedings.mlr.press/v202/dai23a/dai23a.pdf) |
| **Pattern** | Data Poisoning | White-patch trigger | [How To Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf) |
| **Pixel** | Data Poisoning | One pixel as trigger | [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/pdf/1708.06733) |
| **BadNets** | Data Poisoning | Classic BadNets trigger | [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/pdf/1708.06733) |
| **Blended** | Data Poisoning | Blends a trigger pattern with original images | [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526) |
| **Distributed** | Data Poisoning | Distributed backdoor attack across multiple clients | [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/pdf/61dc789b9f12be96506a23ddb7670ac132a51d6d.pdf) |
| **Edge-case** | Data Poisoning | Use edge-case samples as backdoor triggers | [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://proceedings.neurips.cc/paper/2020/file/b8ffa41d4e492f0fad2f13e29e1762eb-Paper.pdf) |
| **A3FL** | Data Poisoning | Optimize a trigger pattern | [A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/c07d71ff0bc042e4b9acd626a79597fa-Paper-Conference.pdf) |
| **IBA** | Data Poisoning | Use a trigger generator to inject triggers | [IBA: Towards Irreversible Backdoor Attacks in Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/d0c6bc641a56bebee9d985b937307367-Paper-Conference.pdf) |

### Implemented Defenses

| Defense | Category | Description | Source |
|---------|----------|-------------|-----------------|
| **FedProx** | Client-side | Adds proximal term to client optimization | [FedProx](https://arxiv.org/abs/1812.06127) |
| **LocalDP** | Client-side | Adds DP noise at client-side after training | [FedProx](https://arxiv.org/abs/2007.15789) |
| **TrimmedMean** | Robust Aggregation | Removes extreme updates before aggregation | [Byzantine-Robust](https://arxiv.org/abs/1703.02757) |
| **MultiKrum** | Robust Aggregation | Selects subset of updates closest to each other | [Krum](https://arxiv.org/abs/1703.02757) |
| **GeometricMedian** | Robust Aggregation | Uses geometric median for aggregation | [Geometric Median](https://arxiv.org/abs/1803.01498) |
| **CoordinateMedian** | Robust Aggregation | Uses coordinate-wise median aggregation | [Coordinate-wise](https://arxiv.org/abs/1803.01498) |
| **FLTrust** | Robust Aggregation | Trust-based weighted aggregation with server dataset | [FLTrust](https://arxiv.org/abs/2012.13995) |
| **RobustLR** | Robust Aggregation | Adaptive learning rate based on update trustworthiness | [RobustLR](https://arxiv.org/pdf/2007.03767) |
| **FoolsGold** | Anomaly Detection | Detects sybil attacks via update similarity | [FoolsGold](https://arxiv.org/abs/1808.04866) |
| **DeepSight** | Anomaly Detection | Clustering-based backdoor detection | [DeepSight](https://arxiv.org/abs/2201.00763) |
| **RFLBAT** | Anomaly Detection | PCA-based malicious update detection | [RFLBAT](https://arxiv.org/abs/2201.03772) |
| **FLDetector** | Anomaly Detection | Sliding window approach for anomaly detection | [FLDetector](https://arxiv.org/pdf/2207.09209) |
| **FLARE** | Anomaly Detection | MMD-based anomaly detection with trust scores | [FLARE](https://dl.acm.org/doi/10.1145/3488932.3517395) |
| **Indicator** | Anomaly Detection | Statistical anomaly detection method | [Indicator](https://www.usenix.org/system/files/usenixsecurity24-li-songze.pdf) |
| **FLAME** | Anomaly Detection | Clustering, norm clipping, and noise addition | [FLAME](https://arxiv.org/abs/2101.02281) |
| **WeakDP** | Post-aggregation | Clip aggregated model and add noise | [WeakDP](https://arxiv.org/abs/1911.07963) |


### Methods Under Development
- Client-side defenses: FLIP, FL-WBC.
- Robust-aggregation defense: FedReDefense.
- Attacks: 3DFED, F3BA.

## Examples

Check the `experiments/` directory for example scripts:

- `clean_training.sh`: Train models without attacks
- `fedavg_vs_attacks.sh`: Evaluate FedAvg against various attacks
- `anomaly_detection.sh`: Test anomaly detection defenses
- `robust_aggregation_multishot.sh`: Test robust aggregation against multishot attacks
- `model_replacement.sh`: Model replacement attack experiments
- `server_lr.sh`: Experiment with different server learning rates
- `sentiment140.sh`: Sentiment140 dataset experiments
- `weakdp_study.sh`: Differential privacy defense studies

## Acknowledgements
+ Ray Backend follows the implementation of [FL-Bench](https://github.com/KarhouTam/FL-bench)
+ Attacks in NLP [Backdoors101](https://github.com/ebagdasa/backdoors101)
+ [BackdoorIndicator](https://github.com/ybdai7/Backdoor-indicator-defense)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


