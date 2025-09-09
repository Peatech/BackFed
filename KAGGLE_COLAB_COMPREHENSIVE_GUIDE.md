# 🚀 BackFed Complete Kaggle/Colab Usage Guide

Welcome to the comprehensive guide for running BackFed (Federated Learning Backdoor Defense Benchmark) in Kaggle and Google Colab environments! This guide covers everything from basic setup to advanced attack/defense configurations.

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Environment Setup](#-environment-setup)
3. [Basic Usage](#-basic-usage)
4. [Dataset Management](#-dataset-management)
5. [Attack Configurations](#-attack-configurations)
6. [Defense Configurations](#-defense-configurations)
7. [FedAvgCKA Defense (NEW!)](#-fedavgcka-defense-new)
8. [Performance Optimization](#-performance-optimization)
9. [Troubleshooting](#-troubleshooting)
10. [Advanced Examples](#-advanced-examples)
11. [Output Analysis](#-output-analysis)
12. [Best Practices](#-best-practices)

## 🚀 Quick Start

### For Complete Beginners

If you're new to federated learning or BackFed, start here:

**Kaggle Notebook:**
```python
# Cell 1: Clone and setup
!git clone https://github.com/your-repo/BackFed.git
%cd BackFed

# Cell 2: Install dependencies (this may take 3-5 minutes)
!pip install -q -r requirements.txt

# Cell 3: Run your first federated learning experiment (clean, no attacks)
# CIFAR10 will be automatically downloaded to /kaggle/working/data/
!python kaggle_main.py no_attack=true num_rounds=10

# Cell 4: Run with a simple attack
!python kaggle_main.py num_rounds=15 atk_config.data_poison_method=pattern
```

**Google Colab Notebook:**
```python
# Cell 1: Clone and setup
!git clone https://github.com/your-repo/BackFed.git
%cd BackFed

# Cell 2: Install dependencies
!pip install -q -r requirements.txt

# Cell 3: Run basic experiment
!python kaggle_main.py num_rounds=10

# Cell 4: Run with defense
!python kaggle_main.py aggregator=fedavgcka num_rounds=20
```

## 🛠 Environment Setup

### Automatic Environment Detection

BackFed automatically detects your environment:
- **Kaggle**: Detects `/kaggle/input` and `/kaggle/working` directories
- **Colab**: Detects Google Colab environment variables
- **Local**: Falls back to local machine settings

### Manual Environment Configuration

```python
# Force specific environment behavior
!python kaggle_main.py outputs_root="/content" num_rounds=20  # Force Colab paths
!python kaggle_main.py outputs_root="/kaggle/working"         # Force Kaggle paths
```

### Dependency Management

**Install all dependencies:**
```bash
!pip install -r requirements.txt
```

**Install minimal dependencies only:**
```bash
!pip install torch torchvision hydra-core omegaconf wandb rich scikit-learn
```

**For memory-constrained environments:**
```bash
!pip install torch torchvision --no-deps
!pip install hydra-core omegaconf rich
```

## 📚 Basic Usage

### Default Configuration

BackFed uses conservative defaults for Kaggle/Colab:
- **20 clients** (vs. 100 locally)
- **5 clients per round** (vs. 10 locally)
- **50 total rounds** (vs. 600 locally)
- **Sequential mode** (no Ray parallelization)
- **Deterministic training** (reproducible results)

### Configuration Override Examples

```bash
# Change number of clients
!python kaggle_main.py num_clients=30 num_clients_per_round=8

# Use different dataset
!python kaggle_main.py dataset=MNIST model=MNISTNet num_classes=10

# Adjust training intensity
!python kaggle_main.py client_config.local_epochs=2 client_config.batch_size=16

# Quick test run
!python kaggle_main.py debug=true num_rounds=3
```

### CLI Override Syntax

BackFed uses [Hydra](https://hydra.cc/) configuration. Key patterns:

```bash
# Simple parameter override
!python kaggle_main.py param_name=value

# Nested parameter override
!python kaggle_main.py client_config.batch_size=32

# List parameter override
!python kaggle_main.py save_model_rounds=[10,20,30]

# Dictionary parameter override  
!python kaggle_main.py multi_layer_weights.avgpool=0.5
```

## 💾 Dataset Management

### Supported Datasets

| Dataset | Model | Classes | Kaggle Dataset | Notes |
|---------|-------|---------|----------------|-------|
| CIFAR10 | ResNet18 | 10 | `cifar-10-python` | Default, most tested |
| CIFAR100 | ResNet18 | 100 | `cifar-100-python` | More challenging |
| MNIST | MNISTNet | 10 | `mnist-in-csv` | Fastest training |
| EMNIST | MNISTNet | 47 | Custom upload needed | Letters + digits |
| TinyImageNet | ResNet18 | 200 | `tiny-imagenet-200` | Large, slow |

### Using Kaggle Datasets

**Method 1: Add Dataset as Input**
1. Go to your Kaggle notebook
2. Click "Add Data" → "Search" → Find dataset (e.g., "cifar-10-python")
3. Add to notebook
4. Update data path:

```python
!python kaggle_main.py datapath="/kaggle/input/cifar-10-python"
```

**Method 2: Auto-download (if allowed)**
```python
# BackFed will automatically download common datasets to /kaggle/working/data
!python kaggle_main.py dataset=CIFAR10  # Downloads automatically
```

**Method 3: Upload Custom Dataset**
```python
# Upload your dataset files to Kaggle as a dataset, then:
!python kaggle_main.py datapath="/kaggle/input/your-custom-dataset"
```

### Dataset-Specific Configurations

**CIFAR10 (Recommended for beginners):**
```bash
!python kaggle_main.py dataset=CIFAR10 model=ResNet18 num_classes=10
```

**MNIST (Fastest training):**
```bash
!python kaggle_main.py dataset=MNIST model=MNISTNet num_classes=10
```

**CIFAR100 (More challenging):**
```bash
!python kaggle_main.py dataset=CIFAR100 model=ResNet18 num_classes=100
```

## ⚔️ Attack Configurations

### Data Poisoning Attacks

BackFed supports various backdoor attacks. Here's how to use each:

#### 1. Pattern Attack (Default, Simple)
```bash
# Basic pattern attack
!python kaggle_main.py atk_config.data_poison_method=pattern

# Custom pattern attack
!python kaggle_main.py \
  atk_config.data_poison_method=pattern \
  atk_config.poison_rate=0.2 \
  atk_config.target_class=5
```

#### 2. Pixel Attack (Single pixel trigger)
```bash
!python kaggle_main.py atk_config.data_poison_method=pixel
```

#### 3. BadNets Attack (Classic backdoor)
```bash
!python kaggle_main.py atk_config.data_poison_method=badnets
```

#### 4. Blended Attack (Transparent overlay)
```bash
!python kaggle_main.py atk_config.data_poison_method=blended
```

#### 5. IBA Attack (Invisible backdoor)
```bash
# Basic IBA
!python kaggle_main.py atk_config.data_poison_method=iba

# Custom IBA with specific parameters
!python kaggle_main.py \
  atk_config.data_poison_method=iba \
  atk_config.data_poison_config.iba.atk_eps=0.2
```

#### 6. A3FL Attack (Adaptive attack)
```bash
!python kaggle_main.py atk_config.data_poison_method=a3fl
```

### Model Poisoning Attacks

#### 1. Neurotoxin (Advanced model poisoning)
```bash
!python kaggle_main.py atk_config.model_poison_method=neurotoxin
```

#### 2. Chameleon (Contrastive learning based)
```bash
!python kaggle_main.py atk_config.model_poison_method=chameleon
```

### Attack Customization

**Target Class Selection:**
```bash
# Attack specific target class
!python kaggle_main.py atk_config.target_class=7

# Random target class (different each run)
!python kaggle_main.py atk_config.random_class=true
```

**Attack Intensity:**
```bash
# High intensity attack
!python kaggle_main.py \
  atk_config.poison_rate=0.4 \
  atk_config.fraction_adversaries=0.2

# Low intensity attack  
!python kaggle_main.py \
  atk_config.poison_rate=0.1 \
  atk_config.fraction_adversaries=0.05
```

**Attack Types:**
```bash
# All-to-one attack (all classes → one target)
!python kaggle_main.py atk_config.attack_type="all2one"

# One-to-one attack (one source → one target)
!python kaggle_main.py atk_config.attack_type="one2one" atk_config.source_class=3
```

### Attack Timing

**Single-shot attacks (recommended for Kaggle/Colab):**
```bash
!python kaggle_main.py \
  atk_config.poison_frequency="single-shot" \
  atk_config.poison_start_round=10
```

**Multi-shot attacks:**
```bash
!python kaggle_main.py \
  atk_config.poison_frequency="multi-shot" \
  atk_config.poison_start_round=5 \
  atk_config.poison_end_round=15
```

## 🛡️ Defense Configurations

### Robust Aggregation Defenses

#### 1. FedAvg (Baseline, no defense)
```bash
!python kaggle_main.py aggregator=unweighted_fedavg
```

#### 2. Trimmed Mean (Remove extreme updates)
```bash
# Default trimming
!python kaggle_main.py aggregator=trimmed_mean

# Custom trimming ratio
!python kaggle_main.py \
  aggregator=trimmed_mean \
  aggregator_config.trimmed_mean.trim_ratio=0.3
```

#### 3. Multi-Krum (Select similar updates)
```bash
!python kaggle_main.py aggregator=multi_krum
```

#### 4. Coordinate/Geometric Median (Robust statistics)
```bash
# Coordinate median
!python kaggle_main.py aggregator=coordinate_median

# Geometric median  
!python kaggle_main.py aggregator=geometric_median
```

#### 5. FLTrust (Server-side validation)
```bash
!python kaggle_main.py aggregator=fltrust
```

### Anomaly Detection Defenses

#### 1. FLAME (Clustering + noise)
```bash
!python kaggle_main.py aggregator=flame
```

#### 2. FoolsGold (Sybil attack detection)
```bash
!python kaggle_main.py aggregator=foolsgold
```

#### 3. RFLBAT (PCA-based detection)
```bash
!python kaggle_main.py aggregator=rflbat
```

#### 4. Indicator (Statistical detection)
```bash
!python kaggle_main.py aggregator=indicator
```

### Client-side Defenses

#### 1. Differential Privacy (Noise addition)
```bash
!python kaggle_main.py aggregator=weakdp
```

#### 2. FedProx (Proximal term)
```bash
!python kaggle_main.py aggregator=fedprox
```

## 🆕 FedAvgCKA Defense (NEW!)

FedAvgCKA is our newest defense that uses Centered Kernel Alignment (CKA) to measure similarity between client model activations and filter out suspicious clients.

### Basic Usage

```bash
# Use FedAvgCKA with default settings
!python kaggle_main.py aggregator=fedavgcka

# FedAvgCKA with pattern attack
!python kaggle_main.py \
  aggregator=fedavgcka \
  atk_config.data_poison_method=pattern
```

### FedAvgCKA Configuration Options

#### Layer Comparison Options

**1. Penultimate Layer (Recommended, fastest):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="penultimate"
```

**2. Specific Layers:**
```bash
# Use layer2 for comparison  
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="layer2"

# Use layer3 for comparison
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="layer3"
```

**3. Multi-layer Comparison (More accurate, slower):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="multi_layer" \
  aggregator_config.fedavgcka.multi_layer_weights.avgpool=0.5 \
  aggregator_config.fedavgcka.multi_layer_weights.layer3=0.3 \
  aggregator_config.fedavgcka.multi_layer_weights.layer2=0.2
```

#### Filtering Intensity

**Conservative filtering (keep more clients):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.trim_fraction=0.1
```

**Aggressive filtering (remove more clients):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.trim_fraction=0.4
```

#### Reference Dataset Configuration

**Small reference dataset (faster, less memory):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.root_dataset_size=16
```

**Large reference dataset (more accurate):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.root_dataset_size=128
```

**Sampling strategy:**
```bash
# Class-balanced sampling (recommended)
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.root_sampling_strategy="class_balanced"

# Random sampling  
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.root_sampling_strategy="random"
```

#### Debugging FedAvgCKA

**Enable detailed logging:**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.log_scores=true
```

### FedAvgCKA Performance Examples

**Memory-efficient setup:**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="penultimate" \
  aggregator_config.fedavgcka.root_dataset_size=32 \
  aggregator_config.fedavgcka.trim_fraction=0.2 \
  num_clients=15 \
  client_config.batch_size=16
```

**High-accuracy setup (if you have enough resources):**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="multi_layer" \
  aggregator_config.fedavgcka.root_dataset_size=64 \
  aggregator_config.fedavgcka.trim_fraction=0.3 \
  num_rounds=30
```

## 🚀 Performance Optimization

### For Limited Memory (Kaggle/Colab Free Tier)

```bash
!python kaggle_main.py \
  num_clients=10 \
  num_clients_per_round=3 \
  client_config.batch_size=16 \
  test_batch_size=64 \
  num_workers=0 \
  pin_memory=false
```

### For Speed (Quick Testing)

```bash
!python kaggle_main.py \
  debug=true \
  debug_fraction_data=0.05 \
  num_rounds=5 \
  num_clients=8 \
  disable_progress_bar=true
```

### For GPU Optimization

```bash
# Single GPU setup
!python kaggle_main.py \
  cuda_visible_devices="0" \
  num_gpus=1.0 \
  client_config.mixed_precision=true

# CPU-only (if GPU is unavailable)
!python kaggle_main.py \
  cuda_visible_devices="" \
  num_gpus=0
```

### Memory Usage Monitoring

```python
# Cell to monitor memory usage during training
import psutil
import GPUtil

def print_memory_usage():
    # CPU Memory
    cpu_memory = psutil.virtual_memory()
    print(f"CPU Memory: {cpu_memory.percent}% used")
    
    # GPU Memory (if available)
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% memory used")
    except:
        print("No GPU available")

# Run this in a separate cell during training
print_memory_usage()
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Out of Memory" Error

**Solution A: Reduce batch sizes**
```bash
!python kaggle_main.py \
  client_config.batch_size=8 \
  test_batch_size=32
```

**Solution B: Reduce number of clients**
```bash
!python kaggle_main.py \
  num_clients=8 \
  num_clients_per_round=2
```

**Solution C: Use debug mode**
```bash
!python kaggle_main.py debug=true
```

#### 2. "Ray not available" Error

This should not happen with `kaggle_main.py`, but if it does:
```bash
!python kaggle_main.py training_mode=sequential
```

#### 3. "Layer not found" Error (FedAvgCKA)

**Find available layers:**
```python
# Run this to see what layers are available in your model
import torch
from backfed.utils import get_model

model = get_model("ResNet18", 10)  # (model_name, num_classes)
for name, module in model.named_modules():
    print(name)
```

**Then use a valid layer:**
```bash
!python kaggle_main.py \
  aggregator=fedavgcka \
  aggregator_config.fedavgcka.layer_comparison="layer4"  # Use valid layer name
```

#### 4. "Dataset not found" or "Read-only file system" Error

BackFed automatically handles dataset downloads in Kaggle/Colab environments. If you see a read-only file system error, this means:

**For CIFAR10/MNIST (auto-download datasets):**
- BackFed will automatically download to `/kaggle/working/data/` if not found in `/kaggle/input/`
- No action needed - just re-run your command

**For custom datasets:**
```bash
# Option A: Upload as Kaggle dataset and specify path
!python kaggle_main.py datapath="/kaggle/input/your-dataset-name"

# Option B: Use working directory
!python kaggle_main.py datapath="/kaggle/working/data"
```

**Troubleshooting:**
If downloads fail, you can manually specify the working directory:
```bash
!python kaggle_main.py datapath="/kaggle/working" dataset=CIFAR10
```

#### 5. "Checkpoint not found" Error

**Disable checkpoints:**
```bash
!python kaggle_main.py \
  checkpoint=null \
  pretrain_model_path=null
```

### Performance Issues

#### Slow Training

```bash
# Use smaller models or datasets
!python kaggle_main.py \
  dataset=MNIST \
  model=MNISTNet \
  num_clients=10

# Reduce data processing
!python kaggle_main.py \
  num_workers=0 \
  pin_memory=false
```

#### High Memory Usage

```bash
# Conservative memory settings
!python kaggle_main.py \
  client_config.batch_size=8 \
  test_batch_size=32 \
  aggregator_config.fedavgcka.root_dataset_size=16 \
  num_workers=0
```

## 🎯 Advanced Examples

### Comprehensive Attack vs Defense Comparison

```bash
# Test multiple defenses against pattern attack
for defense in unweighted_fedavg trimmed_mean multi_krum fedavgcka; do
  echo "Testing $defense"
  python kaggle_main.py \
    aggregator=$defense \
    atk_config.data_poison_method=pattern \
    num_rounds=20 \
    name_tag="defense_${defense}"
done
```

### Multi-Attack Study

```bash
# Test FedAvgCKA against different attacks
for attack in pattern pixel badnets blended; do
  echo "Testing FedAvgCKA vs $attack"
  python kaggle_main.py \
    aggregator=fedavgcka \
    atk_config.data_poison_method=$attack \
    num_rounds=25 \
    name_tag="attack_${attack}"
done
```

### Hyperparameter Sweeps

**FedAvgCKA Trim Fraction Study:**
```bash
for trim in 0.1 0.2 0.3 0.4; do
  python kaggle_main.py \
    aggregator=fedavgcka \
    aggregator_config.fedavgcka.trim_fraction=$trim \
    num_rounds=20 \
    name_tag="trim_${trim}"
done
```

**Attack Intensity Study:**
```bash
for rate in 0.1 0.2 0.3; do
  python kaggle_main.py \
    atk_config.poison_rate=$rate \
    num_rounds=20 \
    name_tag="poison_rate_${rate}"
done
```

## 📊 Output Analysis

### Understanding Results

BackFed saves results in CSV format to `/kaggle/working/csv_results/` (Kaggle) or `/content/csv_results/` (Colab).

**Key Metrics:**
- `test_acc`: Main task accuracy (should stay high)
- `backdoor_acc`: Attack success rate (should be low with good defense)
- `train_time`: Time per round
- `num_failures`: Failed client training attempts

### Analyzing CSV Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('/kaggle/working/csv_results/experiment_results.csv')

# Plot accuracy over rounds
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(results['round'], results['test_acc'], label='Main Task Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Main Task Performance')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results['round'], results['backdoor_acc'], label='Backdoor Accuracy', color='red')
plt.xlabel('Round')
plt.ylabel('Attack Success Rate')
plt.title('Attack Performance')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("Final Results:")
print(f"Main Task Accuracy: {results['test_acc'].iloc[-1]:.3f}")
print(f"Backdoor Attack Success: {results['backdoor_acc'].iloc[-1]:.3f}")
print(f"Average Training Time: {results['train_time'].mean():.2f}s")
```

### Comparing Multiple Experiments

```python
import glob

# Load all result files
result_files = glob.glob('/kaggle/working/csv_results/*.csv')
experiments = {}

for file in result_files:
    name = file.split('/')[-1].replace('.csv', '')
    experiments[name] = pd.read_csv(file)

# Compare final accuracies
comparison = pd.DataFrame({
    'Experiment': experiments.keys(),
    'Main_Accuracy': [df['test_acc'].iloc[-1] for df in experiments.values()],
    'Backdoor_Success': [df['backdoor_acc'].iloc[-1] for df in experiments.values()]
})

print(comparison)
```

## 📝 Best Practices

### For Beginners

1. **Start Simple**: Use `no_attack=true` to understand federated learning first
2. **Use MNIST**: Fastest dataset for initial experiments
3. **Enable Logging**: Set `aggregator_config.fedavgcka.log_scores=true` for debugging
4. **Small Scale**: Start with `num_clients=10, num_rounds=15`

### For Research

1. **Reproducibility**: Always set `deterministic=true` and fixed `seed`
2. **Multiple Runs**: Run each experiment 3-5 times with different seeds
3. **Control Variables**: Change one parameter at a time
4. **Document**: Use `name_tag` to track different experimental conditions

### For Resource Management

1. **Monitor Memory**: Use the memory monitoring code provided
2. **Conservative Settings**: Start with small parameters and increase gradually  
3. **Clean Up**: Delete result files between major experiments
4. **Batch Processing**: Use loops to run multiple experiments automatically

### For Defense Evaluation

1. **Baseline First**: Always run `aggregator=unweighted_fedavg` as baseline
2. **Multiple Attacks**: Test defenses against various attack types
3. **Attack-Free**: Test with `no_attack=true` to ensure defense doesn't hurt performance
4. **Parameter Sensitivity**: Test different defense parameters

### Security Considerations

1. **No External Data**: BackFed works entirely within Kaggle/Colab - no external downloads
2. **Deterministic**: Results are reproducible with same seeds
3. **Resource Limits**: Designed to respect Kaggle/Colab resource constraints
4. **Clean Logging**: No sensitive information in logs

## 📞 Getting Help

### If Something Doesn't Work

1. **Check the Error Message**: Most errors have clear explanations
2. **Try Conservative Settings**: Reduce `num_clients`, `batch_size`, etc.
3. **Enable Debug Mode**: Add `debug=true` to your command
4. **Check Available Resources**: Use memory monitoring code
5. **Simplify**: Start with `no_attack=true` and basic settings

### Example Debugging Session

```bash
# If you get an error, try this sequence:

# 1. Minimal working setup
!python kaggle_main.py no_attack=true num_rounds=3 num_clients=4 debug=true

# 2. Add attack
!python kaggle_main.py atk_config.data_poison_method=pattern num_rounds=5 debug=true  

# 3. Add defense
!python kaggle_main.py aggregator=fedavgcka debug=true num_rounds=5

# 4. Scale up gradually
!python kaggle_main.py aggregator=fedavgcka num_rounds=15 num_clients=10
```

### Community and Support

- **GitHub Issues**: Report bugs and ask questions
- **Documentation**: Check the main README for algorithm details
- **Examples**: Look at `experiments/` folder for script examples
- **Configuration**: Study `config/` files for parameter options

---

🎉 **You're now ready to run sophisticated federated learning experiments with BackFed!** 

Start with the Quick Start section, then explore attacks and defenses that interest you. The FedAvgCKA defense is particularly powerful against backdoor attacks - give it a try!

Remember: Start simple, monitor resources, and scale up gradually. Happy experimenting! 🚀
