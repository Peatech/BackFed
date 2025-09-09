# BackFed Kaggle/Colab Integration

This document explains how to use BackFed in Kaggle and Google Colab environments with single-process execution optimized for cloud notebook environments.

## 🚀 Quick Start

### Kaggle Environment

```python
# In a Kaggle notebook
!git clone https://github.com/your-repo/BackFed.git
%cd BackFed

# Install dependencies
!pip install -r requirements.txt

# Run with default settings (CIFAR10, pattern attack)
!python kaggle_main.py

# Run with custom parameters
!python kaggle_main.py num_rounds=30 num_clients=10
```

### Google Colab Environment  

```python
# In a Colab notebook
!git clone https://github.com/your-repo/BackFed.git
%cd BackFed

# Install dependencies
!pip install -r requirements.txt

# Run experiment
!python kaggle_main.py dataset=CIFAR10 num_rounds=25
```

## 🔧 Key Features

### Automatic Environment Detection
- **Kaggle**: Detects `/kaggle/input` and `/kaggle/working` paths
- **Colab**: Detects Google Colab environment variables
- **Local**: Fallback mode with conservative defaults

### Single-Process Execution
- Forces `training_mode=sequential` (no Ray orchestration)
- Eliminates multi-processing for compatibility
- Runs efficiently on single GPU/CPU

### Conservative Resource Defaults
- **Kaggle/Colab**: 20 clients, 5 per round, 50 total rounds
- **Local**: 50 clients, 10 per round, 100 total rounds
- Smaller batch sizes and reduced epochs for faster execution

### Deterministic Training
- Seeds are set consistently for reproducible results
- CuDNN deterministic mode enabled by default
- Proper handling of random number generators

## 📁 Environment-Specific Paths

### Kaggle Paths
```yaml
outputs_root: "/kaggle/working"
datapath: "/kaggle/input"  # Auto-detects dataset inputs
checkpoints_dir: "/kaggle/working/checkpoints"
csv_results_dir: "/kaggle/working/csv_results"
```

### Colab Paths
```yaml
outputs_root: "/content"
checkpoints_dir: "/content/checkpoints"  
csv_results_dir: "/content/csv_results"
```

## 🎯 Usage Examples

### Basic Federated Learning (No Attack)
```bash
# Clean federated learning
python kaggle_main.py no_attack=True num_rounds=20
```

### Pattern Attack
```bash
# Default configuration includes pattern attack
python kaggle_main.py num_rounds=30 num_clients=15
```

### Different Attacks
```bash
# Pixel attack
python kaggle_main.py atk_config.data_poison_method=pixel

# Blended attack  
python kaggle_main.py atk_config.data_poison_method=blended

# BadNets attack
python kaggle_main.py atk_config.data_poison_method=badnets
```

### Defense Mechanisms
```bash
# Trimmed Mean defense
python kaggle_main.py aggregator=trimmed_mean

# Multi-Krum defense
python kaggle_main.py aggregator=multi_krum

# Coordinate Median defense
python kaggle_main.py aggregator=coordinate_median
```

### Different Datasets
```bash
# MNIST (faster for testing)
python kaggle_main.py dataset=MNIST model=MNISTNet num_classes=10

# CIFAR100 (more challenging)
python kaggle_main.py dataset=CIFAR100 num_classes=100
```

## 🔒 Security & Privacy

### WandB Logging Control
```bash
# Enable WandB logging (requires login)
export ENABLE_WANDB=1
python kaggle_main.py

# Default: CSV logging only (no external connections)
python kaggle_main.py  # Uses CSV by default
```

### Deterministic Training
```bash
# Fully reproducible results (slower)
python kaggle_main.py deterministic=true seed=42

# Faster but less reproducible
python kaggle_main.py deterministic=false
```

## ⚡ Performance Optimization

### Quick Testing
```bash
# Very fast test run
python kaggle_main.py debug=true debug_fraction_data=0.01 num_rounds=3
```

### Memory-Efficient Settings
```bash
# Reduce memory usage
python kaggle_main.py \
  client_config.batch_size=16 \
  test_batch_size=128 \
  num_workers=0 \
  pin_memory=false
```

### GPU Optimization
```bash
# Single GPU setup
python kaggle_main.py \
  cuda_visible_devices="0" \
  num_gpus=1.0

# CPU-only mode
python kaggle_main.py \
  cuda_visible_devices="" \
  num_gpus=0
```

## 🐛 Common Issues & Solutions

### Issue: "Ray not available" Error
```bash
# Solution: Ensure sequential mode is used
python kaggle_main.py training_mode=sequential
```

### Issue: Out of Memory
```bash
# Solution: Reduce batch sizes and clients
python kaggle_main.py \
  client_config.batch_size=16 \
  test_batch_size=64 \
  num_clients=10
```

### Issue: Missing Pretrained Weights
```bash  
# Solution: Train from scratch or disable checkpoints
python kaggle_main.py checkpoint=null pretrain_model_path=null
```

### Issue: Too Slow in Notebook
```bash
# Solution: Use debug mode or reduce parameters
python kaggle_main.py \
  debug=true \
  num_rounds=10 \
  num_clients=8 \
  disable_progress_bar=true
```

## 📊 Expected Outputs

### Files Created
- `{outputs_root}/csv_results/`: CSV logs with metrics
- `{outputs_root}/outputs/`: Hydra run outputs and logs
- `{outputs_root}/checkpoints/`: Model checkpoints (if enabled)

### CSV Results Format
```csv
round,test_acc,test_loss,backdoor_acc,train_time,num_failures
1,0.45,1.2,0.92,15.3,0
2,0.52,1.1,0.89,14.8,0
...
```

## 🧪 Validation Script

To verify your setup works correctly:

```python
# Test environment detection
from kaggle_main import detect_environment
print(f"Environment: {detect_environment()}")

# Test configuration loading
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
initialize(config_path="config")
cfg = compose(config_name="kaggle")
print(f"Sequential mode: {cfg.training_mode}")
print(f"Clients: {cfg.num_clients}")
```

## 📚 Advanced Configuration

### Custom Attack Parameters
```bash
python kaggle_main.py \
  atk_config.poison_rate=0.2 \
  atk_config.target_class=5 \
  atk_config.fraction_adversaries=0.15
```

### Custom Defense Settings
```bash
python kaggle_main.py \
  aggregator=trimmed_mean \
  aggregator_config.trimmed_mean.trim_ratio=0.3
```

### Resource Management
```bash
python kaggle_main.py \
  num_cpus=1 \
  num_gpus=0.5 \
  client_config.timeout=300
```

## 🔄 Migration from main.py

If you have existing commands using `main.py`, they can be easily converted:

### Before (main.py)
```bash
python main.py --cn cifar10 training_mode=parallel num_gpus=0.5
```

### After (kaggle_main.py) 
```bash
python kaggle_main.py --cn kaggle num_gpus=1.0
# Note: training_mode=sequential is enforced automatically
```

## 💡 Best Practices

1. **Start Small**: Use debug mode or small parameters for initial testing
2. **Monitor Resources**: Keep an eye on GPU/RAM usage in notebooks  
3. **Save Results**: CSV results are automatically saved to working directory
4. **Reproducibility**: Use `deterministic=true` and fixed seeds for experiments
5. **Incremental Development**: Test with `no_attack=true` first, then add attacks

## 🆘 Support

For issues specific to Kaggle/Colab integration, check:

1. Environment detection is working correctly
2. Sequential training mode is enforced
3. Paths point to writable directories
4. Dependencies are properly installed
5. GPU/CPU resources are sufficient

The implementation preserves all BackFed features while optimizing for single-process cloud environments.
