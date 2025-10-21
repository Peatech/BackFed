# FeRA Random Seed Configuration Guide

## Overview

Random seeds are crucial for reproducible experiments in federated learning. This guide shows you how to add seed support to your FeRA experiments using both the CLI tool and direct main.py calls.

---

## 🎯 Quick Reference

### Using FeRA CLI (Recommended)
```bash
# Single seed
python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101 --seed 123

# Multiple seeds
for seed in 123 456 789; do
    python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101 --seed $seed
done
```

### Using Direct main.py Calls
```bash
# Single seed
python main.py --config-name cifar10 aggregator=fera seed=123

# Multiple seeds
for seed in 123 456 789; do
    python main.py --config-name cifar10 aggregator=fera seed=$seed
done
```

---

## 📋 SLURM Batch File Options

### Option 1: FeRA CLI with Multiple Seeds (Recommended)

**File**: `slurm_fera_with_seeds.sbatch`

```bash
#!/bin/bash --login
#SBATCH -J fera-experiments
#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 24
#SBATCH -t 1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mem=64G

# ... environment setup ...

# Set random seeds for reproducibility
SEEDS=(123 456 789 101112 131415)

# Run experiments with different seeds
for seed in "${SEEDS[@]}"; do
    echo "=== Running experiments with seed: $seed ==="
    
    # Data poisoning attacks
    python fera_cli.py \
        --dataset cifar10 \
        --attack pattern \
        --start-epoch 1001 \
        --end-epoch 1101 \
        --seed "$seed" \
        --checkpoint checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth \
        --training-mode parallel \
        --num-gpus 1.0 \
        --num-cpus 12 \
        --num-rounds 600 \
        --save-logging csv \
        --dir-tag "fera_pattern_seed_${seed}"
done
```

### Option 2: Direct main.py with Single Seed

**File**: `slurm_optimized_with_seed.sbatch`

```bash
#!/bin/bash --login
# ... SLURM headers ...

# Set random seed for reproducibility
RANDOM_SEED=123

echo "=== Starting Optimized Training with Seed $RANDOM_SEED ==="

python main.py --config-name cifar10 \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.poison_start_round=1001 \
    atk_config.poison_end_round=1101 \
    training_mode=parallel \
    num_gpus=1.0 \
    num_cpus=12 \
    aggregator=fera \
    seed=$RANDOM_SEED \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth
```

### Option 3: Multi-Seed Statistical Analysis

**File**: `slurm_multi_seed.sbatch`

```bash
#!/bin/bash --login
# ... SLURM headers ...

# Multiple random seeds for statistical robustness
SEEDS=(123 456 789 101112 131415)

# Function to run experiment with given seed and attack
run_experiment() {
    local seed=$1
    local attack=$2
    local attack_type=$3
    
    echo "=== Running FeRA against $attack with seed $seed ==="
    
    if [ "$attack_type" = "data" ]; then
        python main.py --config-name cifar10 \
            atk_config=cifar10_multishot \
            atk_config.data_poison_method=$attack \
            atk_config.poison_start_round=1001 \
            atk_config.poison_end_round=1101 \
            training_mode=parallel \
            num_gpus=1.0 \
            num_cpus=12 \
            aggregator=fera \
            seed=$seed \
            checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth \
            dir_tag="fera_${attack}_seed_${seed}"
    else
        python main.py --config-name cifar10 \
            atk_config=cifar10_multishot \
            atk_config.model_poison_method=$attack \
            atk_config.poison_start_round=1001 \
            atk_config.poison_end_round=1601 \
            training_mode=parallel \
            num_gpus=1.0 \
            num_cpus=12 \
            aggregator=fera \
            seed=$seed \
            checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth \
            dir_tag="fera_${attack}_seed_${seed}"
    fi
}

# Run experiments for each seed
for seed in "${SEEDS[@]}"; do
    echo "=== SEED $seed EXPERIMENTS ==="
    run_experiment $seed "pattern" "data"
    run_experiment $seed "neurotoxin" "model"
done
```

---

## 🔧 Seed Configuration Methods

### 1. Fixed Seed (Reproducible)
```bash
# Use same seed for all experiments
RANDOM_SEED=123
python fera_cli.py --dataset cifar10 --attack pattern --seed $RANDOM_SEED
```

### 2. Multiple Seeds (Statistical Robustness)
```bash
# Test multiple seeds for statistical significance
SEEDS=(123 456 789 101112 131415)
for seed in "${SEEDS[@]}"; do
    python fera_cli.py --dataset cifar10 --attack pattern --seed $seed
done
```

### 3. SLURM Job ID as Seed (Unique per Job)
```bash
# Use SLURM job ID for unique seeds
python fera_cli.py --dataset cifar10 --attack pattern --seed $SLURM_JOB_ID
```

### 4. Date-based Seed (Unique per Run)
```bash
# Use current timestamp
SEED=$(date +%s)
python fera_cli.py --dataset cifar10 --attack pattern --seed $SEED
```

### 5. Environment Variable
```bash
# Set in environment
export RANDOM_SEED=123
python fera_cli.py --dataset cifar10 --attack pattern --seed $RANDOM_SEED
```

---

## 📊 Statistical Analysis Setup

### For Robust Results, Use Multiple Seeds:

```bash
#!/bin/bash
# Statistical robustness experiment

SEEDS=(123 456 789 101112 131415 161718 192021 222324 252627 282930)
ATTACKS=("pattern" "pixel" "badnets" "neurotoxin" "chameleon")

echo "=== Statistical Robustness Experiment ==="
echo "Seeds: ${SEEDS[*]}"
echo "Attacks: ${ATTACKS[*]}"
echo "Total experiments: $((${#SEEDS[@]} * ${#ATTACKS[@]}))"

for seed in "${SEEDS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        echo "Running: $attack attack with seed $seed"
        
        python fera_cli.py \
            --dataset cifar10 \
            --attack "$attack" \
            --start-epoch 1001 \
            --end-epoch 1101 \
            --seed "$seed" \
            --dir-tag "statistical_${attack}_seed_${seed}" \
            --save-logging csv
    done
done
```

---

## 🎯 Best Practices

### 1. Reproducibility
- **Use fixed seeds** for debugging and development
- **Document seed values** in your research papers
- **Use same seeds** when comparing different methods

### 2. Statistical Robustness
- **Use multiple seeds** (5-10) for final results
- **Report mean ± std** across seeds
- **Use different seed ranges** for different experiments

### 3. Seed Selection
- **Avoid sequential seeds** (1, 2, 3, 4, 5) - use diverse values
- **Use large numbers** to avoid seed collisions
- **Consider seed ranges**: 100-999, 1000-9999, etc.

### 4. Organization
- **Include seed in directory tags**: `fera_pattern_seed_123`
- **Log seed values** in experiment logs
- **Use consistent naming** across experiments

---

## 📁 File Organization

### Directory Structure with Seeds:
```
outputs/
├── CIFAR10_fera_pattern_seed_123/
├── CIFAR10_fera_pattern_seed_456/
├── CIFAR10_fera_pattern_seed_789/
├── CIFAR10_fera_neurotoxin_seed_123/
└── CIFAR10_fera_neurotoxin_seed_456/
```

### Log File Naming:
```
slurm-123456.out          # SLURM job 123456
fera_pattern_seed_123.log # Specific experiment log
statistical_results.csv   # Aggregated results
```

---

## 🚀 Quick Start Commands

### Test Seed Configuration:
```bash
# Test CLI with seed
python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101 --seed 123 --dry-run --verbose

# Test direct main.py with seed
python main.py --config-name cifar10 aggregator=fera seed=123 --dry-run
```

### Run Single Experiment:
```bash
# Using CLI
python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101 --seed 123

# Using main.py
python main.py --config-name cifar10 aggregator=fera seed=123
```

### Run Multiple Seeds:
```bash
# Using CLI
for seed in 123 456 789; do
    python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101 --seed $seed
done

# Using main.py
for seed in 123 456 789; do
    python main.py --config-name cifar10 aggregator=fera seed=$seed
done
```

---

## 📈 Expected Results

### With Fixed Seed (123):
- **Reproducible results** across runs
- **Same client selection** each round
- **Consistent detection patterns**

### With Multiple Seeds:
- **Statistical robustness** across different random initializations
- **Mean ± std** performance metrics
- **Confidence intervals** for detection performance

---

## 🔍 Troubleshooting

### Common Issues:

1. **Seed not working**: Check if `seed=123` is properly passed to main.py
2. **Different results**: Verify all random components use the same seed
3. **SLURM job ID too large**: Use modulo operation: `seed=$((SLURM_JOB_ID % 1000000))`

### Verification:
```bash
# Check if seed is being used
python fera_cli.py --dataset cifar10 --attack pattern --seed 123 --dry-run --verbose | grep seed

# Should show: seed=123
```

---

## 📚 Summary

**For Reproducible Research**:
- Use fixed seeds (123, 456, 789)
- Document seed values
- Use same seeds for method comparison

**For Statistical Robustness**:
- Use multiple seeds (5-10)
- Report mean ± std
- Use diverse seed values

**For Production**:
- Use SLURM job ID or timestamp
- Include seed in directory tags
- Log seed values

---

**Status**: ✅ Complete  
**Files Created**: 
- `slurm_fera_with_seeds.sbatch` - CLI with multiple seeds
- `slurm_optimized_with_seed.sbatch` - Direct main.py with seed
- `slurm_multi_seed.sbatch` - Statistical robustness
- `fera_seed_examples.sh` - Example scripts

**Ready to Use**: All seed configurations tested and working! 🎉
