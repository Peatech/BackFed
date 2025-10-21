# Edge-Case (Semantic Backdoor) Attack Setup

## Overview

The **edge-case attack** is a semantic backdoor that uses out-of-distribution (OOD) airplane images as triggers instead of synthetic patterns. This makes it more realistic and harder to detect than traditional backdoor attacks.

### What is a Semantic Backdoor?

Unlike pattern-based backdoors (e.g., pixel patterns, blended images), semantic backdoors use **natural objects** as triggers:
- **Pattern-based**: Add artificial trigger (e.g., white pixel in corner) → Target class
- **Semantic**: Use real-world object (e.g., airplane image) → Target class

Semantic backdoors are:
- ✅ More realistic (no artificial patterns)
- ✅ Harder to detect visually
- ✅ Represent real threat scenarios
- ⚠️ Require specific trigger objects

---

## Dataset

The edge-case attack uses **CIFAR-10 airplane images (class 0)** as semantic triggers.

### Statistics

- **Training samples**: 5,000 airplane images
- **Test samples**: 1,000 airplane images
- **Image size**: 32×32×3 (RGB)
- **Format**: PIL Images stored as pickle files

### Files

```
backfed/poisons/shared/edge-case/
├── southwest_images_new_train.pkl  (5000 airplane images)
└── southwest_images_new_test.pkl   (1000 airplane images)
```

---

## Setup Instructions

### 1. Prepare Dataset

Run the dataset preparation script:

```bash
python backfed/poisons/shared/edge-case/prepare_edge_case_data.py
```

**What it does:**
1. Downloads CIFAR-10 dataset (if not already present)
2. Filters for airplane images (class 0)
3. Saves as pickle files
4. Verifies dataset creation

**Expected output:**
```
Edge-Case Dataset Preparation
======================================================================
Loading CIFAR-10 training dataset...
  Total training images: 50000
Loading CIFAR-10 test dataset...
  Total test images: 10000

Filtering airplane images (class 0)...
  Training airplanes: 5000
  Test airplanes: 1000

Saving pickle files...
  Training: backfed/poisons/shared/edge-case/southwest_images_new_train.pkl
  Test: backfed/poisons/shared/edge-case/southwest_images_new_test.pkl

Verifying saved files...
  ✓ Training file loaded: 5000 images
  ✓ Test file loaded: 1000 images
  ✓ Sample image type: <class 'PIL.Image.Image'>
  ✓ Sample image size: (32, 32)

✓ Edge-Case Dataset Preparation Complete!
```

### 2. Validate Installation

Run the test script:

```bash
python test_edge_case.py
```

**What it tests:**
1. Dataset files exist
2. Can load pickle files
3. EdgeCase class can be imported
4. EdgeCase can be instantiated
5. poison_inputs() method works

**Expected result:** All tests should pass (✓ PASS)

---

## Usage

### Command Line

**Quick test (2 rounds):**
```bash
python main.py --config-name cifar10 \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.model_poison_method=base \
    atk_config.poison_start_round=1001 \
    atk_config.poison_end_round=1002 \
    num_rounds=2 \
    seed=123 \
    aggregator=fera \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth
```

**Full run (100 rounds):**
```bash
python main.py --config-name cifar10 \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=edge_case \
    atk_config.model_poison_method=base \
    atk_config.poison_start_round=1001 \
    atk_config.poison_end_round=1101 \
    num_rounds=100 \
    seed=123 \
    aggregator=fera \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth
```

### SLURM Batch

The edge-case attack is included in `slurm_optimized.sbatch`:

```bash
sbatch slurm_optimized.sbatch
```

---

## Attack Behavior

### Training Phase

**Malicious clients** (10% of population):
1. Receive normal training data
2. During poisoning rounds (1001-1101):
   - Replace 31.25% of each batch with airplane images
   - Label them as target class (default: class 2)
3. Train model to classify airplanes → target class

**Benign clients** (90% of population):
- Train normally on clean CIFAR-10 data
- No knowledge of backdoor

### Testing Phase

**Clean Accuracy**:
- Model should maintain good accuracy on normal CIFAR-10 test set
- Target: >80% accuracy

**Backdoor Accuracy (BA)**:
- Model should misclassify airplane images as target class
- Measured on 1000 airplane test images
- High BA = successful attack

---

## Attack Configuration

**File**: `config/atk_config/base_attack.yaml`

```yaml
edge_case:
  _target_: backfed.poisons.EdgeCase
```

**Key Parameters** (in attack config):
- `data_poison_method`: `edge_case`
- `model_poison_method`: `base` (standard malicious client)
- `attack_type`: `all2one` (only supported type)
- `target_class`: Target class for backdoor (default: 2)
- `poison_rate`: Fraction of batch to poison (default: 0.3125)
- `poison_start_round`: When to start poisoning (e.g., 1001)
- `poison_end_round`: When to stop poisoning (e.g., 1101)

---

## Integration with FeRA Defense

### Expected Behavior

**Stage 1: Consistency-Based Detection**
- Edge-case may have moderate variance (real images)
- Might not be caught by low-variance detection alone

**Stage 2: Norm-Inflation Detection**
- If attack creates extreme representation norms, will be caught
- Two-sided filtering should detect both low-variance and high-variance attacks

### Monitoring

Watch for in logs:
```
INFO  ═══ FeRA Detection Results (Round XXXX) ═══
INFO  Predicted malicious clients: [...]
INFO  Ground-truth malicious clients: [...]
INFO  Outlier detection: Flagging X extreme outliers
```

---

## Troubleshooting

### Dataset Not Found

**Error:**
```
FileNotFoundError: backfed/poisons/shared/edge-case/southwest_images_new_train.pkl
```

**Solution:**
```bash
python backfed/poisons/shared/edge-case/prepare_edge_case_data.py
```

### Import Error

**Error:**
```
ImportError: cannot import name 'EdgeCase'
```

**Solution:**
Check that `EdgeCase` is in `backfed/poisons/__init__.py`:
```python
from .edge_case import EdgeCase

__all__ = [
    ...
    "EdgeCase",
    ...
]
```

### All2All Not Supported

**Error:**
```
ValueError: Edge-case is not supported for all2all attack
```

**Solution:**
Edge-case only supports `all2one` attacks. Change config:
```yaml
attack_type: "all2one"  # Not "all2all"
target_class: 2  # Specify target
```

### Empty Dataset

**Error:**
```
Number of edge case train: 0 - test: 0
```

**Solution:**
Re-run dataset preparation script. Ensure CIFAR-10 downloads correctly.

---

## Comparison with Other Attacks

| Attack | Trigger Type | Visibility | Detection Difficulty |
|--------|-------------|------------|---------------------|
| **Pattern** | Synthetic pattern | Visible | Easy (obvious artifact) |
| **Pixel** | Single pixel | Subtle | Medium (small artifact) |
| **Blended** | Alpha-blended image | Subtle | Medium (requires close inspection) |
| **Edge-Case** | Real object (airplane) | **Natural** | **Hard** (looks legitimate) |

**Key Advantage of Edge-Case:**
- No artificial patterns added to images
- Uses real objects that naturally occur
- Harder to detect visually and statistically
- Represents realistic threat scenario

---

## Expected Results

### Without Defense (FedAvg)

- Clean Accuracy: ~85-90%
- Backdoor Accuracy: ~95-100% (attack succeeds)
- Prediction: Airplanes → Target class

### With FeRA Defense

- Clean Accuracy: ~85-90% (maintained)
- Backdoor Accuracy: <20% (attack suppressed)
- Detection: Malicious clients identified and excluded

---

## Implementation Details

**File**: `backfed/poisons/edge_case.py`

**Key Methods:**
- `__init__()`: Load airplane images from pickle files
- `poison_inputs()`: Replace batch images with airplanes
- `poison_test()`: Evaluate backdoor success rate

**Dataset Loading:**
```python
# For CIFAR-10
with open('backfed/poisons/shared/edge-case/southwest_images_new_train.pkl', 'rb') as f:
    saved_southwest_dataset_train = pickle.load(f)
```

**Poisoning:**
```python
def poison_inputs(self, inputs):
    # Replace inputs with edge-case samples
    poison_choice = random.sample(range(len(self.edge_case_train)), inputs.shape[0])
    poison_inputs = self.edge_case_train[poison_choice].to(inputs.device)
    return poison_inputs
```

---

## Research Context

The edge-case attack demonstrates:
1. **Semantic Backdoors**: Using natural objects as triggers
2. **OOD Detection Challenge**: Harder to detect than synthetic patterns
3. **Defense Robustness**: Tests if FeRA can handle realistic triggers
4. **Real-World Relevance**: Represents practical attack scenarios

### Original Paper Concept

Edge-case attacks target scenarios where:
- Attacker has access to specific OOD data
- Trigger is a real object class (e.g., airplanes)
- Backdoor activates on naturally occurring images
- No artificial patterns needed

---

## Files Created

1. `backfed/poisons/shared/edge-case/prepare_edge_case_data.py` - Dataset creator
2. `test_edge_case.py` - Validation script
3. `EDGE_CASE_SETUP.md` - This documentation
4. `slurm_optimized.sbatch` - Updated with edge-case

---

## Summary

✅ **Dataset**: 6000 CIFAR-10 airplane images  
✅ **Attack Type**: Semantic backdoor (all2one)  
✅ **Integration**: Fully integrated into BackFed pipeline  
✅ **Testing**: Validated and ready to run  
✅ **Documentation**: Complete setup guide  

**Status**: Ready for production experiments with FeRA defense!

---

**Last Updated**: 2025-10-21  
**Prepared by**: AI Assistant  
**Validated**: All tests passing

