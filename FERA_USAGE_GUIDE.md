# FeRA Defense Usage Guide

## Overview

**FeRA (Feature Representation Anomaly)** is a defense mechanism that detects malicious clients in federated learning by analyzing their feature representations using:

1. **Spectral Norm**: Measures concentration of change in representations (largest eigenvalue of delta covariance matrix)
2. **Delta Norm**: Measures total deviation from global representations (Frobenius norm)

## Key Features

- ✅ Dual-signal anomaly detection (spectral + delta norms)
- ✅ Robust normalization using median + IQR
- ✅ Configurable top-K% threshold-based detection
- ✅ Multi-layer feature extraction support
- ✅ Graceful error handling
- ✅ Detailed detection logging

## Installation

FeRA is already integrated into BackFed. No additional installation required.

## Basic Usage

### 1. Single-Layer Detection (Default)

```bash
python main.py -cn cifar10 \
    aggregator=fera \
    atk_config=cifar10_multishot \
    num_rounds=600 \
    training_mode=parallel \
    num_gpus=0.5 \
    save_logging=csv
```

### 2. Multi-Layer Detection

```bash
python main.py -cn cifar10 \
    aggregator=fera \
    aggregator_config.fera.use_multi_layer=true \
    "aggregator_config.fera.layers=['layer2','layer3','penultimate']" \
    aggregator_config.fera.combine_layers_method=mean \
    atk_config=cifar10_multishot \
    num_rounds=600 \
    training_mode=parallel \
    num_gpus=0.5
```

### 3. Custom Parameters

```bash
python main.py -cn cifar10 \
    aggregator=fera \
    aggregator_config.fera.spectral_weight=0.7 \
    aggregator_config.fera.delta_weight=0.3 \
    aggregator_config.fera.top_k_percent=0.3 \
    aggregator_config.fera.root_size=128 \
    atk_config=cifar10_multishot \
    num_rounds=600
```

**Note**: Use `-cn` (or `--config-name`) to specify the dataset config file (cifar10, emnist, etc.)

## Configuration Parameters

### Core Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta` | float | 0.5 | Learning rate for aggregation |
| `spectral_weight` | float | 0.6 | Weight for spectral norm signal |
| `delta_weight` | float | 0.4 | Weight for delta norm signal |
| `top_k_percent` | float | 0.5 | Percentage of clients to flag as malicious |

### Multi-Layer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_multi_layer` | bool | false | Enable multi-layer analysis |
| `layers` | list | ['penultimate'] | Layers to extract features from |
| `combine_layers_method` | str | 'mean' | How to combine layer scores ('mean', 'max', 'vote') |

**Available Layer Options:**
- `'penultimate'`: Second-to-last layer (default)
- `'layer2'`: ResNet layer2
- `'layer3'`: ResNet layer3
- `'layer4'`: ResNet layer4
- Custom layer names (if accessible as model attributes)

### Root Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_size` | int | 64 | Number of samples for feature extraction |
| `use_ood_root_dataset` | bool | false | Use out-of-distribution data |

## Detection Output

FeRA provides detailed detection results sorted by anomalousness (ascending order):

```
═══ FeRA Detection Results (Round 10) ═══
Total clients: 10
Flagged as malicious: 5
Flagged as benign: 5

Client scores (sorted by anomalousness, ascending):
Client   Spectral   Delta      Norm_Spec    Norm_Delta   Combined   Status    
--------------------------------------------------------------------------------
1        0.1234     45.6789    0.0234       0.1234       0.0567     benign    
2        0.2345     56.7890    0.1234       0.2345       0.1678     benign    
...
9        2.3456     234.5678   0.8765       0.9012       0.8889     MALICIOUS 
10       2.4567     245.6789   0.9012       0.9234       0.9123     MALICIOUS 
═══════════════════════════════════════════════════
```

## Mathematical Formulation

### Spectral Norm Computation

For each client *i*:

1. **Compute Delta**: Δᵢ = Rᵢ - R_global
2. **Center Delta**: Δ_centered = Δ - mean(Δ)
3. **Covariance**: C = (Δ_centered^T @ Δ_centered) / (n-1)
4. **Eigenvalues**: λ₁ ≥ λ₂ ≥ ... ≥ λ_d
5. **Spectral Norm**: ||Δᵢ||_spectral = λ₁ (largest eigenvalue)

### Delta Norm Computation

Delta Norm = ||Rᵢ - R_global||_F = √(Σᵢⱼ (rᵢⱼ - r_globalⱼ)²)

### Signal Combination

Combined Score = w_spectral × spectral_norm + w_delta × delta_norm

Where w_spectral + w_delta = 1.0

## Examples

### Example 1: Detect Neurotoxin Attack on CIFAR-10

```bash
python main.py -cn cifar10 \
    aggregator=fera \
    atk_config=cifar10_multishot \
    atk_config.model_poison_method=neurotoxin \
    num_rounds=600 \
    aggregator_config.fera.top_k_percent=0.2 \
    save_logging=csv
```

### Example 2: Multi-Layer Detection for Edge-Case Attack

```bash
python main.py -cn emnist \
    aggregator=fera \
    aggregator_config.fera.use_multi_layer=true \
    "aggregator_config.fera.layers=['layer2','layer3','penultimate']" \
    aggregator_config.fera.spectral_weight=0.7 \
    aggregator_config.fera.delta_weight=0.3 \
    atk_config=emnist_multishot \
    atk_config.data_poison_method=edge_case \
    num_rounds=600
```

### Example 3: Conservative Detection (Flag Fewer Clients)

```bash
python main.py -cn cifar10 \
    aggregator=fera \
    aggregator_config.fera.top_k_percent=0.1 \
    aggregator_config.fera.root_size=128 \
    atk_config=cifar10_multishot \
    num_rounds=600
```

## Validation

The implementation has been validated with comprehensive tests:

```bash
# Run validation tests
python test_fera_defense.py
```

**Test Coverage:**
- ✅ Signal computation (spectral norm, delta norm)
- ✅ Robust normalization (median + IQR)
- ✅ Threshold-based detection
- ✅ Edge case handling (insufficient samples, identical scores, single client)

## Performance Tips

1. **Root Dataset Size**: Larger root datasets (e.g., 128) provide more stable feature representations but slower computation
2. **Multi-Layer Analysis**: Use for sophisticated attacks but increases computation time
3. **Top-K Percent**: Adjust based on expected attack proportion (0.1-0.5 typical range)
4. **Signal Weights**: Spectral norm better for targeted attacks, delta norm for indiscriminate perturbations

## Troubleshooting

### Issue: "Insufficient samples" warnings
**Solution**: Increase `root_size` parameter (e.g., 128 or 256)

### Issue: All clients flagged or none flagged
**Solution**: Adjust `top_k_percent` parameter or check signal weights

### Issue: Layer not found in multi-layer mode
**Solution**: Verify layer names match your model architecture. Use `'penultimate'` for general models.

## Citation

If you use FeRA in your research, please cite:

```bibtex
@misc{fera2025,
  title={FeRA: Feature Representation Anomaly Defense for Federated Learning},
  author={BackFed Team},
  year={2025}
}
```

## Related Defenses

- **FedSPECTRE**: Uses CKA similarity and spectral analysis
- **DeepSight**: Clustering-based backdoor detection
- **RFLBAT**: PCA-based malicious update detection
- **FLDetector**: Sliding window approach for anomaly detection

## Support

For issues or questions:
1. Check logs for detailed detection results
2. Verify configuration parameters
3. Run validation tests: `python test_fera_defense.py`
4. Refer to BackFed documentation

---

**Status**: ✅ Fully integrated and validated
**Version**: 1.0
**Last Updated**: 2025-10-20

