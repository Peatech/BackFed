#!/usr/bin/env python3
"""
Edge-Case Attack Validation Script

This script validates that the edge-case (semantic backdoor) attack
can be properly instantiated and used.

Usage:
    python test_edge_case.py
"""

import sys
import pickle
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf

def test_dataset_files():
    """Test that dataset files exist and can be loaded."""
    print("=" * 70)
    print("Test 1: Dataset Files")
    print("=" * 70)
    
    train_path = Path('backfed/poisons/shared/edge-case/southwest_images_new_train.pkl')
    test_path = Path('backfed/poisons/shared/edge-case/southwest_images_new_test.pkl')
    
    # Check existence
    if not train_path.exists():
        print(f"✗ FAIL: Training file not found: {train_path}")
        return False
    print(f"✓ Training file exists: {train_path}")
    
    if not test_path.exists():
        print(f"✗ FAIL: Test file not found: {test_path}")
        return False
    print(f"✓ Test file exists: {test_path}")
    
    # Try loading
    try:
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✓ Loaded training data: {len(train_data)} images")
        
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        print(f"✓ Loaded test data: {len(test_data)} images")
    except Exception as e:
        print(f"✗ FAIL: Could not load pickle files: {e}")
        return False
    
    # Check format
    if len(train_data) == 0 or len(test_data) == 0:
        print("✗ FAIL: Dataset is empty")
        return False
    
    sample = train_data[0]
    print(f"✓ Sample image type: {type(sample)}")
    print(f"✓ Sample image size: {sample.size if hasattr(sample, 'size') else 'N/A'}")
    
    print()
    return True


def test_edge_case_import():
    """Test that EdgeCase class can be imported."""
    print("=" * 70)
    print("Test 2: EdgeCase Import")
    print("=" * 70)
    
    try:
        from backfed.poisons import EdgeCase
        print("✓ EdgeCase imported successfully")
        print(f"✓ EdgeCase class: {EdgeCase}")
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Could not import EdgeCase: {e}")
        print()
        return False


def test_edge_case_instantiation():
    """Test that EdgeCase can be instantiated with minimal config."""
    print("=" * 70)
    print("Test 3: EdgeCase Instantiation")
    print("=" * 70)
    
    try:
        from backfed.poisons import EdgeCase
        
        # Create minimal config
        config = OmegaConf.create({
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'attack_type': 'all2one',
            'target_class': 2,
            'poison_rate': 0.3125,
            'poison_mode': 'online',
            'poisoned_lr': 0.05,
            'poison_epochs': 6,
        })
        
        print("✓ Created test config")
        
        # Instantiate EdgeCase
        poison = EdgeCase(config, client_id=0)
        print(f"✓ EdgeCase instantiated: {poison}")
        print(f"✓ Edge case train samples: {len(poison.edge_case_train)}")
        print(f"✓ Edge case test samples: {len(poison.edge_case_test)}")
        
        # Test poison_inputs method
        dummy_inputs = torch.randn(4, 3, 32, 32)
        poisoned = poison.poison_inputs(dummy_inputs)
        print(f"✓ poison_inputs works: {poisoned.shape}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Could not instantiate EdgeCase: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all validation tests."""
    print()
    print("=" * 70)
    print("EDGE-CASE ATTACK VALIDATION")
    print("=" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(("Dataset Files", test_dataset_files()))
    results.append(("EdgeCase Import", test_edge_case_import()))
    results.append(("EdgeCase Instantiation", test_edge_case_instantiation()))
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Edge-case attack is ready to use!")
        print()
        print("Next steps:")
        print("  1. Quick test (2 rounds):")
        print("     python main.py --config-name cifar10 \\")
        print("         atk_config=cifar10_multishot \\")
        print("         atk_config.data_poison_method=edge_case \\")
        print("         atk_config.poison_start_round=1001 \\")
        print("         atk_config.poison_end_round=1002 \\")
        print("         num_rounds=2 seed=123 aggregator=fera \\")
        print("         checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth")
        print()
        print("  2. Add to slurm_optimized.sbatch for full run")
        print()
        return 0
    else:
        print("=" * 70)
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        print()
        print("Please fix the issues above before proceeding.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

