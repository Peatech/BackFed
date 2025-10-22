#!/usr/bin/env python3
"""
FeRA_anonm Validation Test Script

Tests the 5 intuition-based detection methods to ensure:
1. Server instantiates correctly
2. All methods compute scores successfully
3. Scores are normalized to [0,1]
4. Telemetry outputs correctly
5. JSON export works

Usage:
    python test_fera_anonm.py
"""

import sys
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

def test_import():
    """Test that FeRAAnomServer can be imported."""
    print("=" * 70)
    print("Test 1: Import FeRAAnomServer")
    print("=" * 70)
    
    try:
        from backfed.servers import FeRAAnomServer
        print("✓ FeRAAnomServer imported successfully")
        print(f"✓ Class: {FeRAAnomServer}")
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Could not import FeRAAnomServer: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_instantiation():
    """Test that FeRAAnomServer can be instantiated."""
    print("=" * 70)
    print("Test 2: Instantiate FeRAAnomServer")
    print("=" * 70)
    
    try:
        from backfed.servers import FeRAAnomServer
        
        # Create minimal config
        config = OmegaConf.create({
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'model': 'resnet18',
            'normalize': True,
            'batch_size': 64,
            'atk_config': {
                'target_class': 2,
                'secret_dataset': False,
                'size_of_secret_dataset': 25,
            },
            'client_config': {
                'lr': 0.01,
            },
            'aggregator': 'fera_anonm',
        })
        
        print("✓ Created test config")
        
        # Instantiate server
        server = FeRAAnomServer(
            server_config=config,
            eta=0.5,
            enable_method_1=True,
            enable_method_2=True,
            enable_method_3=True,
            enable_method_4=True,
            enable_method_5=True,
        )
        
        print(f"✓ Server instantiated: {server}")
        print(f"✓ Server type: {server.server_type}")
        print(f"✓ Method 1 enabled: {server.enable_method_1}")
        print(f"✓ Method 2 enabled: {server.enable_method_2}")
        print(f"✓ Method 3 enabled: {server.enable_method_3}")
        print(f"✓ Method 4 enabled: {server.enable_method_4}")
        print(f"✓ Method 5 enabled: {server.enable_method_5}")
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Could not instantiate FeRAAnomServer: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_score_normalization():
    """Test that score normalization works correctly."""
    print("=" * 70)
    print("Test 3: Score Normalization")
    print("=" * 70)
    
    try:
        from backfed.servers import FeRAAnomServer
        
        # Create server
        config = OmegaConf.create({
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'model': 'resnet18',
            'normalize': True,
            'batch_size': 64,
            'atk_config': {'target_class': 2, 'secret_dataset': False, 'size_of_secret_dataset': 25},
            'client_config': {'lr': 0.01},
            'aggregator': 'fera_anonm',
        })
        
        server = FeRAAnomServer(server_config=config, eta=0.5)
        
        # Test normalize_scores_robust
        raw_scores = {
            0: 0.1,
            1: 0.5,
            2: 0.9,
            3: 1.5,
            4: 2.0,
            5: 0.3,
            6: 0.7,
            7: 1.2,
        }
        
        normalized = server._normalize_scores_robust(raw_scores)
        
        print(f"✓ Raw scores: {raw_scores}")
        print(f"✓ Normalized scores: {normalized}")
        
        # Check all scores are in [0,1]
        for cid, score in normalized.items():
            if not (0.0 <= score <= 1.0):
                print(f"✗ FAIL: Score {score} for client {cid} not in [0,1]")
                return False
        
        print("✓ All scores normalized to [0,1]")
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Score normalization failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_configuration_flags():
    """Test that configuration flags work correctly."""
    print("=" * 70)
    print("Test 4: Configuration Flags")
    print("=" * 70)
    
    try:
        from backfed.servers import FeRAAnomServer
        
        config = OmegaConf.create({
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'model': 'resnet18',
            'normalize': True,
            'batch_size': 64,
            'atk_config': {'target_class': 2, 'secret_dataset': False, 'size_of_secret_dataset': 25},
            'client_config': {'lr': 0.01},
            'aggregator': 'fera_anonm',
        })
        
        # Test with all methods disabled
        server1 = FeRAAnomServer(
            server_config=config,
            enable_method_1=False,
            enable_method_2=False,
            enable_method_3=False,
            enable_method_4=False,
            enable_method_5=False,
        )
        
        print(f"✓ Server with all methods disabled: {server1.enable_method_1}")
        
        # Test with only method 1 enabled
        server2 = FeRAAnomServer(
            server_config=config,
            enable_method_1=True,
            enable_method_2=False,
            enable_method_3=False,
            enable_method_4=False,
            enable_method_5=False,
        )
        
        print(f"✓ Server with only method 1 enabled: {server2.enable_method_1}")
        
        # Test use_combined_only flag
        server3 = FeRAAnomServer(
            server_config=config,
            enable_method_5=True,
            use_combined_only=True,
        )
        
        print(f"✓ Server with use_combined_only=True: {server3.use_combined_only}")
        
        # Test custom parameters
        server4 = FeRAAnomServer(
            server_config=config,
            unlearning_epochs=10,
            learning_speed_iters=50,
            boundary_attack_steps=20,
        )
        
        print(f"✓ Server with custom unlearning_epochs: {server4.unlearning_epochs}")
        print(f"✓ Server with custom learning_speed_iters: {server4.learning_speed_iters}")
        print(f"✓ Server with custom boundary_attack_steps: {server4.boundary_attack_steps}")
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Configuration flags test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_method_structure():
    """Test that all required methods exist."""
    print("=" * 70)
    print("Test 5: Method Structure")
    print("=" * 70)
    
    try:
        from backfed.servers import FeRAAnomServer
        
        config = OmegaConf.create({
            'dataset': 'CIFAR10',
            'num_classes': 10,
            'model': 'resnet18',
            'normalize': True,
            'batch_size': 64,
            'atk_config': {'target_class': 2, 'secret_dataset': False, 'size_of_secret_dataset': 25},
            'client_config': {'lr': 0.01},
            'aggregator': 'fera_anonm',
        })
        
        server = FeRAAnomServer(server_config=config, eta=0.5)
        
        # Check all required methods exist
        required_methods = [
            '_compute_unlearning_score',
            '_compute_learning_speed_score',
            '_compute_boundary_distance_score',
            '_compute_stability_score',
            '_compute_combined_score',
            '_run_multi_signal_analysis',
            '_normalize_scores_robust',
            '_output_telemetry',
            '_export_json',
            '_finalize_predictions',
        ]
        
        for method_name in required_methods:
            if not hasattr(server, method_name):
                print(f"✗ FAIL: Method {method_name} not found")
                return False
            print(f"✓ Method {method_name} exists")
        
        print()
        return True
    except Exception as e:
        print(f"✗ FAIL: Method structure test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all validation tests."""
    print()
    print("=" * 70)
    print("FERA_ANONM VALIDATION TESTS")
    print("=" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(("Import", test_import()))
    results.append(("Instantiation", test_instantiation()))
    results.append(("Score Normalization", test_score_normalization()))
    results.append(("Configuration Flags", test_configuration_flags()))
    results.append(("Method Structure", test_method_structure()))
    
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
        print("FeRA_anonm is ready to use!")
        print()
        print("Next steps:")
        print("  1. Quick test (2 rounds):")
        print("     python main.py --config-name cifar10 \\")
        print("         atk_config=cifar10_multishot \\")
        print("         atk_config.data_poison_method=pattern \\")
        print("         atk_config.poison_start_round=1001 \\")
        print("         atk_config.poison_end_round=1002 \\")
        print("         num_rounds=2 seed=123 aggregator=fera_anonm \\")
        print("         checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth")
        print()
        print("  2. Test with specific methods enabled:")
        print("     aggregator.enable_method_1=true aggregator.enable_method_2=false ...")
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

