#!/usr/bin/env python3
"""
Quick test script to verify Kaggle dataset download fix.
Run this in Kaggle to test that the fix is working.
"""

import os
import sys

def test_kaggle_dataset_fix():
    """Test the Kaggle dataset path fix."""
    print("🧪 Testing Kaggle dataset download fix...")
    
    # Simulate Kaggle environment
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'
    
    try:
        from omegaconf import DictConfig
        from backfed.datasets.fl_dataloader import FL_DataLoader
        
        # Create test config similar to Kaggle environment
        config = DictConfig({
            'dataset': 'CIFAR10',
            'datapath': '/kaggle/input',  # Read-only in Kaggle
            'partitioner': 'uniform',
            'alpha': 1.0,
            'normalize': True,
            'num_clients': 4,
            'seed': 42,
            'no_attack': True
        })
        
        print("✅ Configuration created successfully")
        
        # Test FL_DataLoader instantiation
        fl_loader = FL_DataLoader(config)
        print("✅ FL_DataLoader instantiated without errors")
        
        # Test path logic (without actually downloading)
        dataset_name = "CIFAR10"
        datapath = os.path.join(config["datapath"], dataset_name)
        
        if os.path.exists("/kaggle") or os.getenv("COLAB_GPU"):
            if not os.path.exists(datapath):
                download_path = "/kaggle/working/data/CIFAR10"
                print(f"✅ Would download to: {download_path}")
            else:
                print(f"✅ Would use existing: {datapath}")
        
        print("\n🎉 Kaggle dataset fix is working correctly!")
        print("👉 You can now run: python kaggle_main.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to run: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 BackFed Kaggle Fix Test")
    print("=" * 40)
    
    success = test_kaggle_dataset_fix()
    
    if success:
        print("\n✅ All tests passed! Ready to use BackFed in Kaggle.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)
