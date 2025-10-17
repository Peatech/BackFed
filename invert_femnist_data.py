"""
Script to invert FEMNIST data (convert white background to black background)
This processes all .pt files in femnist_train folders and femnist_test.pt
"""

import torch
from pathlib import Path
import shutil
from tqdm import tqdm

def invert_femnist_file(file_path):
    """
    Load a FEMNIST .pt file, invert the images (1 - pixel_value), and save back.
    
    Args:
        file_path: Path to the .pt file
    """
    print(f"Processing: {file_path}")
    
    # Load the data
    data = torch.load(file_path, weights_only=False)
    
    # Check the data structure
    if isinstance(data, tuple) or isinstance(data, list):
        # Assuming format: (images, labels) or [images, labels]
        images = data[0]
        labels = data[1] if len(data) > 1 else None
        
        # Invert images: 1 - pixel_value
        print(f"  Original - Min: {images.min():.4f}, Max: {images.max():.4f}, Mean: {images.mean():.4f}")
        inverted_images = 1.0 - images
        print(f"  Inverted - Min: {inverted_images.min():.4f}, Max: {inverted_images.max():.4f}, Mean: {inverted_images.mean():.4f}")
        
        # Reconstruct data
        if labels is not None:
            inverted_data = (inverted_images, labels)
        else:
            inverted_data = (inverted_images,)
            
    elif isinstance(data, torch.Tensor):
        # Just a tensor of images
        print(f"  Original - Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")
        inverted_data = 1.0 - data
        print(f"  Inverted - Min: {inverted_data.min():.4f}, Max: {inverted_data.max():.4f}, Mean: {inverted_data.mean():.4f}")
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    # # Create backup
    # backup_path = file_path.with_suffix('.pt.backup')
    # if not backup_path.exists():
    #     print(f"  Creating backup: {backup_path}")
    #     shutil.copy2(file_path, backup_path)
    
    # Save inverted data
    torch.save(inverted_data, file_path)
    print(f"  ✓ Saved inverted data to: {file_path}\n")


def process_femnist_data(base_path):
    """
    Process all FEMNIST data files in the given base path.
    
    Args:
        base_path: Path to the FEMNIST data directory
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return
    
    # Process training data (in subdirectories)
    train_path = base_path / "femnist_train"
    if train_path.exists():
        print("=" * 60)
        print("Processing FEMNIST Training Data")
        print("=" * 60)
        
        # Get all .pt files in subdirectories
        pt_files = list(train_path.rglob("*.pt"))
        # Exclude backup files
        pt_files = [f for f in pt_files if not f.name.endswith('.backup')]
        
        print(f"Found {len(pt_files)} training files\n")
        
        for pt_file in tqdm(pt_files, desc="Processing training files"):
            invert_femnist_file(pt_file)
    else:
        print(f"Warning: Training data path not found: {train_path}")
    
    # Process test data
    test_path = base_path / "femnist_test.pt"
    if test_path.exists():
        print("\n" + "=" * 60)
        print("Processing FEMNIST Test Data")
        print("=" * 60)
        invert_femnist_file(test_path)
    else:
        print(f"Warning: Test data file not found: {test_path}")
    
    print("\n" + "=" * 60)
    print("FEMNIST Data Inversion Complete!")
    print("=" * 60)
    print("\nBackup files have been created with .pt.backup extension")
    print("To restore original data, rename .pt.backup files back to .pt")


if __name__ == "__main__":
    # Set the base path to FEMNIST data
    femnist_base_path = "/home/thinh.dd/BackFed/data/FEMNIST"
    
    print("FEMNIST Data Inversion Script")
    print("=" * 60)
    print(f"Base path: {femnist_base_path}")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("\nThis will invert all FEMNIST data (backups will be created).\nContinue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        process_femnist_data(femnist_base_path)
    else:
        print("Operation cancelled.")
