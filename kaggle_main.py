#!/usr/bin/env python3
"""
Kaggle/Colab entry point for BackFed.

This script provides a single-process, lightweight entry point optimized for 
Kaggle and Google Colab environments. It automatically detects the environment
and applies conservative defaults while preserving all BackFed functionality.

Usage:
    python kaggle_main.py                    # Use default CIFAR10 config
    python kaggle_main.py --cn cifar10       # Explicitly specify config
    python kaggle_main.py dataset=MNIST      # Override specific parameters
    python kaggle_main.py --help             # Show available options

Environment Detection:
    - Automatically detects Kaggle (/kaggle/input exists) 
    - Automatically detects Colab (via environment variables)
    - Falls back to safe single-process defaults

Key Differences from main.py:
    - Forces training_mode=sequential (no Ray)
    - Forces save_logging=csv (unless ENABLE_WANDB=1)
    - Conservative resource defaults (small models, fewer clients)
    - Deterministic training enabled by default
    - Progress bars disabled by default in notebooks
    - Graceful handling of missing pretrained weights
"""

import os
import sys
import warnings
from typing import Optional, Dict, Any

# Add the current directory to Python path to import backfed modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import omegaconf
import torch
import traceback

from rich.traceback import install
from rich.console import Console
from backfed.servers.base_server import BaseServer
from backfed.utils import system_startup, log, set_seed
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from logging import ERROR, INFO, WARNING

# Console for rich output
console = Console()

def detect_environment() -> str:
    """
    Detect if we're running in Kaggle, Colab, or local environment.
    
    Returns:
        str: 'kaggle', 'colab', or 'local'
    """
    # Check for Kaggle environment
    if os.path.exists('/kaggle/input') or os.path.exists('/kaggle/working'):
        return 'kaggle'
    
    # Check for Google Colab environment 
    if 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules:
        return 'colab'
    
    # Default to local
    return 'local'

def get_environment_defaults(env: str) -> Dict[str, Any]:
    """
    Get environment-specific default overrides.
    
    Args:
        env: Environment type ('kaggle', 'colab', 'local')
        
    Returns:
        Dict of configuration overrides
    """
    base_overrides = {
        'training_mode': 'sequential',
        'deterministic': True,
        'disable_progress_bar': True,
        'num_workers': 0,
        'pin_memory': False,
    }
    
    if env == 'kaggle':
        return {
            **base_overrides,
            'outputs_root': '/kaggle/working',
            'datapath': '/kaggle/input',
            'save_logging': 'csv',
            'cuda_visible_devices': '0',
            'num_gpus': 1.0,
            # Conservative FL settings
            'num_rounds': 50,
            'num_clients': 20, 
            'num_clients_per_round': 5,
            'client_config.local_epochs': 1,
            'client_config.batch_size': 32,
            'test_batch_size': 256,
            'test_every': 5,
        }
    
    elif env == 'colab':
        return {
            **base_overrides,
            'outputs_root': '/content',
            'save_logging': 'csv',
            'cuda_visible_devices': '0', 
            'num_gpus': 1.0,
            # Conservative FL settings
            'num_rounds': 50,
            'num_clients': 20,
            'num_clients_per_round': 5,
            'client_config.local_epochs': 1, 
            'client_config.batch_size': 32,
            'test_batch_size': 256,
            'test_every': 5,
        }
    
    else:  # local
        return {
            **base_overrides,
            'save_logging': 'csv',
            # Keep original settings but force sequential mode
            'num_rounds': 100,  # Slightly reduced from default 600
            'num_clients': 50,  # Reduced from default 100 
        }

def check_wandb_enabled() -> bool:
    """Check if WandB logging is explicitly enabled via environment variable."""
    return os.getenv('ENABLE_WANDB', '0').lower() in ('1', 'true', 'yes')

def check_pretrained_weights(config: DictConfig, env: str) -> None:
    """
    Check if pretrained weights are available and warn if missing.
    
    Args:
        config: Hydra configuration
        env: Environment type
    """
    if config.get('checkpoint') and config.checkpoint != 'Null':
        checkpoint_path = config.checkpoint
        if isinstance(checkpoint_path, (int, str)) and checkpoint_path != "wandb":
            # Check if checkpoint file exists
            if isinstance(checkpoint_path, int):
                # Round number - construct path
                dataset = config.dataset
                model = config.model  
                aggregator = config.aggregator
                checkpoint_path = f"checkpoints/{dataset}_{aggregator}/{model}_round_{checkpoint_path}.pth"
            
            if not os.path.exists(checkpoint_path):
                warning_msg = f"""
⚠️  Pretrained checkpoint not found: {checkpoint_path}

In {env} environments, pretrained weights must be uploaded as datasets.
To fix this:
1. Set checkpoint: Null to train from scratch (recommended for quick testing)
2. Or upload the required checkpoints as a Kaggle dataset/Colab file

Continuing with random initialization...
"""
                console.print(warning_msg, style="yellow")
                log(WARNING, f"Checkpoint {checkpoint_path} not found, using random initialization")
                config.checkpoint = None

def apply_environment_config(config: DictConfig) -> DictConfig:
    """
    Apply environment-specific configurations and overrides.
    
    Args:
        config: Original Hydra configuration
        
    Returns:
        Modified configuration
    """
    env = detect_environment()
    console.print(f"🔍 Detected environment: [bold blue]{env}[/bold blue]")
    
    # Get environment defaults
    env_defaults = get_environment_defaults(env)
    
    # Apply environment defaults (only if not already set by user)
    for key, value in env_defaults.items():
        if '.' in key:
            # Handle nested keys like 'client_config.batch_size'
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            if keys[-1] not in current:
                current[keys[-1]] = value
        else:
            if key not in config:
                config[key] = value
    
    # Force WandB settings based on environment variable
    if check_wandb_enabled():
        if env in ['kaggle', 'colab']:
            console.print("📊 WandB enabled via ENABLE_WANDB environment variable")
            config.save_logging = 'wandb'
        else:
            config.save_logging = 'both'  # Local can use both
    else:
        # Force CSV logging in cloud environments unless explicitly enabled
        if env in ['kaggle', 'colab']:
            config.save_logging = 'csv'
    
    # Ensure CUDA devices are properly set
    if not torch.cuda.is_available():
        console.print("⚠️  CUDA not available, falling back to CPU")
        config.cuda_visible_devices = ""
        config.num_gpus = 0
    
    # Check pretrained weights
    check_pretrained_weights(config, env)
    
    # Log final configuration
    console.print(f"⚙️  Training mode: [bold green]{config.training_mode}[/bold green]")
    console.print(f"📝 Logging mode: [bold green]{config.save_logging}[/bold green]")  
    console.print(f"🔧 Output root: [bold green]{config.get('outputs_root', '.')}[/bold green]")
    console.print(f"🎯 Deterministic: [bold green]{config.deterministic}[/bold green]")
    
    return config

def validate_config(config: DictConfig) -> None:
    """
    Validate configuration for Kaggle/Colab compatibility.
    
    Args:
        config: Configuration to validate
    """
    # Ensure sequential mode (critical for single-process)
    if config.training_mode != 'sequential':
        console.print("⚠️  Forcing training_mode=sequential for Kaggle/Colab compatibility")
        config.training_mode = 'sequential'
    
    # Ensure reasonable resource limits
    if config.num_clients > 100:
        console.print(f"⚠️  Reducing num_clients from {config.num_clients} to 100 for faster execution")
        config.num_clients = 100
    
    if config.num_rounds > 1000:
        console.print(f"⚠️  Large num_rounds ({config.num_rounds}) may take a long time to complete")

def create_output_directories(config: DictConfig) -> None:
    """Create necessary output directories."""
    outputs_root = config.get('outputs_root', '.')
    
    # Create output directories
    directories = [
        outputs_root,
        os.path.join(outputs_root, 'outputs'),
        os.path.join(outputs_root, 'csv_results'), 
        os.path.join(outputs_root, 'checkpoints'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Use the kaggle config as default, but allow CLI overrides
@hydra.main(config_path="config", config_name="kaggle", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main entry point for Kaggle/Colab BackFed execution.
    
    Args:
        config: Hydra configuration (starts with kaggle.yaml defaults)
    """
    try:
        # Welcome message
        console.print("\n🚀 [bold blue]BackFed - Kaggle/Colab Mode[/bold blue]")
        console.print("Optimized for single-process execution in cloud notebooks\n")
        
        # Apply environment-specific configurations
        config = apply_environment_config(config)
        
        # Validate configuration
        validate_config(config)
        
        # Create output directories
        create_output_directories(config)
        
        # Initialize system (this will skip Ray initialization in sequential mode)
        system_startup(config)
        
        # Instantiate and run server
        aggregator = config["aggregator"]
        console.print(f"🏃 Starting federated learning with [bold green]{aggregator}[/bold green] aggregator")
        
        server: BaseServer = instantiate(config.aggregator_config[aggregator], server_config=config, _recursive_=False)
        server.run_experiment()
        
        console.print("\n✅ [bold green]Experiment completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n⏹️  [yellow]Experiment interrupted by user[/yellow]")
        sys.exit(1)
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        console.print(f"\n❌ [bold red]Error occurred:[/bold red] {e}")
        log(ERROR, f"Error: {e}\n{error_traceback}")
        
        # Provide helpful error messages for common issues
        if "Ray" in str(e) and "not available" in str(e):
            console.print("\n💡 [blue]Hint:[/blue] This error suggests Ray is required but not available.")
            console.print("   Try installing Ray: !pip install ray")
            console.print("   Or ensure training_mode=sequential in your config")
        
        sys.exit(1)

if __name__ == "__main__":
    # Suppress warnings for cleaner notebook output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Rich traceback installation
    OmegaConf.register_new_resolver("eval", eval)
    install(show_locals=False, suppress=[hydra, omegaconf, torch])
    
    # Set Hydra environment variables for better error reporting
    os.environ["HYDRA_FULL_ERROR"] = "1"
    
    # Run main function
    main()
