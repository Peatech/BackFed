#!/usr/bin/env python3
"""
FeRA Defense CLI Tool

A command-line interface for running FeRA defense experiments with BackFed.
Supports configurable parameters including start/end epochs, multi-layer detection,
and various attack configurations.

Usage:
    python fera_cli.py --help
    python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101
    python fera_cli.py --dataset emnist --attack neurotoxin --multi-layer --layers layer2,layer3,penultimate
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional


class FeRACLI:
    """Command-line interface for FeRA defense experiments."""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.args = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all FeRA options."""
        parser = argparse.ArgumentParser(
            description="FeRA Defense CLI - Run backdoor detection experiments",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic pattern attack detection
  python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1101
  
  # Multi-layer detection against neurotoxin
  python fera_cli.py --dataset cifar10 --attack neurotoxin --multi-layer --layers layer2,layer3,penultimate
  
  # Conservative detection with custom parameters
  python fera_cli.py --dataset emnist --attack pattern --top-k 0.2 --spectral-weight 0.7 --delta-weight 0.3
  
  # Full experiment with all options
  python fera_cli.py --dataset cifar10 --attack pattern --start-epoch 1001 --end-epoch 1601 \\
                     --multi-layer --layers layer2,layer3,penultimate --combine-method max \\
                     --top-k 0.3 --spectral-weight 0.6 --delta-weight 0.4 --root-size 128 \\
                     --checkpoint checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth \\
                     --num-rounds 600 --training-mode parallel --num-gpus 1.0 --save-logging csv
            """
        )
        
        # Required arguments
        parser.add_argument(
            '--dataset', '-d',
            required=True,
            choices=['cifar10', 'emnist', 'femnist', 'tinyimagenet', 'reddit', 'sentiment140'],
            help='Dataset to use for the experiment'
        )
        
        parser.add_argument(
            '--attack', '-a',
            required=True,
            choices=['pattern', 'pixel', 'badnets', 'blended', 'distributed', 'edge_case', 
                    'a3fl', 'iba', 'neurotoxin', 'chameleon', 'anticipate'],
            help='Attack method to test against'
        )
        
        # Epoch configuration
        parser.add_argument(
            '--start-epoch', '-s',
            type=int,
            default=1001,
            help='Starting epoch for attack (default: 1001)'
        )
        
        parser.add_argument(
            '--end-epoch', '-e',
            type=int,
            default=1101,
            help='Ending epoch for attack (default: 1101)'
        )
        
        # FeRA-specific parameters
        parser.add_argument(
            '--spectral-weight',
            type=float,
            default=0.6,
            help='Weight for spectral norm signal (default: 0.6)'
        )
        
        parser.add_argument(
            '--delta-weight',
            type=float,
            default=0.4,
            help='Weight for delta norm signal (default: 0.4)'
        )
        
        parser.add_argument(
            '--top-k',
            type=float,
            default=0.5,
            help='Top-K percent to flag as malicious (default: 0.5)'
        )
        
        parser.add_argument(
            '--root-size',
            type=int,
            default=64,
            help='Size of root dataset for feature extraction (default: 64)'
        )
        
        # Multi-layer options
        parser.add_argument(
            '--multi-layer',
            action='store_true',
            help='Enable multi-layer feature extraction'
        )
        
        parser.add_argument(
            '--layers',
            type=str,
            default='penultimate',
            help='Comma-separated list of layers (default: penultimate). Options: penultimate,layer2,layer3,layer4'
        )
        
        parser.add_argument(
            '--combine-method',
            choices=['mean', 'max', 'vote'],
            default='mean',
            help='Method to combine multi-layer scores (default: mean)'
        )
        
        # Training configuration
        parser.add_argument(
            '--num-rounds',
            type=int,
            default=600,
            help='Total number of training rounds (default: 600)'
        )
        
        parser.add_argument(
            '--num-clients',
            type=int,
            default=100,
            help='Total number of clients (default: 100)'
        )
        
        parser.add_argument(
            '--num-clients-per-round',
            type=int,
            default=10,
            help='Number of clients selected per round (default: 10)'
        )
        
        parser.add_argument(
            '--training-mode',
            choices=['parallel', 'sequential'],
            default='parallel',
            help='Training mode (default: parallel)'
        )
        
        parser.add_argument(
            '--num-gpus',
            type=float,
            default=1.0,
            help='GPU fraction per client (default: 1.0)'
        )
        
        parser.add_argument(
            '--num-cpus',
            type=int,
            default=12,
            help='CPU cores per client (default: 12)'
        )
        
        # Checkpoint and logging
        parser.add_argument(
            '--checkpoint',
            type=str,
            help='Path to checkpoint file or round number to resume from'
        )
        
        parser.add_argument(
            '--save-logging',
            choices=['csv', 'wandb', 'both'],
            default='csv',
            help='Logging method (default: csv)'
        )
        
        parser.add_argument(
            '--dir-tag',
            type=str,
            help='Custom directory tag for organizing results'
        )
        
        # Advanced options
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode (use subset of data)'
        )
        
        parser.add_argument(
            '--debug-fraction',
            type=float,
            default=0.1,
            help='Fraction of data to use in debug mode (default: 0.1)'
        )
        
        parser.add_argument(
            '--seed',
            type=int,
            default=123,
            help='Random seed for reproducibility (default: 123)'
        )
        
        parser.add_argument(
            '--alpha',
            type=float,
            default=0.9,
            help='Dirichlet distribution parameter for non-IID data (default: 0.9)'
        )
        
        # Output options
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Print the command that would be executed without running it'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        self.args = self.parser.parse_args(args)
        self._validate_args()
        return self.args
    
    def _validate_args(self):
        """Validate parsed arguments."""
        # Validate weights sum to 1.0
        total_weight = self.args.spectral_weight + self.args.delta_weight
        if not abs(total_weight - 1.0) < 1e-6:
            print(f"Warning: Spectral weight ({self.args.spectral_weight}) + Delta weight ({self.args.delta_weight}) = {total_weight}")
            print("Weights will be normalized to sum to 1.0")
        
        # Validate epoch range
        if self.args.start_epoch >= self.args.end_epoch:
            raise ValueError("Start epoch must be less than end epoch")
        
        # Validate layers if multi-layer is enabled
        if self.args.multi_layer:
            layers = [layer.strip() for layer in self.args.layers.split(',')]
            valid_layers = ['penultimate', 'layer2', 'layer3', 'layer4', 'all']
            invalid_layers = [layer for layer in layers if layer not in valid_layers]
            if invalid_layers:
                print(f"Warning: Invalid layers {invalid_layers}. Valid options: {valid_layers}")
    
    def _build_command(self) -> List[str]:
        """Build the main.py command with all parameters."""
        cmd = ['python', 'main.py']
        
        # Basic configuration
        cmd.extend(['-cn', self.args.dataset])
        cmd.append('aggregator=fera')
        
        # Attack configuration
        if self.args.attack in ['pattern', 'pixel', 'badnets', 'blended', 'distributed', 'edge_case', 'a3fl', 'iba']:
            cmd.append('atk_config=cifar10_multishot' if self.args.dataset == 'cifar10' else f'atk_config={self.args.dataset}_multishot')
            cmd.append(f'atk_config.data_poison_method={self.args.attack}')
        else:
            cmd.append('atk_config=cifar10_multishot' if self.args.dataset == 'cifar10' else f'atk_config={self.args.dataset}_multishot')
            cmd.append(f'atk_config.model_poison_method={self.args.attack}')
        
        # Epoch configuration
        cmd.append(f'atk_config.poison_start_round={self.args.start_epoch}')
        cmd.append(f'atk_config.poison_end_round={self.args.end_epoch}')
        
        # FeRA-specific parameters
        cmd.append(f'aggregator_config.fera.spectral_weight={self.args.spectral_weight}')
        cmd.append(f'aggregator_config.fera.delta_weight={self.args.delta_weight}')
        cmd.append(f'aggregator_config.fera.top_k_percent={self.args.top_k}')
        cmd.append(f'aggregator_config.fera.root_size={self.args.root_size}')
        
        # Multi-layer configuration
        if self.args.multi_layer:
            cmd.append('aggregator_config.fera.use_multi_layer=true')
            layers_list = [layer.strip() for layer in self.args.layers.split(',')]
            layers_str = "['" + "','".join(layers_list) + "']"
            cmd.append(f'aggregator_config.fera.layers={layers_str}')
            cmd.append(f'aggregator_config.fera.combine_layers_method={self.args.combine_method}')
        
        # Training configuration
        cmd.append(f'num_rounds={self.args.num_rounds}')
        cmd.append(f'num_clients={self.args.num_clients}')
        cmd.append(f'num_clients_per_round={self.args.num_clients_per_round}')
        cmd.append(f'training_mode={self.args.training_mode}')
        cmd.append(f'num_gpus={self.args.num_gpus}')
        cmd.append(f'num_cpus={self.args.num_cpus}')
        
        # Checkpoint
        if self.args.checkpoint:
            cmd.append(f'checkpoint={self.args.checkpoint}')
        
        # Logging
        cmd.append(f'save_logging={self.args.save_logging}')
        
        # Directory tag
        if self.args.dir_tag:
            cmd.append(f'dir_tag={self.args.dir_tag}')
        else:
            # Auto-generate dir tag
            attack_type = 'data' if self.args.attack in ['pattern', 'pixel', 'badnets', 'blended', 'distributed', 'edge_case', 'a3fl', 'iba'] else 'model'
            dir_tag = f'{self.args.dataset}_fera_{attack_type}({self.args.attack})_epochs_{self.args.start_epoch}_{self.args.end_epoch}'
            cmd.append(f'dir_tag={dir_tag}')
        
        # Debug options
        if self.args.debug:
            cmd.append('debug=true')
            cmd.append(f'debug_fraction_data={self.args.debug_fraction}')
        
        # Other options
        cmd.append(f'seed={self.args.seed}')
        cmd.append(f'alpha={self.args.alpha}')
        
        return cmd
    
    def run(self) -> int:
        """Run the FeRA experiment."""
        if not self.args:
            self.parse_args()
        
        # Build command
        cmd = self._build_command()
        
        # Print command if verbose or dry-run
        if self.args.verbose or self.args.dry_run:
            print("Command to execute:")
            print(" ".join(cmd))
            print()
        
        if self.args.dry_run:
            print("Dry run completed. Use --verbose to see full command.")
            return 0
        
        # Change to BackFed directory
        backfed_dir = Path(__file__).parent / 'scratch' / 'BackFed'
        if not backfed_dir.exists():
            backfed_dir = Path(__file__).parent
            if not (backfed_dir / 'main.py').exists():
                print(f"Error: Cannot find BackFed directory. Expected main.py in {backfed_dir}")
                return 1
        
        # Execute command
        try:
            print(f"Running FeRA experiment...")
            print(f"Dataset: {self.args.dataset}")
            print(f"Attack: {self.args.attack}")
            print(f"Epochs: {self.args.start_epoch}-{self.args.end_epoch}")
            print(f"Multi-layer: {self.args.multi_layer}")
            if self.args.multi_layer:
                print(f"Layers: {self.args.layers}")
                print(f"Combine method: {self.args.combine_method}")
            print(f"Top-K: {self.args.top_k}")
            print(f"Spectral weight: {self.args.spectral_weight}, Delta weight: {self.args.delta_weight}")
            print()
            
            result = subprocess.run(cmd, cwd=backfed_dir, check=True)
            print("\n✅ Experiment completed successfully!")
            return result.returncode
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Experiment failed with return code {e.returncode}")
            return e.returncode
        except KeyboardInterrupt:
            print("\n⚠️ Experiment interrupted by user")
            return 130
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return 1


def main():
    """Main entry point."""
    cli = FeRACLI()
    try:
        return cli.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
