#!/bin/bash --login
#SBATCH -J backfed-optimized
#SBATCH -p gpuA            # A100 80GB GPUs (best performance)
#SBATCH -G 2               # Request 2 GPUs (max allowed for free users)
#SBATCH -n 24              # 12 cores per GPU (2 GPUs × 12 cores = 24 cores)
#SBATCH -t 1-00:00:00      # 1 day walltime (adjust as needed)
#SBATCH --output=slurm-%j.out
#SBATCH --mem=64G          # Request 64GB RAM (32GB per GPU)

module purge
module load libs/cuda/12.8.1
module load apps/binapps/anaconda3/2024.10

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate BackFed_env

# Set PATH explicitly to ensure the environment python is used
export PATH="/mnt/iusers01/fse-ugpgt01/compsci01/m46997co/.conda/envs/BackFed_env/bin:$PATH"

cd /mnt/iusers01/fse-ugpgt01/compsci01/m46997co/scratch/BackFed

# Environment variables
export RAY_memory_monitor_refresh_ms=0
export WANDB_API_KEY="834655e3bcc06683105ccfd8cdfdb929bf41ff1e"
export WANDB_MODE="online"

# Verify GPU allocation
echo "=== GPU Allocation Verification ==="
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_NTASKS (CPU cores): $SLURM_NTASKS"
nvidia-smi

# Verify environment
echo "=== Environment Verification ==="
echo "Python path: $(which python)"
python --version
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo "=== Starting FeRA Defense Experiments ==="
echo "Testing 10 backdoor attacks with FeRA defense"
echo "Each experiment: 100 rounds (1001-1100)"
echo ""

##############################################################################
# DATA POISONING ATTACKS (7 attacks)
##############################################################################

# # Attack 1: Pattern (white square trigger)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=pattern \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# # Attack 2: Pixel (single pixel trigger)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=pixel \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# # Attack 3: BadNets (pattern-based backdoor)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=badnets \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# # Attack 4: Blended (blended trigger)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=blended \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# Attack 5: Distributed (distributed backdoor)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=distributed \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# # Attack 6: A3FL (adversarial adaptive trigger) - FIXED
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=a3fl \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# # Attack 7: IBA (invisible backdoor attack) - FIXED
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.data_poison_method=iba \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# ##############################################################################
# # MODEL POISONING ATTACKS (3 attacks)
# ##############################################################################

# # Attack 8: Neurotoxin (gradient masking)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.model_poison_method=neurotoxin \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# # Attack 9: Chameleon (supervised contrastive learning)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.model_poison_method=chameleon \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# Attack 10: Anticipate (adaptive optimization)
# python main.py --config-name cifar10 \
#     atk_config=cifar10_multishot \
#     atk_config.model_poison_method=anticipate \
#     atk_config.poison_start_round=1001 \
#     atk_config.poison_end_round=1101 \
#     training_mode=parallel \
#     num_gpus=1.0 \
#     num_cpus=12 \
#     num_rounds=100 \
#     seed=123 \
#     aggregator=fera \
#     checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_1000_dir_0.9.pth

# Edge-Case (Semantic Backdoor)
echo "=== Running Edge-Case (Semantic Backdoor) ==="
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

echo ""
echo "=== All FeRA Experiments Complete ==="
echo "Tested 11 backdoor attacks:"
echo "  Data Poisoning: pattern, pixel, badnets, blended, distributed, edge_case, a3fl, iba"
echo "  Model Poisoning: neurotoxin, chameleon, anticipate"
echo ""
