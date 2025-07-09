#!/bin/bash

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset>"
    echo "Available datasets: cifar10, femnist, emnist, mnist, tinyimagenet, reddit, sentiment140, all"
    exit 1
fi

DATASET=$1

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Change to checkpoints directory
cd checkpoints

# Function to download a specific dataset
download_model() {
    local dataset=$1
    local model_path=""
    local google_drive_id=""
    
    case $dataset in
        "cifar10")
            model_path="CIFAR10_unweighted_fedavg"
            google_drive_id="1oGGXEGf9FtgZA7dPZvMuGzFXS2TsSKdY"
            ;;
        "emnist")
            model_path="checkpoints/EMNIST_BYCLASS_unweighted_fedavg"
            google_drive_id="1bALJ9oxBjz4-GOwZy0zF9epIZkMOHNk8"
            ;;
        "femnist")
            model_path="checkpoints/EMNIST_BYCLASS_unweighted_fedavg"
            google_drive_id="1bALJ9oxBjz4-GOwZy0zF9epIZkMOHNk8"
            ;;
        "mnist")
            model_path="checkpoints/EMNIST_BYCLASS_unweighted_fedavg"
            google_drive_id="1bALJ9oxBjz4-GOwZy0zF9epIZkMOHNk8"
            ;;
        "tinyimagenet")
            model_path="TINYIMAGENET_unweighted_fedavg"
            google_drive_id="1m0FMpuS9RuA6LVvBe2vAfW7RHrmU9f67"
            ;;
        "reddit")
            model_path="REDDIT_unweighted_fedavg"
            google_drive_id="12HjxDq_S1gLbfhOVO0DsZGW0bGpMdwSf"
            ;;
        "sentiment140")
            model_path="SENTIMENT140_unweighted_fedavg"
            google_drive_id="17d9fIz8DwxU_1dDcwt67e45uGWTggElP"
            ;;
        *)
            echo "Unknown dataset: $dataset"
            return 1
            ;;
    esac
    
    if [ ! -d "$model_path" ]; then
        echo "Downloading $dataset models..."
        gdown "$google_drive_id" -O "${dataset}_models.zip"
        unzip "${dataset}_models.zip"
        rm "${dataset}_models.zip"
    else
        echo "$dataset models already exist, skipping..."
    fi
}

# Download models based on dataset argument
if [ "$DATASET" = "all" ]; then
    echo "Downloading all checkpoints..."
    download_model "cifar10"
    download_model "femnist"
    download_model "tinyimagenet"
    download_model "reddit"
    download_model "sentiment140"
else
    download_model "$DATASET"
fi

echo "Done!"