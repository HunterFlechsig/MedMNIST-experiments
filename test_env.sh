#!/bin/bash

set -e

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

source activate image_pytorch

cd $HOME/MedMNIST-experiments/MedMNIST2D

echo "========================================="
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "========================================="

echo
echo "===== Python Environment ====="
echo "which python:"
which python

echo
echo "python --version:"
python --version

echo
echo "CONDA_PREFIX:"
echo "$CONDA_PREFIX"

echo
echo "Python executable:"
python -c "import sys; print(sys.executable)"

echo
echo "Checking medmnist..."
python -c "import medmnist; print('medmnist version:', medmnist.__version__)"

echo
echo "Checking torch..."
python -c "import torch; print('torch version:', torch.__version__)"

echo
echo "GPU:"
nvidia-smi

echo
echo "Running training..."

python train_and_eval_pytorch.py \
    --data_flag chestmnist \
    --model_flag resnet50 \
    --num_epochs 1 \
    --gpu_ids 0 \
    --download \
    --run test

echo
echo "Done."