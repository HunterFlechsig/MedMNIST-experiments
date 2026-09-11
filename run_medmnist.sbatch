#!/bin/bash
#SBATCH --job-name=medmnist_r50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

source activate image_pytorch

# Check that a dataset name was provided
DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: sbatch run_medmnist.sbatch <dataset>"
    echo "Example: sbatch run_medmnist.sbatch chestmnist"
    exit 1
fi

cd $HOME/MedMNIST-experiments/MedMNIST2D

echo "========================================="
echo "Dataset      : ${DATASET}"
echo "Run          : ${SLURM_ARRAY_TASK_ID}"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Hostname     : $(hostname)"
echo "========================================="

nvidia-smi

python train_and_eval_pytorch.py \
    --data_flag ${DATASET} \
    --model_flag resnet50 \
    --gpu_ids 0 \
    --download \
    --num_epochs 50 \
    --run run${SLURM_ARRAY_TASK_ID}

echo "Finished ${DATASET} run ${SLURM_ARRAY_TASK_ID}"