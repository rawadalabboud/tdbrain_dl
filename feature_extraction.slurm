#!/bin/bash
#SBATCH --job-name=EEGFeatureExtraction
#SBATCH --output=logs/features_%A_%a.out
#SBATCH --error=logs/features_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-120  # 121 subjects
#SBATCH --partition=gpu-volta

module load miniconda
module load CUDA
source activate my_env  # Replace with your Conda environment

# Define dataset location
DATASET_PATH="/network/iss/home/rawad.alabboud/TDBRAIN-dataset-derivatives/TD_BRAIN_CODE/wavelet_dataset_3D.npy"
OUTPUT_PATH="/network/iss/home/rawad.alabboud/features/"

# Extract features for each subject
python feature_extraction.py --subject $SLURM_ARRAY_TASK_ID --dataset $DATASET_PATH --output_path $OUTPUT_PATH
