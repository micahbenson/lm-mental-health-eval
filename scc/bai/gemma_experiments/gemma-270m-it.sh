#!/bin/bash -l

#$ -t 1
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -N g270mit-bai
#$ -j y
#$ -o logs/g270mit_bai.log

# Load conda module
module load miniconda

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate lm-mental-health

# Export HF_HOME
export HF_HOME="/projectnb/ivc-ml/micahb/.cache/huggingface"

# Run evaluation
python lm_eval run \
    --config 'configs/gemma/gemma-3-270m-it.yaml' \
    --tasks 'bai' \
    --include_path './winoreferral' \
    --seed $SGE_TASK_ID