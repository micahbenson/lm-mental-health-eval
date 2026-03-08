#!/bin/bash -l

#$ -t 1
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=80G
#$ -N gemma-bdi
#$ -j y
#$ -o logs/gemma_bdi.log

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
    --config 'configs/gemma/gemma-3-12b.yaml' \
    --tasks 'bdi' \
    --include_path './winoreferral' \
    --seed $SGE_TASK_ID