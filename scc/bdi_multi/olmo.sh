#!/bin/bash -l

#$ -t 1
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -N olmo-d-multi
#$ -j y
#$ -o logs/olmo_d_multi1.log

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
    --config 'configs/olmo/olmo-3-7b.yaml' \
    --tasks 'bdi_multi_olmo' \
    --include_path './winoreferral' \
    --seed $SGE_TASK_ID