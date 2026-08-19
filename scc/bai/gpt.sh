#!/bin/bash -l

#$ -t 1
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -N gpt-bai
#$ -j y
#$ -o logs/gpt_bai.log

# Load conda module
module load miniconda

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate lm-mental-health-py311

# Export HF_HOME
export HF_HOME="/projectnb/ivc-ml/micahb/.cache/huggingface"

# Run evaluation
python lm_eval run \
    --config 'configs/gpt/gpt-oss-20b.yaml' \
    --tasks 'bai' \
    --include_path './winoreferral' \
    --seed $SGE_TASK_ID