#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively.
# Example usage:
#   Interactive: $ ./submit_job_one_split.sh config.yml
#   Batch job:   $ qsub ./submit_job_one_split.sh config.yml

# Make sure you're logged in to huggingface before running, if you're not sure
# you should login using "huggingface-cli login" before running this script

# Requesting resources from SCC
#$ -P vkolagrp
#$ -l h_rt=2:00:00
#$ -pe omp 8
#$ -l mem_per_core=2G
#$ -l gpus=1
#$ -l gpu_c=9 # GPU capability, must be at least 8 for this project
#$ -m bea
#$ -l gpu_type=H200
#$ -e logs/$JOB_ID.stderr
#$ -o logs/$JOB_ID.stdout

# Check that exactly one config file was passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 config_file.yml"
    exit 1
fi

# module load python3

# conda init
conda activate /projectnb/vkolagrp/skowshik/conda_envs/position_paper
module load cuda
module load gcc

export HF_HOME=/projectnb/vkolagrp/skowshik/.cache
export VLLM_CACHE_ROOT=/projectnb/vkolagrp/skowshik/.cache
export UV_CACHE_DIR=/projectnb/vkolagrp/skowshik/.cache
export FLASHINFER_WORKSPACE_BASE=/projectnb/vkolagrp/skowshik/.cache

# If this env var is set to 1, vLLM will skip the peer-to-peer check,
# and trust the driver's peer-to-peer capability report. Use this if using more than one gpu
# export VLLM_SKIP_P2P_CHECK=1

python -V

python main.py config_file=$1

