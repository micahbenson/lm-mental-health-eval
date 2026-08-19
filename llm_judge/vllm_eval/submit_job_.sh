#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively.
# Example usage:
#   Interactive: $ ./submit_job_one_split.sh config.yml
#   Batch job:   $ qsub ./submit_job_one_split.sh config.yml

# Make sure you're logged in to huggingface before running, if you're not sure
# you should login using "huggingface-cli login" before running this script

# Requesting resources from SCC
#$ -P ivc-ml
#$ -cwd
#$ -l h_rt=12:00:00
#$ -pe omp 8
#$ -l mem_per_core=4G
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=80G
#$ -m bea
#$ -j y
#$ -o logs/$JOB_ID.log

# Load conda module
module load miniconda

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate lm-mental-health
module load cuda
module load gcc

# Check that exactly one config file was passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 config_file.yml"
    exit 1
fi

# module load python3

export HF_HOME=/projectnb/ivc-ml/micahb/.cache/huggingface
export TRITON_CACHE_DIR=/projectnb/ivc-ml/micahb/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/projectnb/ivc-ml/micahb/.cache/inductor
export VLLM_CACHE_ROOT=/projectnb/ivc-ml/micahb/.cache/vllm
export XDG_CACHE_HOME=/projectnb/ivc-ml/micahb/.cache
# If this env var is set to 1, vLLM will skip the peer-to-peer check,
# and trust the driver's peer-to-peer capability report. Use this if using more than one gpu
# export VLLM_SKIP_P2P_CHECK=1

python -V

python /projectnb/ivc-ml/micahb/lm-mental-health-eval/llm_judge/vllm_eval/main.py config_file=$1

