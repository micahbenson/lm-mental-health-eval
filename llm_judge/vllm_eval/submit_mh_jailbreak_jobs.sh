#!/bin/bash -l

# Submits all 20 chunks as a single SGE array job, using one shared config.
# Usage:
#   Batch job: $ qsub ./submit_job_array.sh

# Make sure you're logged in to huggingface before running:
#   huggingface-cli login

# Requesting resources from SCC
#$ -P ivc-ml
#$ -cwd
#$ -l h_rt=12:00:00
#$ -pe omp 8
#$ -l mem_per_core=4G
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=80G
#$ -t 1-20
#$ -m bea
#$ -j y
#$ -o logs/$JOB_ID.$TASK_ID.log

# Load conda module
module load miniconda

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate lm-mental-health
module load cuda
module load gcc

export HF_HOME=/projectnb/ivc-ml/micahb/.cache/huggingface
export TRITON_CACHE_DIR=/projectnb/ivc-ml/micahb/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/projectnb/ivc-ml/micahb/.cache/inductor
export VLLM_CACHE_ROOT=/projectnb/ivc-ml/micahb/.cache/vllm
export XDG_CACHE_HOME=/projectnb/ivc-ml/micahb/.cache
# export VLLM_SKIP_P2P_CHECK=1

python -V

# SGE_TASK_ID runs 1-20; chunk files are chunk_0..chunk_19
CHUNK_IDX=$((SGE_TASK_ID - 1))
DATA_FILE="/projectnb/ivc-ml/micahb/lm-mental-health-eval/llm_judge/data/mh_jailbreak_chunks/chunk_${CHUNK_IDX}.csv"
echo "Task $SGE_TASK_ID -> $DATA_FILE"

python /projectnb/ivc-ml/micahb/lm-mental-health-eval/llm_judge/vllm_eval/main.py \
    config_file=/projectnb/ivc-ml/micahb/lm-mental-health-eval/llm_judge/vllm_eval/configs/mental_health/config_mh_jailbreak_qwen36_27B.yml \
    Dataset.data_file=$DATA_FILE
