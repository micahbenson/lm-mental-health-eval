#!/bin/bash -l

#$ -t 1-10
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -N mistral-bdi-p
#$ -j y
#$ -o logs/mistral_bdi_p.log

# Load conda module
module load miniconda

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate lm-mental-health-py311

# Export HF_HOME
export HF_HOME="/projectnb/ivc-ml/micahb/.cache/huggingface"

#extract system prompt
SYSTEM_PROMPT=$(sed -n "${SGE_TASK_ID}p" ./scc/bdi_persona/system_instructions.txt)

echo "Task $SGE_TASK_ID using prompt: $SYSTEM_PROMPT"

# Run evaluation
python lm_eval run \
    --config 'configs/mistral/mistral-7b.yaml' \
    --tasks 'bdi' \
    --system_instruction "$SYSTEM_PROMPT" \
    --include_path './winoreferral' \
    --output_path './results/persona/' \
    --seed 1