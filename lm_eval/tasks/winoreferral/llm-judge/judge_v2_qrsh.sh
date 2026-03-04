#!/bin/bash
# Run the per-trait (3 calls per example) LLM judge evaluation.
# Uses AWQ 4-bit quantized Llama-3.3-70B (~35 GB weights) so it fits
# on 4x 24 GB GPUs (Quadro RTX 6000 / A5000, etc.).
#
# Usage:
#   qrsh -l gpus=4 -l gpu_c=6.0 -l gpu_memory=24G -pe omp 8
#   bash judge_v2_qrsh.sh
#
# 300 examples x 3 traits = 900 model calls per seed.
set -eo pipefail

mkdir -p logs

cd "${SGE_O_WORKDIR:-$PWD}"

# --- environment ---
set +u
source ~/.bashrc
set -u
conda activate eval-env

# --- CUDA toolkit (needed by FlashInfer JIT) ---
module load cuda/12.8
export CUDA_HOME="/share/pkg.8/cuda/12.8/install"

# --- hf token ---
: "${HF_TOKEN:?HF_TOKEN is not set. Run: export HF_TOKEN=hf_... }"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# --- paths ---
TASK_DIR="/projectnb/buinlp/afitab/lm-mental-health-eval/lm_eval/tasks/winoreferral/llm-judge"
JSONL_OUTPUT="${LMEVAL_JUDGE_OUTPUT:-${TASK_DIR}/results/judge_per_trait_results.jsonl}"
OUTPUT_PATH="${TASK_DIR}/results/sample_judge_v2"

# --- model ---
MODEL="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"

SEEDS=(42 123 456 789 1000)
export LMEVAL_JUDGE_OUTPUT="$JSONL_OUTPUT"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CUDA_HOME=${CUDA_HOME}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}: {torch.cuda.get_device_name(i)}")
PY

echo "=========================================="
echo "Model:  $MODEL (AWQ INT4)"
echo "Task:   sample_judge_v2"
echo "Seeds:  ${SEEDS[*]}"
echo "JSONL:  $JSONL_OUTPUT"
echo "Output: $OUTPUT_PATH"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  echo "=== Seed $SEED ==="
  export LMEVAL_JUDGE_SEED="$SEED"

  lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},quantization=awq,tensor_parallel_size=4,dtype=auto,max_model_len=2048,trust_remote_code=True,enforce_eager=True,gpu_memory_utilization=0.85" \
    --tasks sample_judge_v2 \
    --include_path "$TASK_DIR" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --seed "$SEED" \
    --log_samples
done

echo "Done. JSONL: $JSONL_OUTPUT"
echo "Done. lm_eval output: $OUTPUT_PATH"
