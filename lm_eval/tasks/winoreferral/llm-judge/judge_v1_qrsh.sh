#!/bin/bash
set -eo pipefail

mkdir -p logs

# workdir (qrsh keeps where started, but keep this robust)
cd "${SGE_O_WORKDIR:-$PWD}"

# --- environment ---
set +u
source ~/.bashrc
set -u
conda activate eval-env

# --- CUDA toolkit (needed for FlashInfer JIT on SCC) ---
module load cuda/12.8
export CUDA_HOME="/share/pkg.8/cuda/12.8/install"

# --- hf token ---
: "${HF_TOKEN:?HF_TOKEN is not set. Run: export HF_TOKEN=hf_... }"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# --- output ---
JSONL_OUTPUT="${LMEVAL_JUDGE_OUTPUT:-/projectnb/buinlp/afitab/lm-mental-health-eval/lm_eval/tasks/winoreferral/llm-judge/results/llama_judge_results.jsonl}"
OUTPUT_PATH="/projectnb/buinlp/afitab/lm-mental-health-eval/lm_eval/tasks/winoreferral/llm-judge/results/sample_judge_v1"
TASK_DIR="/projectnb/buinlp/afitab/lm-mental-health-eval/lm_eval/tasks/winoreferral/llm-judge"
MODEL="${JUDGE_MODEL:-ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4}"
SEEDS=(42 123 456 789 1000)
export LMEVAL_JUDGE_OUTPUT="$JSONL_OUTPUT"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CUDA_HOME=${CUDA_HOME}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

python - <<'PYTORCH'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}: {torch.cuda.get_device_name(i)}")
PYTORCH

echo "Model:  $MODEL"
echo "Task:   sample_judge_v1"
echo "Running seeds: ${SEEDS[*]}"
echo "JSONL:  $JSONL_OUTPUT"
echo "Output: $OUTPUT_PATH"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
  echo "=== Seed $SEED ==="
  export LMEVAL_JUDGE_SEED="$SEED"

  lm_eval     --model vllm     --model_args "pretrained=${MODEL},quantization=awq,tensor_parallel_size=4,dtype=auto,max_model_len=2048,trust_remote_code=True,enforce_eager=True,gpu_memory_utilization=0.85,disable_custom_all_reduce=True"     --tasks sample_judge_v1     --include_path "$TASK_DIR"     --batch_size 1     --output_path "$OUTPUT_PATH"     --seed "$SEED"     --log_samples
done

echo "Done. JSONL: $JSONL_OUTPUT"
echo "Done. lm_eval output: $OUTPUT_PATH"
