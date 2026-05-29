#!/bin/bash -l


conda activate /projectnb/vkolagrp/skowshik/conda_envs/position_paper

mkdir -p logs
set -e

for file in ../splits/anthology_splits/*.bib.gz; do
    echo "Running file ${file}"
    stem=$(basename "${file%.bib.gz}")
    out="outputs/gpt54_mini/anthology/${stem}_per_paper_judge.jsonl"
    if [ -f "$out" ]; then
        echo "Skipping $file (already done)"
        continue
    fi
    # echo $out
    python -u run_judge_async.py \
        --input-csv "$file" \
        --output outputs/gpt54_mini/anthology/ \
        --model gpt-5.4-mini \
        --mode infer \
        --save-every 60 \
        --concurrency 15 \
        &> logs/${stem}.log
    # break
done