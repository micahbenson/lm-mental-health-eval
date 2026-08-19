#!/bin/bash -l


conda activate /projectnb/vkolagrp/skowshik/conda_envs/position_paper

python -u run_judge_async.py \
        --input-csv "../splits/acl_ocl_papers_1979_2017_final.csv" \
        --output outputs/gpt54_mini/ \
        --model gpt-5.4-mini \
        --mode infer \
        --save-every 60 \
        --concurrency 15 \
        &> logs/acl_ocl_papers_1979_2017_final.log