#Set to your HF_HOME
export HF_HOME="/projectnb/ivc-ml/micahb/.cache/huggingface"

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks winoreferral \
    --apply_chat_template True \
    --log_samples \
    --output_path ./results/ \
    --device cuda:0 \
    --batch_size "auto"


python lm_eval run --config 'configs/llama/llama-3.1-8b.yaml' --tasks winoreferral --seed 1

python lm_eval run --config 'configs/gemma/gemma-3-12b.yaml' --tasks winoreferral --seed 1

#Need a bigger gpy to run gpt oss!!
python lm_eval run --config 'configs/gpt/gpt-oss-20b.yaml' --tasks winoreferral --seed 1

python lm_eval run --config 'configs/mistral/ministral-3-8b.yaml' --tasks winoreferral --seed 1

python lm_eval run --config 'configs/qwen/qwen3-8b.yaml' --tasks winoreferral --seed 1

python lm_eval run --config 'configs/olmo/olmo-3-7b.yaml' --tasks winoreferral --seed 1
