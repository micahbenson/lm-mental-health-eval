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