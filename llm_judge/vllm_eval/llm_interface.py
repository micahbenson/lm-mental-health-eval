from omegaconf import OmegaConf
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class LLMWrapper:
    """Thin wrapper around vLLM for chat completion."""

    def __init__(self, config):
        llm_cfg = dict(OmegaConf.to_container(config.LLM, resolve=True))
        enable_lora = llm_cfg.pop("enable_lora", False)
        lora_path = llm_cfg.pop("lora_path", None)

        self.llm = LLM(**llm_cfg)
        self.sampling_params = SamplingParams(
            **OmegaConf.to_container(config.sampling_params, resolve=True),
        )

        if enable_lora:
            if not lora_path:
                raise ValueError("LLM.enable_lora is true but lora_path is not set")
            self.lora_request = LoRARequest("adapter", 1, lora_path)
        else:
            self.lora_request = None

    def generate(self, messages, enable_thinking=None, enable_thinking_argument=None):
        """Generate completions for a batch of chat conversations.

        Args:
            messages: list of conversations; each conversation is a list of
                {"role": ..., "content": ...} dicts (one per paperxaxis prompt).
            enable_thinking: passed to the model chat template when not None.
        """
        kwargs = dict(
            messages=messages,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request,
        )
        if enable_thinking is not None:
            kwargs["chat_template_kwargs"] = {enable_thinking_argument: enable_thinking}

        return self.llm.chat(**kwargs)
