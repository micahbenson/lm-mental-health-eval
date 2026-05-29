# ============================================================================
# CONFIGURATION - Edit these values to apply to all generated config files
# ============================================================================

from pathlib import Path

# Directory containing anthology_split_*.bib.gz files
SPLITS_DIR = (
    Path(__file__).resolve().parents[4] / "splits" / "anthology_splits"
)

# Path prefix for data_file, relative to vllm_eval/ (where main.py runs)
DATA_FILE_PREFIX = "../splits/anthology_splits"

MODEL = "Qwen/Qwen3.6-27B"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 8096
SEED = 1

VENUES = "acl,naacl,emnlp"
INCLUDE_FINDINGS = True
SAMPLE_PER_YEAR = None  # set an int to subsample per year

SAMPLING_PARAMS = {
    "n": 1,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 4096,
}

JUDGE = {
    "binary_prompt": False,
    "max_papers": None,
    "skip": False,
    "enable_thinking": False,
    "enable_thinking_argument": "enable_thinking",
    "use_system_prompt": False,
    "use_dspy_prompts": True
}

RESULT_DIR = "../results_abstract/qwen36_27B/anthology"

# ============================================================================
# Template
# ============================================================================

CONFIG_TEMPLATE = """Dataset:
  data_file: "{data_file}"
  venues: "{venues}"
  include_findings: {include_findings}
  sample_per_year: {sample_per_year}  # set an int to subsample per year

# passed to vllm.LLM 
LLM:
  model: "{model}" # can be a path
  tensor_parallel_size: {tensor_parallel_size} # Number of GPUs used. Passed to vllm.LLM.tensor_parallel_size. 
  gpu_memory_utilization: {gpu_memory_utilization} # Fraction of gpu memory reserved for model + attention
  max_model_len: {max_model_len}
  # distributed_executor_backend: 'mp'
  seed: {seed} # seed to initialize RNG for sampling
  enable_lora: false

# passed to vllm.SamplingParams
# With thinking
# sampling_params:
#   n: 1 # Number of sequences to generate per prompt
#   temperature: 1.0  # Temperature for generation
#   top_p: 0.95  # Top-p for generation
#   top_k: 20
#   min_p: 0.0
#   presence_penalty: 1.5
#   repetition_penalty: 1.0
#   max_tokens: 20000

# Non-thinking
sampling_params:
  n: {sampling_n} # Number of sequences to generate per prompt
  temperature: {sampling_temperature}  # Temperature for generation
  top_p: {sampling_top_p}  # Top-p for generation
  top_k: {sampling_top_k}
  min_p: {sampling_min_p}
  presence_penalty: {sampling_presence_penalty}
  repetition_penalty: {sampling_repetition_penalty}
  max_tokens: {sampling_max_tokens}

judge:
  binary_prompt: {judge_binary_prompt}
  max_papers: {judge_max_papers}  # set e.g. 100 
  skip: {judge_skip}
  enable_thinking: {judge_enable_thinking}
  enable_thinking_argument: "{judge_enable_thinking_argument}"
  use_system_prompt: {judge_use_system_prompt}
  use_dspy_prompts: {judge_use_dspy_prompts}

output:
  result_dir: "{result_dir}"
  suffix: "_dspy"
"""


def _yaml_bool(value: bool) -> str:
    return "true" if value else "false"


def _yaml_null(value) -> str:
    return "null" if value is None else str(value)


def format_config(split_filename: str) -> str:
    data_file = f"{DATA_FILE_PREFIX}/{split_filename}"
    return CONFIG_TEMPLATE.format(
        data_file=data_file,
        venues=VENUES,
        include_findings=_yaml_bool(INCLUDE_FINDINGS),
        sample_per_year=_yaml_null(SAMPLE_PER_YEAR),
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        seed=SEED,
        sampling_n=SAMPLING_PARAMS["n"],
        sampling_temperature=SAMPLING_PARAMS["temperature"],
        sampling_top_p=SAMPLING_PARAMS["top_p"],
        sampling_top_k=SAMPLING_PARAMS["top_k"],
        sampling_min_p=SAMPLING_PARAMS["min_p"],
        sampling_presence_penalty=SAMPLING_PARAMS["presence_penalty"],
        sampling_repetition_penalty=SAMPLING_PARAMS["repetition_penalty"],
        sampling_max_tokens=SAMPLING_PARAMS["max_tokens"],
        judge_binary_prompt=_yaml_bool(JUDGE["binary_prompt"]),
        judge_max_papers=_yaml_null(JUDGE["max_papers"]),
        judge_skip=_yaml_bool(JUDGE["skip"]),
        judge_enable_thinking=_yaml_bool(JUDGE["enable_thinking"]),
        judge_enable_thinking_argument=JUDGE["enable_thinking_argument"],
        judge_use_system_prompt=_yaml_bool(JUDGE["use_system_prompt"]),
        judge_use_dspy_prompts=_yaml_bool(JUDGE["use_dspy_prompts"]),
        result_dir=RESULT_DIR,
    )


# ============================================================================
# Generate configuration files
# ============================================================================

if __name__ == "__main__":
    if not SPLITS_DIR.is_dir():
        raise SystemExit(f"Splits directory not found: {SPLITS_DIR}")

    split_files = sorted(SPLITS_DIR.glob("*.bib.gz"))
    if not split_files:
        raise SystemExit(f"No .bib.gz files found in {SPLITS_DIR}")

    output_dir = Path(__file__).parent
    for split_path in split_files:
        # anthology_split_00.bib.gz -> config_anthology_split_00.yml
        stem = split_path.name.removesuffix(".bib.gz")
        config_name = f"config_{stem}.yml"
        config_path = output_dir / config_name

        config_path.write_text(format_config(split_path.name))
        print(f"Generated: {config_path}")

    print(f"Done. Generated {len(split_files)} config files in {output_dir}")
