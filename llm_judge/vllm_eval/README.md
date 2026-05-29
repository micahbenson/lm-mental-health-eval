# vLLM Abstract Judge Evaluation

This folder contains batch inference pipeline for scoring paper abstracts on six research-position axes using [vLLM](https://github.com/vllm-project/vllm). Loads papers from BibTeX or CSV, builds judge prompts, runs a single batched `llm.chat()` call per job, and writes per-paper CSV scores.

<!-- ## Prerequisites

- Linux with NVIDIA GPU (CUDA capability ≥ 8; H200 used on BU SCC)
- Hugging Face access for gated models (`huggingface-cli login`)
- Dataset files under `abstract_analysis/splits/` (paths in configs are relative to this folder) -->

---

## 1. Install vLLM

### Create environment

```bash
conda create -n position_paper python=3.12 -y
conda activate position_paper
```

### vLLM and runtime deps

```bash
pip install uv
uv pip install vllm --torch-backend=auto
pip install omegaconf pandas numpy bibtexparser
```

### Load cuda and gcc

On SCC, load modules before running jobs:

```bash
module load cuda
module load gcc
```
---

## 2. Configure a run (`configs/`)

Create or edit a YAML config. Config paths in `data_file` and `result_dir` are relative to `vllm_eval/` (where `main.py` runs).

### Config layout

| Section | Purpose |
|--------|---------|
| `Dataset` | Input bib/CSV, venue filter (bib only), optional subsampling |
| `LLM` | Arguments passed to `vllm.LLM()` (model, GPUs, memory, model-specific flags) |
| `sampling_params` | Arguments passed to `vllm.SamplingParams()` |
| `judge` | Prompt mode, thinking, limits |
| `output` | Result directory and output filename suffix |

### Example: CSV dataset (`configs/best_papers_acl/config_qwen36_27B.yml`)

```yaml
Dataset:
  data_file: "../splits/best_papers_acl_full.csv"
  venues: "acl,naacl,emnlp,eacl,aacl,tacl"   # used for .bib only
  include_findings: true
  sample_per_year: null

LLM:
  model: "Qwen/Qwen3.6-27B"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 8096
  seed: 1
  enable_lora: false

sampling_params:
  n: 1
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  max_tokens: 4096

judge:
  binary_prompt: false
  max_papers: null          
  skip: false
  enable_thinking: false
  enable_thinking_argument: "enable_thinking"
  use_system_prompt: false
  use_dspy_prompts: true

output:
  result_dir: "../results_abstract/qwen36_27B/best_papers_acl"
  suffix: "_dspy"
```

### Example: Anthology splits

Each split has its own config under `configs/anthology/qwen36_27B/`. Regenerate all split configs from one template:

```bash
cd configs/anthology/qwen36_27B
python generate_configs.py
```

Edit the `CONFIGURATION` block at the top of `generate_configs.py` (model, venues, sampling, judge flags, `RESULT_DIR`) before regenerating.

### `judge` flags

| Key | Description |
|-----|-------------|
| `use_dspy_prompts` | `true`: DSPy-style prompts from `JUDGE_SYSTEM_DSPY` in `prompts.py`. `false`: `JUDGE_AXES` + `JUDGE_SYSTEM` templates. |
| `use_system_prompt` | Put axis instructions in a `system` message; title/abstract in `user`. |
| `binary_prompt` | 0/1 ratings instead of 0/1/2. |
| `enable_thinking` | Passed to the model chat template (e.g. Qwen thinking). Model-specific. |
| `enable_thinking_argument` | Template kwarg name (e.g. `"enable_thinking"` or `"reasoning_effort"` for Mistral). |
| `max_papers` | Truncate dataset |
| `skip` | Exit after load without inference. |

---

## 3. Set prompts (`prompts.py`)

**`prompts.py`** (the module imported by `main.py` and `utils.py`) has all the prompts being used.

---

## 4. Run evaluation

### Interactive (from `vllm_eval/`)

```bash
conda activate /path/to/position_paper   # or your env
module load cuda gcc
export HF_HOME=/path/to/.cache

cd abstract_analysis/vllm_eval
python main.py config_file=configs/best_papers_acl/config_qwen36_27B.yml
```

### SCC batch (single config)

```bash
# Interactive on a GPU node:
./submit_job_one_split.sh configs/best_papers_acl/config_qwen36_27B.yml

# Or submit to the queue:
qsub ./submit_job_one_split.sh configs/best_papers_acl/config_qwen36_27B.yml
```

### Submit many configs

```bash
./submit_all.sh configs/anthology/qwen36_27B
```

Submits every `*.yml` in that directory via `qsub submit_job_one_split.sh <config>`.

---

## 5. Results

The results are written to `{result_dir}/{dataset_stem}_per_paper_judge{suffix}.csv` with columns `ID`, `year`, and one column per axis.

Logs from batch jobs are written to `logs/$JOB_ID.stdout` and `logs/$JOB_ID.stderr`.

---

## File structure

```
vllm_eval/
├── README.md                 
├── main.py                   # Entry: load config, dataset, run judge
├── llm_interface.py          # vLLM LLM + SamplingParams wrapper
├── load_dataset.py           # Parse .bib/.bib.gz (venue filter) or .csv
├── prompts.py                # Judge prompts and templates
├── utils.py                  # Prompt building, inference, CSV I/O
├── submit_job_one_split.sh   # SCC: one config per job
├── submit_all.sh             # SCC: qsub all YAMLs in a directory
├── logs/                     # Batch job stdout/stderr
└── configs/
    ├── best_papers_acl/
    │   ├── config_qwen36_27B.yml
    │   └── config_mistral_small_4_119B.yml
    └── anthology/
        └── qwen36_27B/
            ├── generate_configs.py    # Batch-generate split configs
            ├── config_anthology_split_00.yml
            ├── ...
            └── config_anthology_split_19.yml
```

**Related paths** (parent `abstract_analysis/`):

```
abstract_analysis/
├── splits/
│   ├── anthology_splits/          # anthology_split_*.bib.gz
│   └── best_papers_acl_full.csv
└── results_abstract/              # Default output root (from configs)
    └── <model>/<dataset>/
        └── *_per_paper_judge*.csv
```

---

## Output format

Per-paper CSV example: `best_papers_acl_full_per_paper_judge_dspy.csv`

| Column | Description |
|--------|-------------|
| `ID` | Paper bibkey or CSV ID |
| `year` | Publication year |
| `language_specific` … `societal_impact` | Integer 0–2 (or 0–1 if `binary_prompt: true`); `NaN` if unparsed |

