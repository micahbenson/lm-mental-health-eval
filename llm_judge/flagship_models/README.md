# Abstract Judge Evaluation using API

This folder contains a inference pipeline for scoring paper abstracts on six research-position axes using commercial APIs: [OpenAI](https://platform.openai.com/) (`responses` API) and [Anthropic](https://docs.anthropic.com/) (`messages` API). Loads papers from BibTeX or CSV, builds one judge prompt per paper per axis, runs inference (or token counting only), and writes JSONL plus a wide per-paper CSV.

For local GPU inference with [vLLM](https://github.com/vllm-project/vllm), see [`../vllm_eval/README.md`](../vllm_eval/README.md).

---

## 1. Install dependencies

### Create environment

```bash
conda create -n position_paper python=3.12 -y
conda activate position_paper
```

### API clients and dataset parsing

```bash
pip install openai anthropic pandas bibtexparser
```

### API keys

Set keys for the provider you use (provider is inferred from the model name; see below):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

---

## 2. Configure a run (CLI flags)

Paths in `--input-csv` and `--output` are relative to `flagship_models/`.

### CLI layout

| Flag | Purpose |
|------|---------|
| `--input-csv` | Input `.csv`, `.bib`, or `.bib.gz` (required) |
| `--output` | Output directory (default: `outputs/judge_results.csv` stem is ignored; files are named from the input stem) |
| `--model` | Model name; provider auto-detected from name |
| `--mode` | `infer` (run judge) or `count` (input tokens only; does not use api credits) |
| `--venues` | Comma-separated venue filter for `.bib` inputs (default: ACL anthology venues) |
| `--include-findings` | Include Findings-track papers when filtering `.bib` |
| `--limit` | Max papers to load |
| `--max-calls` | Max prompt calls (papers × 6 axes) |
| `--concurrency` | Max in-flight requests (`run_judge_async.py` only; default: 10) |
| `--save-every` | Checkpoint JSONL / wide CSV every N completed calls (default: 10) |
| `--sleep` | Seconds to sleep between calls (`run_judge.py` only; default: 0.1) |
| `--max-tokens` | `max_tokens` for Anthropic inference (ignored by OpenAI) |

### Model / provider mapping

| Model name pattern | Provider | API |
|--------------------|----------|-----|
| `claude-*` (e.g. `claude-haiku-4-5`, `claude-sonnet-4-6`) | Anthropic | `messages.create` |
| Everything else (e.g. `gpt-5.4-mini`) | OpenAI | `responses.create` |

---

## 3. Set prompts (`prompts.py`)

**`prompts.py`** defines the judge axes and templates used by both runners:

- **`JUDGE_AXES`**: six `(axis_key, axis_desc, evaluation_guidelines)` tuples
- **`JUDGE_SYSTEM`**: system instructions with `{axis_desc}` and `{evaluation_guidelines}`
- **`PROMPT_TEMPLATE`**: user message with `{title}` and `{abstract}`

---

## 4. Run evaluation

### Recommended: async parallel (`run_judge_async.py`)

Requests run concurrently up to `--concurrency`, with periodic checkpoints.

### Example: best-papers CSV

```bash
cd abstract_analysis/flagship_models
conda activate position_paper

python run_judge_async.py \
  --input-csv "../splits/best_papers_acl_full.csv" \
  --model "gpt-5.4-mini" \
  --mode infer \
  --output "outputs/gpt54_mini/best_papers_acl_full/final" \
  --save-every 30 \
  --concurrency 15
```

### Example: anthology split (BibTeX)

```bash
python run_judge_async.py \
  --input-csv "../splits/anthology_splits/anthology_split_00.bib.gz" \
  --model "gpt-5.4-mini" \
  --mode infer \
  --output "outputs/gpt54_mini/anthology/" \
  --include-findings \
  --save-every 60 \
  --concurrency 15
```

### Token counting only

Use `--mode count` to estimate input tokens without generation:

```bash
python run_judge_async.py \
  --input-csv "../splits/best_papers_acl_full.csv" \
  --model "gpt-5.4-mini" \
  --mode count \
  --output "outputs/gpt54_mini/best_papers_acl_full/count" \
  --concurrency 15
```


### Batch all anthology splits

`run_all.sh` loops over `../splits/anthology_splits/*.bib.gz`, skips splits whose JSONL output already exists, and logs each split to `logs/<stem>.log`:

```bash
./run_all.sh
```

Edit the loop in `run_all.sh` to change model, `--mode`, `--output`, or concurrency.

---

## 5. Results

For input stem `{dataset_stem}` (e.g. `best_papers_acl_full` or `anthology_split_00`), files are written under `{output}/`:

| File | Description |
|------|-------------|
| `{dataset_stem}_per_paper_judge.jsonl` | One JSON object per axis call (long format, includes prompts, raw output, tokens, errors) |
| `{dataset_stem}_per_paper_judge_wide.csv` | One row per paper with axis rating columns |

Batch logs from `run_all.sh` go to `logs/<stem>.log`.

---

## File structure

```
flagship_models/
├── README.md
├── run_judge_async.py      # Recommended: parallel API inference
├── run_judge.py            # Sequential API inference
├── load_dataset.py         # Parse .bib/.bib.gz (venue filter) or .csv
├── prompts.py              # Judge axes and prompt templates
├── run_all.sh              # Loop anthology splits with run_judge_async.py
├── run_inference.sh        # Example batch driver (edit before use)
├── logs/                   # Per-split stdout from run_all.sh
└── outputs/
    └── <model>/
        ├── best_papers_acl_full/
        └── anthology/
            └── anthology_split_*_per_paper_judge.{jsonl,wide.csv}
```

**Related paths** (parent `abstract_analysis/`):

```
abstract_analysis/
├── splits/
│   ├── anthology_splits/          # anthology_split_*.bib.gz
│   └── best_papers_acl_full.csv
└── results_abstract/              # vLLM outputs (see vllm_eval/)
```

---

## Output format

### Wide CSV (`*_per_paper_judge_wide.csv`)

| Column | Description |
|--------|-------------|
| `ID` | Paper bibkey or CSV ID |
| `Title` | Paper title |
| `Abstract` | Paper abstract |
| `year` | Publication year |
| `benchmark_sota_framing` … `societal_impact` | Integer 0–2; empty if unparsed |

### JSONL (`*_per_paper_judge.jsonl`)

Each line is one axis-level call with fields such as `paper_key`, `axis_key`, `model_output`, `rating`, `input_tokens`, `output_tokens`, `error`, plus the prompt fields used for that call.
