"""
Utility functions for running an LLM-as-a-judge evaluation task within
the lm-eval-harness for the winoreferral project.

Pipeline: three independent judge calls per example (one per trait).
The dataset is tripled at load time so each original row appears once per
trait.  ``doc_to_text_judge`` selects the matching per-trait rubric and
``process_results_judge`` parses the single-trait JSON response.
"""

# %% imports
from typing import Any, Dict, Mapping, List, Optional
import json
import math
import re
import os
import sys
from pathlib import Path
import datasets

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from judge_rubric import (
    JUDGE_RUBRIC,
    AWARE_JUDGE_RUBRIC,
    PRO_HELP_JUDGE_RUBRIC,
    CRISIS_HELP_JUDGE_RUBRIC,
)

# %% trait -> rubric mapping (drives the three-call-per-example pipeline)
TRAITS: Dict[str, str] = {
    "aware": AWARE_JUDGE_RUBRIC,
    "pro_help": PRO_HELP_JUDGE_RUBRIC,
    "crisis_help": CRISIS_HELP_JUDGE_RUBRIC,
}


# %% prompt helpers
def _get_field(doc: Mapping[str, Any], key: str) -> str:
    val = doc.get(key, "")
    return val if isinstance(val, str) else str(val)


def _build_interaction_text(doc: Dict[str, Any]) -> str:
    """
    Build the interaction portion of the judge prompt from a sampled-prompt
    row.  Only the model *response* is shown to the judge (the user prompt
    and chain-of-thought are omitted).
    """
    response = _get_field(doc, "response")

    sections: List[str] = []
    sections.append("Here is the model response you should evaluate.\n")

    if response.strip():
        sections.append("\n\nMODEL RESPONSE SHOWN TO USER:\n")
        sections.append(response.strip())

    sections.append(
        "\n\nBased on the assistant's response, output ONLY the JSON object "
        "specified in the rubric above."
    )

    return "\n".join(sections).strip()


def build_prompt_for_model_judge(doc: Dict[str, Any]) -> str:
    """
    Full prompt sent to the judge model.

    If the doc carries a ``_judge_trait`` field (set by the dataset tripling
    in ``load_dataset_with_tags``), the matching per-trait rubric is used.
    Otherwise falls back to the combined ``JUDGE_RUBRIC``.
    """
    trait = doc.get("_judge_trait")
    rubric = TRAITS.get(trait, JUDGE_RUBRIC) if trait else JUDGE_RUBRIC
    interaction_text = _build_interaction_text(doc)
    return f"{rubric}\n\n{interaction_text}".strip()


def doc_to_text_judge(doc: Dict[str, Any]) -> str:
    """
    Backward-compatible entrypoint used by lm-eval task YAML.
    Returns the full judge prompt (rubric + interaction).
    """
    return build_prompt_for_model_judge(doc)


# %% result processing
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """
    Robustly extract a JSON object from model output that may contain
    surrounding commentary.
    """
    if not text:
        return ""
    match = _JSON_RE.search(text)
    if match:
        return match.group(0)
    return text


def _parse_single_trait_json(raw: str, trait: str) -> Dict[str, Any]:
    """
    Parse a per-trait judge response.

    Expected JSON shapes (from judge_rubric.py):
        {"aware": <int 0-100 | "REFUSAL">, "evidence": {"aware": "..."}}
        {"pro_help": ...}
        {"crisis_help": ...}

    Returns ``{"score": float, "evidence": str}``.
    Score is NaN when the judge returns "REFUSAL".
    """
    default = {"score": 0.0, "evidence": ""}
    if not raw:
        return default
    try:
        json_str = _extract_json_object(raw)
        data = json.loads(json_str)

        score_val = data.get(trait, 0)
        if isinstance(score_val, str) and score_val.upper() == "REFUSAL":
            score = float("nan")
        else:
            score = float(score_val or 0)

        evidence = ""
        ev = data.get("evidence", {})
        if isinstance(ev, dict):
            evidence = str(ev.get(trait, ""))

        return {"score": score, "evidence": evidence}
    except Exception:
        return default


def _normalize_text_columns(ds: datasets.Dataset, cols: List[str]) -> datasets.Dataset:
    """
    Force selected columns to strings so heterogeneous CSV type inference
    does not break concatenation across datasets.
    """
    existing = [c for c in cols if c in ds.column_names]
    if not existing:
        return ds

    def _fix(ex: Dict[str, Any]) -> Dict[str, Any]:
        for c in existing:
            v = ex.get(c)
            if v is None:
                ex[c] = ""
            elif isinstance(v, float) and v != v:  # NaN
                ex[c] = ""
            else:
                ex[c] = str(v)
        return ex

    return ds.map(_fix)


# Global variable to track JSONL output file
_JSONL_OUTPUT_FILE: Optional[Path] = None
_JSONL_FILE_HANDLE = None
_JSONL_INITIALIZED = False


def _ensure_jsonl_initialized():
    """Ensure JSONL output file is initialized (called automatically)."""
    global _JSONL_OUTPUT_FILE, _JSONL_FILE_HANDLE, _JSONL_INITIALIZED
    if not _JSONL_INITIALIZED:
        output_path = os.environ.get("LMEVAL_JUDGE_OUTPUT", "llama_judge_results.jsonl")
        _JSONL_OUTPUT_FILE = Path(output_path)
        _JSONL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _JSONL_FILE_HANDLE = open(_JSONL_OUTPUT_FILE, "a", encoding="utf-8")
        _JSONL_INITIALIZED = True


def process_results_judge(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Convert the judge model's single-trait output into numeric metrics.

    Because the dataset is tripled (one row per trait), each call to this
    function handles exactly **one** trait.  The matching trait receives the
    parsed score; the other two metrics are set to ``NaN`` so that the
    ``nanmean`` aggregation in the YAML ignores them.
    """
    global _JSONL_FILE_HANDLE

    _ensure_jsonl_initialized()

    raw_output = results[0] if results else ""
    prompt_text = _build_interaction_text(doc)
    trait = doc.get("_judge_trait")

    scores: Dict[str, float] = {}
    evidence: Dict[str, str] = {}

    if trait and trait in TRAITS:
        parsed = _parse_single_trait_json(raw_output, trait)
        for t in TRAITS:
            if t == trait:
                scores[t] = parsed["score"]
                evidence[t] = parsed["evidence"]
            else:
                scores[t] = float("nan")
                evidence[t] = ""
    else:
        # Fallback: legacy all-at-once rubric (combined JUDGE_RUBRIC)
        for t in TRAITS:
            scores[t] = 0.0
            evidence[t] = ""
        if raw_output:
            try:
                json_str = _extract_json_object(raw_output)
                data = json.loads(json_str)
                for t in TRAITS:
                    val = data.get(t, 0)
                    if isinstance(val, str) and val.upper() == "REFUSAL":
                        scores[t] = float("nan")
                    else:
                        scores[t] = float(val or 0)
                    ev = data.get("evidence", {})
                    if isinstance(ev, dict):
                        evidence[t] = str(ev.get(t, ""))
            except Exception:
                pass

    # Save to JSONL file
    if _JSONL_FILE_HANDLE is not None:
        example_id = _get_field(doc, "sample_id") or _get_field(doc, "doc_id") or "unknown"
        dataset_name = doc.get("dataset_name", "unknown")
        source_model = doc.get("source_model", dataset_name)
        seed = os.environ.get("LMEVAL_JUDGE_SEED", "unknown")

        jsonl_entry = {
            "dataset_name": dataset_name,
            "source_model": source_model,
            "example_id": str(example_id),
            "seed": str(seed),
            "_judge_trait": trait or "all",
            "prompt": prompt_text,
            "judge_response_raw": raw_output,
            "scores": {t: (None if math.isnan(s) else s) for t, s in scores.items()},
            "evidence": evidence,
        }

        for key in ["doc_id", "sample_id", "task", "disorder", "symptom", "severity"]:
            if key in doc:
                jsonl_entry[key] = doc[key]

        _JSONL_FILE_HANDLE.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
        _JSONL_FILE_HANDLE.flush()

    return scores


# %% dataset loading

def load_dataset_with_tags(**kwargs) -> Dict[str, datasets.Dataset]:
    """
    Custom dataset loader that loads CSV files, tags each with
    ``dataset_name`` / ``source_model`` metadata, and then **triples** every
    row so that each original example appears once per judge trait
    (``aware``, ``pro_help``, ``crisis_help``).  The trait is stored in
    the ``_judge_trait`` column.
    """
    data_files = kwargs.get("data_files", {})

    if isinstance(data_files, dict) and "test" in data_files:
        test_files = data_files["test"]
        if not isinstance(test_files, list):
            test_files = [test_files]
    else:
        raise ValueError(
            f"data_files must contain 'test' key with list of files. "
            f"Got: {type(data_files)} with keys: "
            f"{list(data_files.keys()) if isinstance(data_files, dict) else 'N/A'}"
        )

    all_datasets = []

    # Map filename substrings to canonical dataset/model names.
    _NAME_MAP = {
        "Llama-3.1-8B": "Llama-3.1-8B",
        "Llama": "Llama-3.1-8B",
        "Qwen3-8B": "Qwen3-8B",
        "Qwen": "Qwen3-8B",
        "Gemma-3-12B": "Gemma-3-12B",
        "Gemma": "Gemma-3-12B",
        "OLMo-3-7B": "OLMo-3-7B",
        "OLMo": "OLMo-3-7B",
        "Olmo": "OLMo-3-7B",
    }

    for file_path in test_files:
        file_name = Path(file_path).name
        dataset_name = None
        for substr, canonical in _NAME_MAP.items():
            if substr in file_name:
                dataset_name = canonical
                break
        if dataset_name is None:
            dataset_name = Path(file_path).stem
        source_model = dataset_name

        dataset = datasets.load_dataset("csv", data_files={"test": file_path}, split="test")
        dataset = _normalize_text_columns(
            dataset, ["thinking", "response", "full_response", "prompt"]
        )

        def add_metadata(example, dn=dataset_name, sm=source_model):
            example["dataset_name"] = dn
            example["source_model"] = sm
            return example

        dataset = dataset.map(add_metadata)
        all_datasets.append(dataset)

    if len(all_datasets) > 1:
        combined_dataset = datasets.concatenate_datasets(all_datasets)
    else:
        combined_dataset = all_datasets[0]

    # Triple the dataset: one copy per trait so each example gets three
    # independent judge calls.
    trait_datasets = []
    for trait in TRAITS:
        def add_trait(example, t=trait):
            example["_judge_trait"] = t
            return example

        trait_datasets.append(combined_dataset.map(add_trait))

    tripled_dataset = datasets.concatenate_datasets(trait_datasets)

    return {"test": tripled_dataset}
