"""
Stratified sampling of lm_eval BDI-rephrase results for the LLM-judge pipeline.

Reads lm_eval samples JSONL files from bdi_full_results/, picks one seed per
model, applies severity-balanced stratified sampling, and writes one CSV per
model (plus a combined CSV) to the data/ directory.

Output CSVs have the columns expected by judge_utils.load_dataset_with_tags.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# =========================================================================
# Config
# =========================================================================
BASE_DIR = Path(
    "/projectnb/buinlp/afitab/lm-mental-health-eval/"
    "lm_eval/tasks/winoreferral/data"
)
RESULTS_DIR = BASE_DIR / "bdi_full_results"
OUT_DIR = BASE_DIR

MODELS: Dict[str, str] = {
    "allenai__Olmo-3-7B-Instruct": "OLMo-3-7B",
    "google__gemma-3-12b-it": "Gemma-3-12B",
    "meta-llama__Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}

N_TOTAL = 100
SEED = 42

# =========================================================================
# JSONL loader
# =========================================================================

def load_lmeval_jsonl(path: Path) -> pd.DataFrame:
    """Flatten one lm_eval samples JSONL into a DataFrame with the columns
    expected by judge_utils (sample_id, prompt, response, severity, ...)."""
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            doc = rec.get("doc", {})
            tags = doc.get("tags", {})

            resps = rec.get("filtered_resps", [])
            response = resps[0] if resps else ""

            rows.append({
                "task": "bdi_rephrase",
                "doc_id": rec.get("doc_id", ""),
                "sample_id": doc.get("id", ""),
                "prompt": doc.get("prompt_text", ""),
                "disorder": tags.get("disorder", ""),
                "symptom": tags.get("symptom", ""),
                "severity": tags.get("severity", np.nan),
                "version": tags.get("version", ""),
                "rephrase_source": tags.get("rephrase_source", ""),
                "thinking": "",
                "response": response,
                "full_response": response,
                "awareness_rate": rec.get("awareness_rate", 0.0),
                "referral_rate": rec.get("referral_rate", 0.0),
                "hotline_rate": rec.get("hotline_rate", 0.0),
            })
    return pd.DataFrame(rows)


def pick_seed_file(model_dir: Path) -> Path:
    """Return the first (alphabetically sorted) samples JSONL in the model's
    result directory tree."""
    jsonl_files = sorted(model_dir.rglob("samples_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No samples JSONL found under {model_dir}")
    return jsonl_files[0]


# =========================================================================
# Balanced stratified sampler (kept from original script)
# =========================================================================

def balanced_stratified_sample(
    df_in: pd.DataFrame,
    n: int,
    group_cols: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grouped = list(df_in.groupby(group_cols, sort=True))
    if not grouped:
        raise ValueError(f"No groups found for {group_cols}.")

    keys = [k for k, _ in grouped]
    base = n // len(keys)
    rem = n % len(keys)

    alloc = {k: base for k in keys}

    keys_shuf = keys.copy()
    rng.shuffle(keys_shuf)
    for k in keys_shuf:
        if rem == 0:
            break
        g = df_in.groupby(group_cols).get_group(k)
        if alloc[k] < len(g):
            alloc[k] += 1
            rem -= 1

    parts = []
    leftover = 0
    for k, g in grouped:
        take = min(alloc[k], len(g))
        parts.append(g.sample(n=take, random_state=seed))
        leftover += alloc[k] - take

    sampled = pd.concat(parts, ignore_index=True)

    if leftover > 0:
        if "sample_id" in df_in.columns and "sample_id" in sampled.columns:
            remaining = df_in[~df_in["sample_id"].isin(sampled["sample_id"])].copy()
        else:
            remaining = df_in.copy()

        leftover = min(leftover, len(remaining))
        if leftover > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=leftover, random_state=seed)],
                ignore_index=True,
            )

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--n", type=int, default=N_TOTAL,
                        help="Number of stratified samples per model")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_slug, short_name in MODELS.items():
        model_dir = results_dir / model_slug
        if not model_dir.exists():
            print(f"[SKIP] {model_slug}: directory not found at {model_dir}")
            continue

        seed_file = pick_seed_file(model_dir)
        print(f"\n{'=' * 60}")
        print(f"Model: {short_name}  ({model_slug})")
        print(f"Seed file: {seed_file.name}")

        df = load_lmeval_jsonl(seed_file)
        df = df.dropna(subset=["severity"]).copy()
        df["severity"] = df["severity"].astype(int)
        df = df[df["severity"].isin([0, 1, 2, 3])].copy()

        print(f"Loaded {len(df)} rows")
        print(f"  Severity distribution: {df['severity'].value_counts().sort_index().to_dict()}")
        print(f"  Symptoms: {df['symptom'].nunique()}")
        print(f"  Rephrase sources: {df['rephrase_source'].nunique()}")

        sampled = balanced_stratified_sample(
            df, args.n, ["severity"], seed=args.seed,
        )

        print(f"\nSampled {len(sampled)} rows:")
        print(f"  Severity: {sampled['severity'].value_counts().sort_index().to_dict()}")
        print(f"  Symptoms: {sampled['symptom'].nunique()}")

        out_path = out_dir / f"{short_name}_samples_stratified{args.n}.csv"
        sampled.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
