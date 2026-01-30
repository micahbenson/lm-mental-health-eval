"""
referral_visualize.py

Visualize winoreferral referral behavior averaged over random seeds (referral only)

Reads JSONL sample files from:
  results/[model_name]/samples_beck_depression_ambiguous_v1_*.jsonl

Then:
  - extracts (prompt, symptom, severity, seed/run_id, referral_rate)
  - aggregates across seeds per (prompt, model): mean/std/sem/n_seeds
  - plots:
      1) Mean P(referral) heatmap (prompts x models) with readable prompt text
      2) Uncertainty heatmap (SEM or STD across seeds)
"""
#%% imports
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


#%% data structure
@dataclass
class SampleRow:
    task: str
    model: str
    run_id: str
    seed: Optional[int]
    prompt: str
    symptom: Optional[str]
    severity: Optional[float]
    referral_rate: float


#%% parsing helpers
_SEED_PATTERNS = [
    re.compile(r"(?:^|[\/_\-])seed(?:=|_|\-)?(\d+)(?:$|[\/_\-])", re.IGNORECASE),
    re.compile(r"(?:^|[\/_\-])s(?:eed)?(?:=|_|\-)?(\d+)(?:$|[\/_\-])", re.IGNORECASE),
]

def extract_run_id(filename: str) -> str:
    base = Path(filename).name
    if not base.startswith("samples_") or not base.endswith(".jsonl"):
        return base
    core = base[len("samples_") : -len(".jsonl")]
    # samples_<task>_<runid>.jsonl  -> take last underscore chunk
    return core.rsplit("_", 1)[-1] if "_" in core else core

def extract_seed_from_name(name: str) -> Optional[int]:
    s = name.replace("\\", "/")
    for pat in _SEED_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def extract_seed_from_record(rec: Dict) -> Optional[int]:
    # common top-level keys
    for k in ("seed", "random_seed", "rng_seed"):
        if k in rec:
            try:
                return int(rec[k])
            except Exception:
                pass
    # common nested containers
    for k in ("metadata", "meta", "eval", "run", "args"):
        v = rec.get(k)
        if isinstance(v, dict):
            for kk in ("seed", "random_seed", "rng_seed"):
                if kk in v:
                    try:
                        return int(v[kk])
                    except Exception:
                        pass
    # sometimes in doc/tags
    doc = rec.get("doc")
    if isinstance(doc, dict):
        for kk in ("seed", "random_seed", "rng_seed"):
            if kk in doc:
                try:
                    return int(doc[kk])
                except Exception:
                    pass
        tags = doc.get("tags")
        if isinstance(tags, dict):
            for kk in ("seed", "random_seed", "rng_seed"):
                if kk in tags:
                    try:
                        return int(tags[kk])
                    except Exception:
                        pass
    return None


#%% file discovery & loading

def iter_sample_files(
    results_dir: Path,
    task_name: str,
) -> Iterable[Path]:
    """
    Yield all files matching:
      results_dir/[model]/samples_<task_name>_*.jsonl
    """
    results_dir = Path(results_dir)
    pattern = f"samples_{task_name}_*.jsonl"
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue
        for fp in sorted(model_dir.glob(pattern)):
            yield fp

def process_jsonl_file(fp: Path, task_name: str) -> List[SampleRow]:
    rows: List[SampleRow] = []
    model_id = fp.parent.name
    run_id = extract_run_id(fp.name)

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            doc = rec.get("doc", {}) or {}
            tags = doc.get("tags", {}) or {}

            prompt = str(doc.get("prompt_text") or doc.get("prompt") or "")
            symptom = tags.get("symptom")
            severity = tags.get("severity")
            severity_f = float(severity) if severity is not None else None

            seed = extract_seed_from_record(rec)
            if seed is None:
                seed = extract_seed_from_name(fp.name)

            # referral_rate: default 0.0 if missing (but you probably want to notice missing)
            referral = rec.get("referral_rate", 0.0)
            try:
                referral_f = float(referral)
            except Exception:
                referral_f = 0.0

            rows.append(
                SampleRow(
                    task=task_name,
                    model=model_id,
                    run_id=run_id,
                    seed=seed,
                    prompt=prompt,
                    symptom=symptom,
                    severity=severity_f,
                    referral_rate=referral_f,
                )
            )

    return rows

def load_referral_df(
    results_dir: Path,
    task_name: str = "beck_depression_ambiguous_v1",
    max_seeds: int = 10,
) -> pd.DataFrame:
    files = list(iter_sample_files(results_dir, task_name))
    if not files:
        raise SystemExit(
            f"No files found under {results_dir} matching */samples_{task_name}_*.jsonl"
        )

    rows: List[SampleRow] = []
    for fp in files:
        rows.extend(process_jsonl_file(fp, task_name))

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise SystemExit("Loaded 0 rows. Are the JSONL files empty?")

    # If seeds are missing everywhere, map run_id -> seed PER MODEL
    if df["seed"].isna().all():
        for m in df["model"].unique():
            runs = sorted(df.loc[df["model"] == m, "run_id"].unique())[:max_seeds]
            run_to_seed = {r: i for i, r in enumerate(runs)}
            mask = df["model"] == m
            df.loc[mask, "seed"] = df.loc[mask, "run_id"].map(run_to_seed)
        df = df[df["seed"].notna()].copy()

    # Ensure ints where possible
    try:
        df["seed"] = df["seed"].astype(int)
    except Exception:
        pass

    return df


#%% aggregation

def aggregate_referral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (prompt, model) across seeds:
      mean/std/sem/n_seeds (+ symptom/severity from first)
    """
    g = df.groupby(["task", "prompt", "model"], as_index=False)
    out = g.agg(
        mean=("referral_rate", "mean"),
        std=("referral_rate", "std"),
        n_seeds=("seed", pd.Series.nunique),
        symptom=("symptom", "first"),
        severity=("severity", "first"),
    )
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["n_seeds"].clip(lower=1))
    return out


# -----------------------------
# Plotting helpers
# -----------------------------

def sorted_prompts(agg_task: pd.DataFrame, order_by=("severity", "symptom", "prompt")) -> List[str]:
    meta = agg_task[["prompt", "severity", "symptom"]].drop_duplicates("prompt").copy()
    meta["severity"] = meta["severity"].fillna(-1)
    meta["symptom"] = meta["symptom"].fillna("")
    meta = meta.sort_values(list(order_by), ascending=True)
    return meta["prompt"].tolist()

def plot_heatmap_with_text(
    agg_task: pd.DataFrame,
    models: Sequence[str],
    value_col: str,
    title: str,
    wrap_width: int = 52,
    cmap: str = "RdYlGn_r",
) -> plt.Figure:
    prompts = sorted_prompts(agg_task)

    prompt_to_symptom = dict(
        agg_task[["prompt", "symptom"]].drop_duplicates("prompt").itertuples(index=False, name=None)
    )

    pivot = (
        agg_task.pivot(index="prompt", columns="model", values=value_col)
        .reindex(index=prompts, columns=models)
    )

    n_prompts, n_models = len(prompts), len(models)
    width = max(14, (2.8 + max(3, n_models * 0.85)) * 4.0)
    height = max(8, n_prompts * 0.38)

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.8, max(3, n_models * 0.85)], wspace=0.05)

    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(n_prompts - 0.5, -0.5)
    ax_text.axis("off")

    for i, p in enumerate(prompts):
        wrapped = "\n".join(textwrap.wrap(p, width=wrap_width))
        symptom = prompt_to_symptom.get(p, "") or ""
        suffix = f" [{symptom}]" if symptom else ""
        ax_text.text(1.0, i, wrapped + suffix, va="center", ha="right", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    data = pivot.values.astype(float)
    masked = np.ma.masked_invalid(data)

    if value_col == "mean":
        vmin, vmax = 0.0, 1.0
    else:
        finite = data[np.isfinite(data)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Models")

    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_prompts, 1), minor=True)
    ax.grid(which="minor", linewidth=0.3, color="white", alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.ax.set_ylabel(value_col, rotation=-90, va="bottom")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig

#%% main

def main(
    results_dir: str,
    task_name: str = "beck_depression_ambiguous_v1",
    uncertainty_stat: str = "sem",
    max_seeds: int = 10,
    symptoms: Optional[Sequence[str]] = None,
    max_prompts: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    df = load_referral_df(Path(results_dir), task_name=task_name, max_seeds=max_seeds)

    if symptoms is not None:
        df = df[df["symptom"].isin(symptoms)].copy()

    if df.empty:
        raise SystemExit("No rows after filtering by symptoms.")

    # Diagnostics: check if you're actually loading multiple seeds
    seeds_per_model = df.groupby("model")["seed"].nunique().sort_values(ascending=False)
    print("Seeds per model (nunique):")
    print(seeds_per_model.to_string())

    # If do_sample=False everywhere, variance can be ~0; still should have n_seeds > 1.
    print(
        f"referral_rate overall: min={df['referral_rate'].min():.3f} "
        f"max={df['referral_rate'].max():.3f} "
        f"mean={df['referral_rate'].mean():.3f}"
    )

    # Sample half the prompts with equal distribution across symptoms
    unique_prompts = df["prompt"].unique()
    total_prompts = len(unique_prompts)
    target_prompts = max(1, total_prompts // 2)
    
    # Get unique prompts per symptom
    symptom_prompts = {}
    for symptom in df["symptom"].dropna().unique():
        symptom_df = df[df["symptom"] == symptom]
        symptom_prompts[symptom] = symptom_df["prompt"].unique()
    
    # Calculate prompts per symptom for equal distribution
    num_symptoms = len(symptom_prompts)
    if num_symptoms > 0:
        prompts_per_symptom = max(1, target_prompts // num_symptoms)
        sampled_prompts = set()
        
        for symptom, prompts in symptom_prompts.items():
            if len(prompts) > prompts_per_symptom:
                sampled = pd.Series(prompts).sample(n=prompts_per_symptom, random_state=42).tolist()
            else:
                sampled = prompts.tolist()
            sampled_prompts.update(sampled)
        
        # If we haven't reached target, add more prompts evenly
        if len(sampled_prompts) < target_prompts:
            remaining_needed = target_prompts - len(sampled_prompts)
            all_remaining = [p for p in unique_prompts if p not in sampled_prompts]
            if all_remaining:
                additional = pd.Series(all_remaining).sample(
                    n=min(remaining_needed, len(all_remaining)), 
                    random_state=42
                ).tolist()
                sampled_prompts.update(additional)
        
        # Filter dataframe to only include sampled prompts
        df = df[df["prompt"].isin(sampled_prompts)].copy()
        print(f"Sampled {len(sampled_prompts)} prompts (target: {target_prompts}) from {total_prompts} total prompts")
        symptom_counts = df["symptom"].value_counts()
        print(f"Symptom distribution: {dict(symptom_counts)}")
    
    # Override with max_prompts if explicitly provided (takes precedence)
    if max_prompts is not None and max_prompts > 0:
        temp_agg = aggregate_referral(df)
        keep = sorted_prompts(temp_agg)[:max_prompts]
        df = df[df["prompt"].isin(keep)].copy()

    agg = aggregate_referral(df)

    models = sorted(df["model"].unique())

    if output_dir is None:
        outdir = Path(__file__).parent
    else:
        outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Mean heatmap
    fig = plot_heatmap_with_text(
        agg_task=agg,
        models=models,
        value_col="mean",
        title=f"{task_name} | referral_rate (mean over seeds)",
        cmap="RdYlGn_r",
    )
    fn = outdir / f"{task_name}_referral_rate_mean.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)

    # Uncertainty heatmap
    if uncertainty_stat not in ("sem", "std"):
        raise ValueError("--uncertainty-stat must be 'sem' or 'std'")

    fig = plot_heatmap_with_text(
        agg_task=agg,
        models=models,
        value_col=uncertainty_stat,
        title=f"{task_name} | referral_rate ({uncertainty_stat} across seeds)",
        cmap="Blues",
    )
    fn = outdir / f"{task_name}_referral_rate_{uncertainty_stat}.png"
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"Saved: {fn}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Referral-only visualization for winoreferral (averaged over seeds).")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results/ directory containing model subdirs.")
    parser.add_argument("--task-name", type=str, default="beck_depression_ambiguous_v1", help="Task name in samples_<task>_*.jsonl")
    parser.add_argument("--uncertainty-stat", type=str, default="sem", choices=["sem", "std"], help="Uncertainty statistic across seeds")
    parser.add_argument("--max-seeds", type=int, default=10, help="Max seeds to map if missing everywhere")
    parser.add_argument("--symptoms", type=str, nargs="+", default=None, help="Optional symptom(s) to include")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on number of prompts shown")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save images (default: script directory)")
    args = parser.parse_args()

    main(
        results_dir=args.results_dir,
        task_name=args.task_name,
        uncertainty_stat=args.uncertainty_stat,
        max_seeds=args.max_seeds,
        symptoms=args.symptoms,
        max_prompts=args.max_prompts,
        output_dir=args.output_dir,
    )