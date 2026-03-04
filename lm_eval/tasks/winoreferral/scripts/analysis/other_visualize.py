"""
Visualization for empathy and engagement scores:
- Loads JSONL samples from a directory structure (results_dir/model/*.jsonl) OR explicit --jsonl-files.
- Computes empathy_score and engagement_score *inside this script* from the assistant text in `filtered_resps` / `resps`.
  - Strips <think>...</think> blocks before scoring.
- Optionally filters to symptom categories.
- Aggregates per (task, prompt, model) across seeds: mean/std/sem/n_seeds for each metric.
- Plots per-task:
    1) mean heatmap (prompts x models) with prompt text + [symptom]
    2) uncertainty heatmap (SEM or STD across seeds)

"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Add parent directory to path to import keywords
_script_dir = Path(__file__).parent
_winoreferral_dir = _script_dir.parent.parent
sys.path.insert(0, str(_winoreferral_dir))

from keywords import (
    ACKNOWLEDGEMENT_WORDS, DISSMISIVE_WORDS,
    ENGAGEMENT_WORDS
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

#%% keyword scoring helpers

def _normalize_text(text: str) -> str:
    t = (text or "").lower()
    # normalize apostrophes
    t = t.replace("’", "'")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks if present."""
    if not text:
        return ""
    return re.sub(_THINK_BLOCK_RE, "", text).strip()

def count_substring_hits(text: str, phrases: Sequence[str]) -> int:
    """Counts non-overlapping substring hits (simple)."""
    t = _normalize_text(text)
    total = 0
    for p in phrases:
        pp = _normalize_text(p)
        if not pp:
            continue
        total += t.count(pp)
    return total

def empathy_score_from_text(text: str) -> float:
    """
    Scalar empathy proxy:
      + acknowledgement hits
      - 0.5 * dismissive hits
    """
    t = _normalize_text(text)
    ack = count_substring_hits(t, ACKNOWLEDGEMENT_WORDS)
    dis = count_substring_hits(t, DISSMISIVE_WORDS)
    return float(ack - 0.5 * dis)

def engagement_score_from_text(text: str) -> float:
    """
    Engagement proxy:
      + engagement phrase hits
      + 1 if a '?' appears OR if a sentence starts with question starter.
    """
    t = _normalize_text(text)
    hits = count_substring_hits(t, ENGAGEMENT_WORDS)

    qmark = 1 if "?" in t else 0

    return float(hits + qmark)

def extract_assistant_text(rec: Dict) -> str:
    """
    extract the assistant-visible text from a lm-eval-harness style record
    """
    fr = rec.get("filtered_resps")
    if isinstance(fr, list) and fr:
        s = fr[-1]
        return str(s)

    # resps: list[list[str]]
    rr = rec.get("resps")
    if isinstance(rr, list) and rr:
        last = rr[-1]
        if isinstance(last, list) and last:
            return str(last[-1])
        return str(last)

    return ""

#%% parsing the data structures
@dataclass
class SampleRow:
    task: str
    model: str
    run_id: str
    seed: Optional[int]
    prompt: str
    disorder: Optional[str]
    symptom: Optional[str]
    severity: Optional[float]
    empathy_score: float
    engagement_score: float


_SEED_PATTERNS = [
    re.compile(r"(?:^|[\/_\-])seed(?:=|_|\-)?(\d+)(?:$|[\/_\-])", re.IGNORECASE),
    re.compile(r"(?:^|[\/_\-])s(?:eed)?(?:=|_|\-)?(\d+)(?:$|[\/_\-])", re.IGNORECASE),
]

def _extract_run_id(filename: str) -> str:
    base = Path(filename).name
    if not base.startswith("samples_") or not base.endswith(".jsonl"):
        return base
    core = base[len("samples_") : -len(".jsonl")]
    return core.rsplit("_", 1)[-1] if "_" in core else core

def _extract_seed_from_name(name: str) -> Optional[int]:
    s = name.replace("\\", "/")
    for pat in _SEED_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def _extract_seed_from_record(rec: Dict) -> Optional[int]:
    for k in ("seed", "random_seed", "rng_seed"):
        if k in rec:
            try:
                return int(rec[k])
            except Exception:
                pass
    for k in ("metadata", "meta", "eval", "run", "args"):
        v = rec.get(k)
        if isinstance(v, dict):
            for kk in ("seed", "random_seed", "rng_seed"):
                if kk in v:
                    try:
                        return int(v[kk])
                    except Exception:
                        pass
    return None

def _task_from_filename(filename: str, task_names: Sequence[str]) -> Optional[str]:
    for tn in task_names:
        if f"samples_{tn}_" in filename:
            return tn
    return None

#%% loading

def _process_jsonl_file(
    jsonl_file: Path,
    task_names: Sequence[str],
    symptoms_to_include: Optional[Sequence[str]] = None,
) -> List[SampleRow]:
    rows: List[SampleRow] = []
    model_id = jsonl_file.parent.name
    task = _task_from_filename(jsonl_file.name, task_names)
    if task is None:
        return rows

    run_id = _extract_run_id(jsonl_file.name)

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            doc = rec.get("doc", {}) or {}
            tags = doc.get("tags", {}) or {}
            symptom = tags.get("symptom")

            if symptoms_to_include is not None and symptom not in symptoms_to_include:
                continue

            prompt = str(doc.get("prompt_text") or doc.get("prompt") or "")
            seed = _extract_seed_from_record(rec)
            if seed is None:
                seed = _extract_seed_from_name(jsonl_file.name)

            # Extract assistant text and compute metrics here
            assistant_raw = extract_assistant_text(rec)
            assistant_visible = _strip_think(assistant_raw)

            emp = empathy_score_from_text(assistant_visible)
            eng = engagement_score_from_text(assistant_visible)

            rows.append(
                SampleRow(
                    task=task,
                    model=model_id,
                    run_id=run_id,
                    seed=seed,
                    prompt=prompt,
                    disorder=tags.get("disorder"),
                    symptom=symptom,
                    severity=float(tags["severity"]) if tags.get("severity") is not None else None,
                    empathy_score=float(emp),
                    engagement_score=float(eng),
                )
            )
    return rows

def _postprocess_dataframe(df: pd.DataFrame, max_seeds: int) -> pd.DataFrame:
    if df.empty:
        return df

    # If seeds are missing everywhere, map run_id->seed PER MODEL
    if df["seed"].isna().all():
        for m in df["model"].unique():
            runs = sorted(df.loc[df["model"] == m, "run_id"].unique())[:max_seeds]
            run_to_seed = {r: i for i, r in enumerate(runs)}
            mask = df["model"] == m
            df.loc[mask, "seed"] = df.loc[mask, "run_id"].map(run_to_seed)
        df = df[df["seed"].notna()].copy()

    # Force int when possible
    try:
        df["seed"] = df["seed"].astype(int)
    except Exception:
        pass

    return df

def load_samples_from_files(
    jsonl_files: Sequence[str],
    task_names: Sequence[str],
    max_seeds: int = 10,
    symptoms_to_include: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows: List[SampleRow] = []
    for f in jsonl_files:
        p = Path(f)
        if not p.exists():
            print(f"Warning: File not found: {p}")
            continue
        rows.extend(_process_jsonl_file(p, task_names, symptoms_to_include))

    df = pd.DataFrame([r.__dict__ for r in rows])
    return _postprocess_dataframe(df, max_seeds)

def iter_jsonl_files_under_results_dir(
    results_dir: Path,
    exclude_tasks: Optional[Sequence[str]] = None,
) -> Iterable[Path]:
    """Yield results_dir/model/*.jsonl, excluding specified tasks"""
    results_dir = Path(results_dir)
    exclude_tasks = exclude_tasks or []
    
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            # Skip files matching excluded tasks
            should_exclude = False
            if exclude_tasks:
                for excluded_task in exclude_tasks:
                    if f"samples_{excluded_task}_" in jsonl.name:
                        should_exclude = True
                        break
            if not should_exclude:
                yield jsonl

def load_samples_from_results_dir(
    results_dir: str,
    task_names: Sequence[str],
    max_seeds: int = 10,
    symptoms_to_include: Optional[Sequence[str]] = None,
    exclude_tasks: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load samples from results directory, including all models like meta-llama__Llama-3.1-8B-Instruct"""
    files = [str(p) for p in iter_jsonl_files_under_results_dir(Path(results_dir), exclude_tasks=exclude_tasks)]
    
    # Debug: show which models have files
    models_with_files = {}
    for f in files:
        model = Path(f).parent.name
        if model not in models_with_files:
            models_with_files[model] = []
        models_with_files[model].append(Path(f).name)
    
    print(f"\nFiles found per model:")
    for model, file_list in sorted(models_with_files.items()):
        task_files = [f for f in file_list if any(f"samples_{tn}_" in f for tn in task_names)]
        print(f"  {model}: {len(task_files)} task files (out of {len(file_list)} total files)")
        if task_files:
            print(f"    Example: {task_files[0]}")
    
    return load_samples_from_files(
        jsonl_files=files,
        task_names=task_names,
        max_seeds=max_seeds,
        symptoms_to_include=symptoms_to_include,
    )

#%% aggregation

def aggregate_prompt_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not in df columns.")

    g = df.groupby(["task", "prompt", "model"], as_index=False)
    out = g.agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n_seeds=("seed", pd.Series.nunique),
        disorder=("disorder", "first"),
        symptom=("symptom", "first"),
        severity=("severity", "first"),
    )
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["n_seeds"].clip(lower=1))
    return out

#%% plotting helpers

def _sorted_prompts(df_task_agg: pd.DataFrame, order_by=("severity", "symptom", "prompt")) -> List[str]:
    meta = df_task_agg[["prompt", "severity", "symptom"]].drop_duplicates("prompt").copy()
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
    prompts = _sorted_prompts(agg_task)

    prompt_to_symptom = dict(
        agg_task[["prompt", "symptom"]].drop_duplicates("prompt").itertuples(index=False, name=None)
    )

    pivot = (
        agg_task.pivot(index="prompt", columns="model", values=value_col)
        .reindex(index=prompts, columns=models)
    )

    n_prompts, n_models = len(prompts), len(models)
    width = max(14, (2.6 + max(3, n_models * 0.85)) * 4.0)
    height = max(8, n_prompts * 0.38)

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.6, max(3, n_models * 0.85)], wspace=0.05)

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

    # Use [0,1] scaling if looks like probability-like
    if value_col == "mean" and agg_task["mean"].between(0, 1).all():
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
    task_names: Sequence[str],
    results_dir: Optional[str] = None,
    jsonl_files: Optional[Sequence[str]] = None,
    metrics: Sequence[str] = ("empathy_score", "engagement_score"),
    uncertainty_stat: str = "sem",
    max_seeds: int = 10,
    symptoms_to_include: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    exclude_tasks: Optional[Sequence[str]] = None,
) -> None:
    if jsonl_files is None and results_dir is None:
        raise SystemExit("Provide either --results-dir or --jsonl-files.")

    if jsonl_files is not None:
        df = load_samples_from_files(
            jsonl_files=jsonl_files,
            task_names=task_names,
            max_seeds=max_seeds,
            symptoms_to_include=symptoms_to_include,
        )
    else:
        df = load_samples_from_results_dir(
            results_dir=results_dir or "",
            task_names=task_names,
            max_seeds=max_seeds,
            symptoms_to_include=symptoms_to_include,
            exclude_tasks=exclude_tasks or ["depression_ambiguous_v1"],
        )

    if df.empty:
        raise SystemExit("No samples found. Check input files and filters.")

    # Get all models BEFORE sampling to ensure we include all models
    all_models_before_sampling = sorted(df["model"].unique())
    print(f"\nModels found before prompt sampling: {all_models_before_sampling}")

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
        
        # Check which models remain after sampling
        models_after_sampling = sorted(df["model"].unique())
        print(f"Models found after prompt sampling: {models_after_sampling}")
        missing_models = set(all_models_before_sampling) - set(models_after_sampling)
        if missing_models:
            print(f"WARNING: Models missing after sampling: {missing_models}")
            print("This may happen if these models don't have data for the sampled prompts.")

    # Quick diagnostics (helps catch uniform metrics)
    for m in ["empathy_score", "engagement_score"]:
        if m in df.columns:
            nz = float((df[m] != 0).mean())
            print(f"{m}: nonzero%={nz:.3f} min={df[m].min():.3f} max={df[m].max():.3f}")

    if output_dir is None:
        output_dir_path = Path(__file__).parent
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Use all models that were found before sampling, but only include those with data
    all_models = sorted(df["model"].unique())
    # Ensure we include all models that were found initially, even if they have no data for sampled prompts
    # (This ensures consistent model ordering in plots)
    all_models_for_plotting = sorted(set(all_models_before_sampling) & set(all_models))
    if len(all_models_for_plotting) != len(all_models_before_sampling):
        print(f"\nNote: Using {len(all_models_for_plotting)} models with data (out of {len(all_models_before_sampling)} total models)")
    tasks = sorted(df["task"].unique())

    for task in tasks:
        df_task = df[df["task"] == task].copy()
        print(f"\n=== Task: {task} | models={len(all_models)} | rows={len(df_task)} ===")

        for metric in metrics:
            if metric not in df_task.columns:
                print(f"[warn] metric {metric!r} not present; skipping")
                continue

            agg = aggregate_prompt_metric(df_task, metric=metric)
            
            # Get models that actually have data for this task
            models_with_data = sorted(agg["model"].unique())
            print(f"  Models with data for {metric}: {models_with_data}")

            # mean heatmap
            fig = plot_heatmap_with_text(
                agg_task=agg,
                models=models_with_data,
                value_col="mean",
                title=f"{task} | {metric} (mean over seeds)",
                cmap="viridis",
            )
            fname = output_dir_path / f"{task}_{metric}_mean.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
            plt.close(fig)

            # uncertainty heatmap
            fig = plot_heatmap_with_text(
                agg_task=agg,
                models=models_with_data,
                value_col=uncertainty_stat,
                title=f"{task} | {metric} ({uncertainty_stat} across seeds)",
                cmap="Blues",
            )
            fname = output_dir_path / f"{task}_{metric}_{uncertainty_stat}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
            plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None, help="Directory containing results_dir/model/*.jsonl")
    parser.add_argument("--jsonl-files", type=str, nargs="+", default=None, help="Explicit JSONL files to load")
    parser.add_argument("--task-names", type=str, nargs="+", required=True)
    parser.add_argument("--metrics", type=str, nargs="*", default=["empathy_score", "engagement_score"])
    parser.add_argument("--uncertainty-stat", type=str, default="sem", choices=["sem", "std"])
    parser.add_argument("--max-seeds", type=int, default=10)
    parser.add_argument("--symptoms", type=str, nargs="+", default=None, help="Symptom categories to include")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--exclude-tasks", type=str, nargs="+", default=["depression_ambiguous_v1"], help="Task names to exclude when loading from results directory")
    args = parser.parse_args()

    main(
        task_names=tuple(args.task_names),
        results_dir=args.results_dir,
        jsonl_files=args.jsonl_files,
        metrics=tuple(args.metrics),
        uncertainty_stat=args.uncertainty_stat,
        max_seeds=args.max_seeds,
        symptoms_to_include=args.symptoms,
        output_dir=args.output_dir,
        exclude_tasks=args.exclude_tasks,
    )
