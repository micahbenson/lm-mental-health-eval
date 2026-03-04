#!/usr/bin/env python3
"""
Post-process LLM-judge outputs from llama_judge_results.jsonl.

Main outputs:
1) Heatmaps comparing dataset_name values on sampled prompts.
2) Symptom x severity matrices (CSV) per dataset_name/source_model.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from judge_analysis_common import METRICS, build_scored_df, load_jsonl


def aggregate_prompt_stats(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_name", "source_model", "example_id"]
    keep_first = [c for c in ["doc_id", "sample_id", "task", "disorder", "symptom", "severity", "prompt"] if c in df.columns]

    agg_spec: Dict[str, Any] = {"seed": pd.Series.nunique}
    for metric in METRICS:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_min"] = (metric, "min")
        agg_spec[f"{metric}_max"] = (metric, "max")

    g = df.groupby(group_cols, as_index=False)
    base = g.agg(num_seeds=("seed", "nunique"), **{k: v for k, v in agg_spec.items() if k != "seed"})

    if keep_first:
        meta = g[keep_first].first()
        out = base.merge(meta, on=group_cols, how="left")
    else:
        out = base

    for metric in METRICS:
        out[f"{metric}_std"] = out[f"{metric}_std"].fillna(0.0)
    return out


def dataset_summary_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name, sub in agg_df.groupby("dataset_name"):
        for metric in METRICS:
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "metric": metric,
                    "num_prompts": int(len(sub)),
                    "avg_num_seeds": float(sub["num_seeds"].mean()),
                    "mean_of_prompt_means": float(sub[f"{metric}_mean"].mean()),
                    "std_of_prompt_means": float(sub[f"{metric}_mean"].std(ddof=0)),
                    "mean_prompt_std": float(sub[f"{metric}_std"].mean()),
                    "min_prompt_mean": float(sub[f"{metric}_mean"].min()),
                    "max_prompt_mean": float(sub[f"{metric}_mean"].max()),
                }
            )
    return pd.DataFrame(rows)


def symptom_severity_summary_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate at dataset_name/source_model/symptom/severity level.
    Uses per-prompt seed-averaged values (e.g., aware_mean) as inputs.
    """
    df = agg_df.copy()
    if "symptom" not in df.columns:
        df["symptom"] = "unknown"
    if "severity" not in df.columns:
        df["severity"] = np.nan

    df["symptom"] = df["symptom"].fillna("unknown").astype(str)

    rows: List[Dict[str, Any]] = []
    group_cols = ["dataset_name", "source_model", "symptom", "severity"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        dataset_name, source_model, symptom, severity = keys
        row: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "source_model": source_model,
            "symptom": symptom,
            "severity": severity,
            "num_prompts": int(len(sub)),
            "avg_num_seeds": float(sub["num_seeds"].mean()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(sub[f"{metric}_mean"].mean())
            row[f"{metric}_std"] = float(sub[f"{metric}_mean"].std(ddof=0))
            row[f"{metric}_min"] = float(sub[f"{metric}_mean"].min())
            row[f"{metric}_max"] = float(sub[f"{metric}_mean"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["dataset_name", "source_model", "symptom", "severity"],
        na_position="last",
    )


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def save_and_print_symptom_severity_matrices(
    symsev_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    For each dataset_name/source_model and metric, save a pivot table with:
      rows = symptom
      cols = severity
      values = metric mean
    """
    matrix_dir = output_dir / "symptom_severity_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    if symsev_df.empty:
        print("\nNo symptom/severity rows available for matrix tables.")
        return

    print("\nSymptom x Severity tables (mean across seeds):")
    grouped = symsev_df.groupby(["dataset_name", "source_model"], dropna=False)
    for (dataset_name, source_model), sub in grouped:
        print(f"\n=== dataset_name={dataset_name} | source_model={source_model} ===")
        for metric in METRICS:
            value_col = f"{metric}_mean"
            pivot = (
                sub.pivot_table(
                    index="symptom",
                    columns="severity",
                    values=value_col,
                    aggfunc="mean",
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            print(f"\n[{metric}]")
            print(pivot.to_string())

            out_name = (
                f"{_slugify(dataset_name)}__{_slugify(source_model)}__"
                f"{metric}_symptom_x_severity.csv"
            )
            pivot.to_csv(matrix_dir / out_name)

    print(f"\nSaved matrix CSVs to: {matrix_dir}")


def _balanced_counts(total_n: int, n_groups: int) -> List[int]:
    base = total_n // n_groups
    rem = total_n % n_groups
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def sample_balanced_example_ids(
    agg_df: pd.DataFrame,
    dataset_names: Sequence[str],
    target_total: int,
    random_state: int,
) -> List[str]:
    """
    Sample example_ids with equal counts across severity levels.
    Uses intersection across dataset_name values to ensure comparable columns.
    """
    sub = agg_df[agg_df["dataset_name"].isin(dataset_names)].copy()
    sub = sub[sub["severity"].notna()].copy()
    if sub.empty:
        return []

    # Require each chosen example_id to exist in every selected dataset_name.
    counts = sub.groupby("example_id")["dataset_name"].nunique()
    valid_ids = counts[counts == len(dataset_names)].index
    sub = sub[sub["example_id"].isin(valid_ids)].copy()
    if sub.empty:
        return []

    sev_map = (
        sub.groupby("example_id", as_index=False)["severity"]
        .first()
        .sort_values("severity")
    )

    sev_groups = [(sev, grp["example_id"].tolist()) for sev, grp in sev_map.groupby("severity")]
    if not sev_groups:
        return []

    target_total = min(target_total, len(sev_map))
    requested = _balanced_counts(target_total, len(sev_groups))
    sampled_ids: List[str] = []
    rng = np.random.RandomState(random_state)
    remaining_pool: List[str] = []

    for i, (_, candidates) in enumerate(sev_groups):
        rng.shuffle(candidates)
        n_take = min(requested[i], len(candidates))
        sampled_ids.extend(candidates[:n_take])
        remaining_pool.extend(candidates[n_take:])

    # Fill leftovers if some severity groups had too few candidates.
    need = target_total - len(sampled_ids)
    if need > 0 and remaining_pool:
        rng.shuffle(remaining_pool)
        sampled_ids.extend(remaining_pool[:need])

    sampled_ids = sampled_ids[:target_total]
    return sampled_ids


def _build_row_labels(df_meta: pd.DataFrame) -> List[str]:
    labels = []
    for _, row in df_meta.iterrows():
        sev = row.get("severity")
        sev_str = "NA" if pd.isna(sev) else f"{int(sev)}"
        sym = str(row.get("symptom", "unknown") or "unknown")
        labels.append(f"{sym} | severity={sev_str}")
    return labels


def plot_prompt_heatmap(
    pivot_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    title: str,
    value_label: str,
    out_path: Path,
    cmap: str,
    force_01: bool = False,
) -> None:
    row_labels = _build_row_labels(meta_df)
    n_rows, n_cols = pivot_df.shape

    fig = plt.figure(figsize=(max(12, 3.5 + n_cols * 1.2 + 9), max(8, n_rows * 0.48)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.5, max(3, n_cols * 1.2)], wspace=0.05)

    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(n_rows - 0.5, -0.5)
    ax_text.axis("off")
    for i, lbl in enumerate(row_labels):
        ax_text.text(1.0, i, lbl, ha="right", va="center", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    arr = pivot_df.values.astype(float)
    masked = np.ma.masked_invalid(arr)

    if force_01:
        vmin, vmax = 0.0, 1.0
    else:
        finite = arr[np.isfinite(arr)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

    im = ax.imshow(masked, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(list(pivot_df.columns), rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("dataset_name")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.3, color="white", alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.ax.set_ylabel(value_label, rotation=-90, va="bottom")
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_sampled_heatmaps(
    agg_df: pd.DataFrame,
    output_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    datasets = sorted(agg_df["dataset_name"].dropna().unique().tolist())
    if len(datasets) < 2:
        print("Heatmaps skipped: need at least 2 dataset_name values.")
        return pd.DataFrame()

    # Match requested behavior: number of sampled prompts == number of unique symptoms.
    target_total = int(
        agg_df["symptom"].dropna().astype(str).nunique()
        if "symptom" in agg_df.columns
        else 0
    )
    if target_total <= 0:
        print("Heatmaps skipped: no symptom labels found.")
        return pd.DataFrame()

    sampled_ids = sample_balanced_example_ids(
        agg_df=agg_df,
        dataset_names=datasets,
        target_total=target_total,
        random_state=random_state,
    )
    if not sampled_ids:
        print("Heatmaps skipped: no severity-balanced overlapping examples.")
        return pd.DataFrame()

    sub = agg_df[agg_df["example_id"].isin(sampled_ids)].copy()
    meta = (
        sub.groupby("example_id", as_index=False)[["severity", "symptom"]]
        .first()
        .sort_values(["severity", "example_id"], na_position="last")
        .reset_index(drop=True)
    )
    ordered_ids = meta["example_id"].tolist()

    for metric in METRICS:
        mean_piv = (
            sub.pivot_table(index="example_id", columns="dataset_name", values=f"{metric}_mean", aggfunc="first")
            .reindex(index=ordered_ids, columns=datasets)
        )
        std_piv = (
            sub.pivot_table(index="example_id", columns="dataset_name", values=f"{metric}_std", aggfunc="first")
            .reindex(index=ordered_ids, columns=datasets)
        )

        plot_prompt_heatmap(
            pivot_df=mean_piv,
            meta_df=meta,
            title=f"{metric} mean over seeds | severity-balanced sampled prompts",
            value_label=f"{metric}_mean",
            out_path=output_dir / f"judge_{metric}_mean_heatmap.png",
            cmap="viridis",
            force_01=True,
        )
        plot_prompt_heatmap(
            pivot_df=std_piv,
            meta_df=meta,
            title=f"{metric} std across seeds | severity-balanced sampled prompts",
            value_label=f"{metric}_std",
            out_path=output_dir / f"judge_{metric}_std_heatmap.png",
            cmap="Blues",
            force_01=False,
        )

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LLM-judge JSONL outputs.")
    parser.add_argument(
        "--input",
        type=str,
        default="/projectnb/buinlp/afitab/lm-mental-health-eval/lm_eval/tasks/winoreferral/llm-judge/results/llama_judge_results.jsonl",
        help="Input JSONL with fields dataset_name/source_model/example_id/seed/prompt/judge_response_raw/...",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/projectnb/buinlp/afitab/lm-mental-health-eval/lm_eval/tasks/winoreferral/llm-judge/results/judge_analysis_output/v2/",
        help="Directory for heatmaps and symptom/severity matrices.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for severity-balanced prompt sampling.",
    )
    parser.add_argument(
        "--max_per_severity",
        type=int,
        default=None,
        help="Deprecated (ignored). Kept for backward CLI compatibility.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {in_path}")
    raw_df = load_jsonl(in_path)
    print(f"Rows loaded: {len(raw_df)}")

    scored_df = build_scored_df(raw_df)
    agg_df = aggregate_prompt_stats(scored_df)
    symsev_df = symptom_severity_summary_table(agg_df)

    # Only produce symptom/severity matrices and heatmaps.
    save_and_print_symptom_severity_matrices(symsev_df=symsev_df, output_dir=out_dir)

    sampled_meta = generate_sampled_heatmaps(
        agg_df=agg_df,
        output_dir=out_dir,
        random_state=args.random_state,
    )
    if not sampled_meta.empty:
        print(f"Sampled prompts: {len(sampled_meta)}")

    print("\nDone.")


if __name__ == "__main__":
    main()


