#!/usr/bin/env python3
"""Analyze judge JSONL and generate summary, grouped bar, and severity heatmaps."""

import argparse
from pathlib import Path

import pandas as pd

from judge_analysis_common import (
    METRICS,
    METRIC_LABELS,
    build_scored_df,
    dataset_summary,
    load_jsonl,
    plot_grouped_bar,
    plot_severity_heatmaps,
    symptom_severity_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=(
            "/projectnb/buinlp/afitab/lm-mental-health-eval/"
            "lm_eval/tasks/winoreferral/llm-judge/results/llama_judge_results.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/projectnb/buinlp/afitab/lm-mental-health-eval/"
            "lm_eval/tasks/winoreferral/llm-judge/results/judge_analysis_output/llama_judge_jsonl/"
        ),
    )
    parser.add_argument("--ci", type=float, default=1.96, help="z value for error bars")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {in_path}")
    raw_df = load_jsonl(in_path)
    print(f"  Total JSONL rows: {len(raw_df)}")

    has_trait = "_judge_trait" in raw_df.columns
    if has_trait:
        n_old = int((raw_df["_judge_trait"].isna() | (raw_df["_judge_trait"] == "all")).sum())
        n_new = len(raw_df) - n_old
        print(f"  Old-format rows (all-at-once): {n_old}")
        print(f"  New-format rows (per-trait):   {n_new}")

    # Preserve mixed-format compatibility by scoring old/new partitions separately.
    if has_trait:
        mask_new = raw_df["_judge_trait"].isin(METRICS)
        parts = []
        df_old = raw_df[~mask_new].copy()
        df_new = raw_df[mask_new].copy()
        if not df_old.empty:
            df_old = df_old.drop(columns=["_judge_trait", "scores", "evidence"], errors="ignore")
            parts.append(build_scored_df(df_old))
        if not df_new.empty:
            parts.append(build_scored_df(df_new))
        scored_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        scored_df = build_scored_df(raw_df)

    print(f"  Scored examples: {len(scored_df)}")
    print(f"  Datasets: {sorted(scored_df['dataset_name'].dropna().unique())}")

    summary = dataset_summary(scored_df)
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = f"{'Dataset':<20} {'n':>5}"
    for m in METRICS:
        header += f"  {METRIC_LABELS[m]:>12} (SE)"
    print(header)
    for _, row in summary.iterrows():
        line = f"{row['dataset_name']:<20} {int(row['n']):>5}"
        for m in METRICS:
            line += f"  {row[f'{m}_mean']:>6.3f} ({row[f'{m}_se']:.3f})"
        print(line)

    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved summary CSV: {out_dir / 'summary.csv'}")

    plot_grouped_bar(
        summary_df=summary,
        out_path=out_dir / "grouped_bar.png",
        z=args.ci,
        title="LLM Judge Results by Source Model",
        y_label="Mean Score (0-100)",
        y_max=100.0,
    )
    print(f"Saved grouped bar: {out_dir / 'grouped_bar.png'}")

    symsev = symptom_severity_table(scored_df)
    if not symsev.empty:
        symsev.to_csv(out_dir / "symptom_severity.csv", index=False)
        print(f"Saved symptom-severity CSV: {out_dir / 'symptom_severity.csv'}")
        plot_severity_heatmaps(symsev, out_dir)
        print(f"Saved severity heatmaps to: {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
