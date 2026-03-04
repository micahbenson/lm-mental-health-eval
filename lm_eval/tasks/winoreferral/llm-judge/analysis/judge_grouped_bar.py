#!/usr/bin/env python3
"""Generate grouped bar chart from judge JSONL (thin wrapper)."""

import argparse
from pathlib import Path

from judge_analysis_common import METRICS, METRIC_LABELS, build_scored_df, dataset_summary, load_jsonl, plot_grouped_bar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=str,
        default=(
            "/projectnb/buinlp/afitab/lm-mental-health-eval/"
            "lm_eval/tasks/winoreferral/llm-judge/results/llama_judge_results_seed42.jsonl"
        ),
        help="Path to JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save figure. Defaults to same directory as input.",
    )
    parser.add_argument("--ci", type=float, default=1.96, help="z-multiplier for error bars")
    args = parser.parse_args()

    in_path = Path(args.samples)
    raw_df = load_jsonl(in_path)
    scored_df = build_scored_df(raw_df)
    summary = dataset_summary(scored_df)

    seeds_present = sorted({str(s) for s in scored_df.get("seed", []).tolist()}) if "seed" in scored_df.columns else []
    if len(seeds_present) == 1:
        title = f"LLM Judge Evaluation (Seed {seeds_present[0]})"
    elif len(seeds_present) > 1:
        title = f"LLM Judge Evaluation (Averaged Across {len(seeds_present)} Seeds)"
    else:
        title = "LLM Judge Evaluation"

    out_path = Path(args.output) if args.output else in_path.parent / "judge_grouped_bar.png"
    plot_grouped_bar(summary, out_path, z=args.ci, title=title, y_label="Mean Score (0-100)", y_max=100.0)

    print(f"Loaded {len(raw_df)} JSONL rows -> {len(scored_df)} scored examples")
    print(f"Saved figure to: {out_path}")

    print("\n--- Summary ---")
    header = f"{'Group':<20} {'n_examples':>10}"
    for m in METRICS:
        header += f"  {METRIC_LABELS[m]:>12} (SE, n)"
    print(header)
    for _, row in summary.iterrows():
        group = row["dataset_name"]
        n = int(row["n"])
        line = f"{group:<20} {n:>10}"
        for m in METRICS:
            line += f"  {row[f'{m}_mean']:>6.2f} ({row[f'{m}_se']:.2f}, {int(row[f'{m}_n'])})"
        print(line)


if __name__ == "__main__":
    main()
