#!/usr/bin/env python3
"""Shared utilities for LLM-as-a-judge analysis scripts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS: Sequence[str] = ("aware", "pro_help", "crisis_help")
METRIC_LABELS = {"aware": "Aware", "pro_help": "Pro-Help", "crisis_help": "Crisis-Help"}
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def _parse_all_at_once(raw: Any) -> Dict[str, float]:
    out = {m: 0.0 for m in METRICS}
    if not isinstance(raw, str) or not raw.strip():
        return out
    m = JSON_RE.search(raw)
    blob = m.group(0) if m else raw
    try:
        parsed = json.loads(blob)
    except Exception:
        return out
    for metric in METRICS:
        try:
            val = parsed.get(metric, 0)
            if isinstance(val, str) and val.upper() == "REFUSAL":
                out[metric] = float("nan")
            else:
                out[metric] = float(val or 0)
        except Exception:
            out[metric] = 0.0
    return out


def build_scored_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (example, seed) with metric columns."""
    df = raw_df.copy()
    is_per_trait = "_judge_trait" in df.columns

    # Case A: per-trait pipeline with pre-parsed scores dict.
    if is_per_trait and "scores" in df.columns:
        score_rows = df["scores"].apply(lambda x: x if isinstance(x, dict) else {}).apply(pd.Series)

    # Case B: per-trait pipeline without scores dict (fallback to parsing raw).
    elif is_per_trait:
        raw_col = "judge_response_raw" if "judge_response_raw" in df.columns else None
        parsed = (
            df[raw_col].apply(_parse_all_at_once).apply(pd.Series)
            if raw_col
            else pd.DataFrame(index=df.index)
        )
        score_rows = parsed
        for m in METRICS:
            score_rows.loc[df["_judge_trait"] != m, m] = np.nan

    # Case C: already-scored lm_eval sample rows (aware/pro_help/crisis_help at top level).
    elif all(m in df.columns for m in METRICS):
        score_rows = df[list(METRICS)].apply(pd.to_numeric, errors="coerce")

    # Case D: legacy JSON in judge_response_raw.
    else:
        raw_col = "judge_response_raw" if "judge_response_raw" in df.columns else None
        score_rows = (
            df[raw_col].apply(_parse_all_at_once).apply(pd.Series)
            if raw_col
            else pd.DataFrame(index=df.index)
        )

    for m in METRICS:
        if m not in score_rows.columns:
            score_rows[m] = np.nan if is_per_trait else 0.0

    df = pd.concat([df, score_rows[list(METRICS)]], axis=1)

    if is_per_trait:
        id_cols = [c for c in ["dataset_name", "source_model", "example_id", "seed"] if c in df.columns]
        meta_cols = [
            c
            for c in ["doc_id", "sample_id", "task", "disorder", "symptom", "severity", "prompt"]
            if c in df.columns and c not in id_cols
        ]
        agg: Dict[str, Any] = {m: "first" for m in METRICS}
        agg.update({c: "first" for c in meta_cols})
        if id_cols:
            df = df.groupby(id_cols, as_index=False).agg(agg)

    for col in ("dataset_name", "source_model", "example_id", "seed", "prompt"):
        if col not in df.columns:
            df[col] = "unknown"

    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    else:
        df["severity"] = np.nan

    df["example_id"] = df["example_id"].astype(str)
    return df


def dataset_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Mean + SE for continuous 0-100 rubric scores."""

    def _row(sub: pd.DataFrame, name: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {"dataset_name": name, "n": len(sub)}
        for m in METRICS:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna().astype(float)
            n = len(vals)
            if n == 0:
                mean, se = 0.0, 0.0
            elif n == 1:
                mean, se = float(vals.iloc[0]), 0.0
            else:
                mean = float(vals.mean())
                se = float(vals.std(ddof=1) / np.sqrt(n))
            row[f"{m}_mean"] = mean
            row[f"{m}_se"] = se
            row[f"{m}_n"] = n
        return row

    rows = [_row(sub, str(ds)) for ds, sub in scored_df.groupby("dataset_name")]
    rows.append(_row(scored_df, "OVERALL"))
    return pd.DataFrame(rows)


def symptom_severity_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    if "symptom" not in scored_df.columns:
        return pd.DataFrame()
    df = scored_df.copy()
    df["symptom"] = df["symptom"].fillna("unknown").astype(str)
    group_cols = ["dataset_name", "source_model", "symptom", "severity"]
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        ds, sm, sym, sev = keys
        r: Dict[str, Any] = {
            "dataset_name": ds,
            "source_model": sm,
            "symptom": sym,
            "severity": sev,
            "n": len(sub),
        }
        for m in METRICS:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            r[f"{m}_mean"] = float(vals.mean()) if len(vals) else 0.0
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["dataset_name", "source_model", "symptom", "severity"], na_position="last")


def plot_grouped_bar(
    summary_df: pd.DataFrame,
    out_path: Path,
    z: float = 1.96,
    title: str = "LLM Judge Results by Source Model",
    y_label: str = "Mean Score (0-100)",
    y_max: float = 100.0,
):
    groups = []
    for _, row in summary_df.iterrows():
        groups.append((
            row["dataset_name"],
            int(row["n"]),
            {m: (row[f"{m}_mean"], row[f"{m}_se"]) for m in METRICS},
        ))

    n_groups = len(groups)
    x = np.arange(len(METRICS))
    width = 0.8 / n_groups

    colors = plt.cm.Set2(np.linspace(0, 0.8, n_groups))
    fig, ax = plt.subplots(figsize=(max(8, 2 + n_groups * 2), 5))

    for i, (label, n, stats) in enumerate(groups):
        means = [stats[m][0] for m in METRICS]
        errs = [z * stats[m][1] for m in METRICS]
        offset = (i - (n_groups - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width * 0.88,
            yerr=errs,
            capsize=4,
            label=f"{label}\n(n={n})",
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
            error_kw=dict(lw=1.2, capthick=1.2),
        )
        label_offset = (max(errs) if errs else 0.0) + 0.8
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    ci_label = "95% CI" if abs(z - 1.96) < 0.01 else f"±{z:.1f} SE"
    ax.legend(title="Source Dataset", loc="upper right", framealpha=0.9, fontsize=8, title_fontsize=9)
    ax.annotate(
        f"Error bars: {ci_label}  (SE = sample SD / sqrt(n))",
        xy=(0.01, 0.97),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_severity_heatmaps(symsev_df: pd.DataFrame, out_dir: Path):
    if symsev_df.empty:
        return
    for ds, ds_sub in symsev_df.groupby("dataset_name"):
        for m in METRICS:
            pivot = ds_sub.pivot_table(
                index="symptom", columns="severity", values=f"{m}_mean", aggfunc="mean"
            ).sort_index(axis=0).sort_index(axis=1)
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.5), max(5, pivot.shape[0] * 0.45)))
            vals = pivot.values.astype(float)
            masked = np.ma.masked_invalid(vals)
            finite = vals[np.isfinite(vals)]
            vmin = float(np.min(finite)) if finite.size else 0.0
            vmax = float(np.max(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1e-6

            im = ax.imshow(masked, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=9)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(list(pivot.index), fontsize=8)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Symptom")
            ax.set_title(f"{METRIC_LABELS[m]} — {ds}", fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

            baseline = float(np.nanmean(finite)) if finite.size else 0.0
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.iloc[r, c]
                    if not (isinstance(val, float) and math.isnan(val)):
                        ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if val > baseline else "black")

            fig.tight_layout()
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ds)).strip("_")
            fig.savefig(out_dir / f"{slug}_{m}_severity_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
