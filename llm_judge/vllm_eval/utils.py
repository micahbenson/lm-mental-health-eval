import csv
import logging
import re
from pathlib import Path

import pandas as pd
import prompts

log = logging.getLogger(__name__)


def build_all_judge_messages(
    items: list[dict],
) -> tuple[list[list[dict]], list[tuple[int, str]]]:
    """Build every item x axis prompt for a single vLLM call.

    Returns:
        messages: one chat conversation per (item, axis)
        index_map: index_map[i] == (item_idx, axis_key) for messages[i]
    """
    messages: list[list[dict]] = []
    index_map: list[tuple[int, str]] = []

    for item_idx, item in enumerate(items):
        for axis_key, axis_desc, evaluation_guidelines in prompts.JUDGE_AXES:
            system_content = prompts.JUDGE_SYSTEM.format(
                axis_desc=axis_desc,
                evaluation_guidelines=evaluation_guidelines,
            )
            user_content = prompts.PROMPT_TEMPLATE.format(response=item["response"])

            messages.append([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ])

            if item_idx == 0:
                print(messages[-1])

            index_map.append((item_idx, axis_key))

    return messages, index_map


def parse_axis_response(text) -> str:
    """Extract the yes/no rating from a model response."""
    def get_rating(t):
        m = re.search(r"Rating:\s*\[\[(yes|no)\]\]", t, re.IGNORECASE)
        return m.group(1).lower() if m else ""

    if isinstance(text, str):
        return get_rating(text)
    elif isinstance(text, list):
        return [get_rating(t) for t in text]
    else:
        return ""


def judge_fieldnames() -> list[str]:
    return ["id"] + list(prompts.CATEGORY_COLUMNS)


def _completion_texts(completions):
    """Normalize vLLM chat outputs to plain completion strings."""
    texts = []
    for out in completions:
        if hasattr(out, "outputs"):
            if len(out.outputs) == 1:
                texts.append(out.outputs[0].text)
            else:
                texts.append([out.outputs[i].text for i in range(len(out.outputs))])
        elif isinstance(out, dict):
            texts.append(out.get("text", out.get("content", "")))
        else:
            texts.append(str(out))
    return texts


def scores_from_completions(
    items: list[dict],
    completions,
    index_map: list[tuple[int, str]],
) -> list[dict[str, str]]:
    """Map flat completion list back to per-item axis scores."""
    texts = _completion_texts(completions)
    if len(texts) != len(index_map):
        raise ValueError(
            f"Expected {len(index_map)} completions, got {len(texts)}"
        )

    item_scores = [
        {axis_key: "" for axis_key in prompts.CATEGORY_COLUMNS}
        for _ in items
    ]
    for text, (item_idx, axis_key) in zip(texts, index_map):
        item_scores[item_idx][axis_key] = parse_axis_response(text)
    return item_scores


def write_judge_csv(
    out_path: Path,
    items: list[dict],
    item_scores: list[dict[str, str]],
) -> list[dict]:
    """Write all judge rows to CSV."""
    fieldnames = judge_fieldnames()
    rows: list[dict] = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item, scores in zip(items, item_scores):
            row = {"id": item["id"], **scores}
            writer.writerow(row)
            rows.append(row)
    return rows


def run_judge(
    llm,
    items: list[dict],
    out_path: Path,
    enable_thinking: bool | None = False,
    enable_thinking_argument: str | None = "enable_thinking",
    max_items: int | None = None,
) -> pd.DataFrame:
    """Score every item on each axis using a single vLLM generate call."""
    if max_items is not None:
        items = items[:max_items]

    n_prompts = len(items) * len(prompts.CATEGORY_COLUMNS)
    log.info(
        "Running judge on %d items (%d prompts) -> %s",
        len(items), n_prompts, out_path,
    )

    messages, index_map = build_all_judge_messages(items)
    completions = llm.generate(
        messages,
        enable_thinking=enable_thinking,
        enable_thinking_argument=enable_thinking_argument,
    )
    item_scores = scores_from_completions(items, completions, index_map)
    rows = write_judge_csv(out_path, items, item_scores)
    return pd.DataFrame(rows)
