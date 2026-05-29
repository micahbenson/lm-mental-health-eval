import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from anthropic import Anthropic

import load_dataset, prompts


# =========================
# HELPERS
# =========================

def get_provider(model: str) -> str:
    """Infer provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def build_prompt_records(items: list[dict]) -> list[dict]:
    records = []

    for item_idx, item in enumerate(items):
        for axis_key, axis_desc, evaluation_guidelines in prompts.JUDGE_AXES:
            system_prompt = prompts.JUDGE_SYSTEM.format(
                axis_desc=axis_desc,
                evaluation_guidelines=evaluation_guidelines,
            )
            user_prompt = prompts.PROMPT_TEMPLATE.format(
                response=item["response"],
            )

            records.append({
                "item_index": item_idx,
                "item_id": item["id"],
                "response": item["response"],
                "axis_key": axis_key,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            })

    return records


def extract_rating(text: str) -> str | None:
    if not text:
        return None

    match = re.search(r"Rating:\s*\[\[(yes|no)\]\]", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return None


# =========================
# OPENAI
# =========================

def run_inference_openai(client: OpenAI, model: str, record: dict) -> dict:
    response = client.responses.create(
        model=model,
        instructions=record["system_prompt"],
        input=record["user_prompt"],
    )

    output_text = response.output_text
    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "model_output": output_text,
        "rating": extract_rating(output_text),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def count_tokens_openai(client: OpenAI, model: str, record: dict) -> dict:
    count = client.responses.input_tokens.count(
        model=model,
        instructions=record["system_prompt"],
        input=record["user_prompt"],
    )

    return {
        "input_tokens": count.input_tokens,
    }


# =========================
# ANTHROPIC
# =========================

def run_inference_anthropic(
    client: Anthropic,
    model: str,
    record: dict,
    max_tokens: int = 1024,
) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=record["system_prompt"],
        messages=[{"role": "user", "content": record["user_prompt"]}],
    )

    # content is a list of blocks; concatenate any text blocks
    output_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    total_tokens = (
        (input_tokens or 0) + (output_tokens or 0)
        if (input_tokens is not None or output_tokens is not None)
        else None
    )

    return {
        "model_output": output_text,
        "rating": extract_rating(output_text),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def count_tokens_anthropic(client: Anthropic, model: str, record: dict) -> dict:
    count = client.messages.count_tokens(
        model=model,
        system=record["system_prompt"],
        messages=[{"role": "user", "content": record["user_prompt"]}],
    )

    return {
        "input_tokens": count.input_tokens,
    }


# =========================
# DISPATCH
# =========================

def run_inference(client, model: str, record: dict, max_tokens: int = 1024) -> dict:
    provider = get_provider(model)
    if provider == "openai":
        return run_inference_openai(client, model, record)
    elif provider == "anthropic":
        return run_inference_anthropic(client, model, record, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider for model: {model}")


def count_tokens(client, model: str, record: dict) -> dict:
    provider = get_provider(model)
    if provider == "openai":
        return count_tokens_openai(client, model, record)
    elif provider == "anthropic":
        return count_tokens_anthropic(client, model, record)
    else:
        raise ValueError(f"Unknown provider for model: {model}")


# =========================
# I/O
# =========================

def save_csv(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_jsonl(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path.with_suffix(".jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_wide_ratings_csv(rows: list[dict], output_path: str | Path) -> None:
    """Saves one row per response: id, mental_health_referral, healthcare_referral, hotline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis_columns = [ax[0] for ax in prompts.JUDGE_AXES]

    item_rows: dict[str, dict] = {}

    for row in rows:
        item_id = row.get("item_id", "")

        if item_id not in item_rows:
            item_rows[item_id] = {"id": item_id, **{ax: "" for ax in axis_columns}}

        axis_key = row.get("axis_key", "")
        if axis_key in axis_columns:
            item_rows[item_id][axis_key] = row.get("rating", "")

    wide_rows = list(item_rows.values())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id"] + axis_columns)
        writer.writeheader()
        writer.writerows(wide_rows)


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI or Anthropic judge prompts, or count input tokens."
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV file, e.g. ../splits/best_papers_acl_full.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/judge_results.csv",
        help="Path to output CSV file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help=(
            "Model name. Provider is auto-detected: names starting with "
            "'claude' use Anthropic; everything else uses OpenAI. "
            "Examples: gpt-5.4-mini, claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7."
        ),
    )

    # parser.add_argument(
    #     "--prompt-version",
    #     choices=["old", "new", "dspy"],
    #     default="old",
    #     help="Whether to use old or new prompts.",
    # )

    parser.add_argument(
        "--mode",
        choices=["infer", "count"],
        default="count",
        help="infer = run model outputs; count = only count input tokens.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of responses to process. If omitted, processes all responses.",
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum number of prompt calls to process. ",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, uses OPENAI_API_KEY environment variable.",
    )

    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key. If omitted, uses ANTHROPIC_API_KEY environment variable.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max_tokens for Anthropic inference calls (ignored by OpenAI).",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API calls.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save results after every N calls.",
    )

    return parser.parse_args()


def build_client(args: argparse.Namespace):
    """Build the right client based on the model name."""
    provider = get_provider(args.model)

    if provider == "openai":
        print(f"Using openai key for model: {args.model}")
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Pass --api-key or set OPENAI_API_KEY."
            )
        return OpenAI(api_key=api_key), provider

    elif provider == "anthropic":
        print(f"Using anthropic key for model: {args.model}")
        api_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. "
                "Pass --anthropic-api-key or set ANTHROPIC_API_KEY."
            )
        return Anthropic(api_key=api_key), provider

    else:
        raise ValueError(f"Unknown provider for model: {args.model}")


def main() -> None:
    args = parse_args()
    start_time = time.time() 

    client, provider = build_client(args)

    items = load_dataset.load_responses_csv(args.input_csv, limit=args.limit)

    if not items:
        print("No responses found in input file. Aborting.")
        raise SystemExit(1)

    records = build_prompt_records(items)

    if args.max_calls is not None:
        records = records[:args.max_calls]

    print(f"Prepared {len(records)} prompt calls.")
    print(f"Mode: {args.mode}")
    print(f"Provider: {provider}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    input_stem = Path(args.input_csv).stem
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{input_stem}_judge.csv"

    results = []

    for i, record in enumerate(records, start=1):
        row = {
            **record,
            "provider": provider,
            "model": args.model,
            "mode": args.mode,
            "call_index": i,
            "error": "",
            "model_output": "",
            "rating": "",
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": "",
        }

        try:
            if args.mode == "infer":
                api_result = run_inference(
                    client, args.model, record, max_tokens=args.max_tokens
                )
                row.update(api_result)

            elif args.mode == "count":
                api_result = count_tokens(client, args.model, record)
                row.update(api_result)

            else:
                raise ValueError(f"Unknown mode: {args.mode}")

        except Exception as e:
            row["error"] = repr(e)

        results.append(row)

        print(
            f"[{i}/{len(records)}] "
            f"{record['item_id']} | {record['axis_key']} | "
            f"tokens={row.get('input_tokens', '')} | "
            f"rating={row.get('rating', '')} | "
            f"error={bool(row['error'])}"
        )

        if i % args.save_every == 0:
            # save_csv(results, output)
            save_jsonl(results, output)
            save_wide_ratings_csv(results, str(output).replace(".csv", "_wide.csv"))

        if args.sleep > 0:
            time.sleep(args.sleep)

    # save_csv(results, output)
    save_jsonl(results, output)
    save_wide_ratings_csv(results, str(output).replace(".csv", "_wide.csv"))

    elapsed = time.time() - start_time   # ← total elapsed time
    print("Done.")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved JSONL to: {Path(output).with_suffix('.jsonl')}")
    print(f"Saved wide ratings CSV to: {str(output).replace('.csv', '_wide.csv')}")


if __name__ == "__main__":
    main()