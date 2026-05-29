import argparse
import asyncio
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

import load_dataset, prompts


# =========================
# HELPERS
# =========================

def get_provider(model: str) -> str:
    """Infer provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def build_prompt_records(papers: list[dict]) -> list[dict]:
    records = []

    for paper_idx, paper in enumerate(papers):
        for axis_key, axis_desc, evaluation_guidelines in prompts.JUDGE_AXES:
            system_prompt = prompts.JUDGE_SYSTEM.format(
                axis_desc=axis_desc,
                evaluation_guidelines=evaluation_guidelines,
            )
            user_prompt = prompts.PROMPT_TEMPLATE.format(
                title=paper["title"],
                abstract=paper["abstract"],
            )

            records.append({
                "paper_index": paper_idx,
                "paper_key": paper["key"],
                "year": paper["year"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                # "venue": paper["venue"],
                # "award": paper["award"],
                "axis_key": axis_key,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            })

    return records


def extract_rating(text: str) -> int | None:
    if not text:
        return None

    if "[[ ## rating ## ]]" in text:
        return extract_rating_dspy(text)

    match = re.search(r"Rating:\s*\[\[(0|1|2)\]\]", text)
    if match:
        return int(match.group(1))

    return None


def extract_rating_dspy(text: str) -> int | None:
    if not text:
        return None

    if "[[ ## rating ## ]]" in text.lower():
        tail = text.lower().split("[[ ## rating ## ]]")[-1]
    else:
        tail = text.split(":")[-1]
    pattern = r"[012]"
    m = re.search(pattern, tail)
    return int(m.group()) if m else None


# =========================
# OPENAI (async)
# =========================

async def run_inference_openai(client: AsyncOpenAI, model: str, record: dict) -> dict:
    response = await client.responses.create(
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


async def count_tokens_openai(client: AsyncOpenAI, model: str, record: dict) -> dict:
    count = await client.responses.input_tokens.count(
        model=model,
        instructions=record["system_prompt"],
        input=record["user_prompt"],
    )

    return {
        "input_tokens": count.input_tokens,
    }


# =========================
# ANTHROPIC (async)
# =========================

async def run_inference_anthropic(
    client: AsyncAnthropic,
    model: str,
    record: dict,
    max_tokens: int = 1024,
) -> dict:
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=record["system_prompt"],
        messages=[{"role": "user", "content": record["user_prompt"]}],
    )

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


async def count_tokens_anthropic(client: AsyncAnthropic, model: str, record: dict) -> dict:
    count = await client.messages.count_tokens(
        model=model,
        system=record["system_prompt"],
        messages=[{"role": "user", "content": record["user_prompt"]}],
    )

    return {
        "input_tokens": count.input_tokens,
    }


# =========================
# DISPATCH (async)
# =========================

async def run_inference(client, model: str, record: dict, max_tokens: int = 1024) -> dict:
    provider = get_provider(model)
    if provider == "openai":
        return await run_inference_openai(client, model, record)
    elif provider == "anthropic":
        return await run_inference_anthropic(client, model, record, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider for model: {model}")


async def count_tokens(client, model: str, record: dict) -> dict:
    provider = get_provider(model)
    if provider == "openai":
        return await count_tokens_openai(client, model, record)
    elif provider == "anthropic":
        return await count_tokens_anthropic(client, model, record)
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
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_wide_ratings_csv(rows: list[dict], output_path: str | Path) -> None:
    """
    Saves one row per paper:

    ID, Title, Abstract, year,
    benchmark_sota_framing,
    language_specific,
    linguistic_phenomenon,
    llm_as_object,
    reasoning_cognition_claims,
    societal_impact

    Axis columns contain the extracted rating.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis_columns = [
        "benchmark_sota_framing",
        "language_specific",
        "linguistic_phenomenon",
        "llm_as_object",
        "reasoning_cognition_claims",
        "societal_impact",
    ]

    paper_rows = {}

    for row in rows:
        if row is None:
            continue
        paper_id = row.get("paper_key", "")

        if paper_id not in paper_rows:
            paper_rows[paper_id] = {
                "ID": paper_id,
                "Title": row.get("title", ""),
                "Abstract": row.get("abstract", ""),
                "year": row.get("year", ""),
                "benchmark_sota_framing": "",
                "language_specific": "",
                "linguistic_phenomenon": "",
                "llm_as_object": "",
                "reasoning_cognition_claims": "",
                "societal_impact": "",
            }

        axis_key = row.get("axis_key", "")
        if axis_key in axis_columns:
            paper_rows[paper_id][axis_key] = row.get("rating", "")

    wide_rows = list(paper_rows.values())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ID",
                "Title",
                "Abstract",
                "year",
                "benchmark_sota_framing",
                "language_specific",
                "linguistic_phenomenon",
                "llm_as_object",
                "reasoning_cognition_claims",
                "societal_impact",
            ],
        )
        writer.writeheader()
        writer.writerows(wide_rows)


def _bib_stem(bibfile: Path) -> str:
    bib_name = bibfile.name
    for suffix in (".bib.gz", ".bib.txt", ".bib", ".csv"):
        if bib_name.endswith(suffix):
            return bib_name[: -len(suffix)]
    return bibfile.stem


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI or Anthropic judge prompts in parallel, or count input tokens."
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
        help="Path to output directory.",
    )

    parser.add_argument(
        "--venues",
        type=str,
        default="acl,naacl,emnlp,eacl,aacl,tacl",
        help="Venues to include.",
    )

    parser.add_argument(
        "--include-findings",
        action="store_true",
        help="Include findings.",
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
        help="Number of papers to process. If omitted, processes all papers.",
    )

    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum number of prompt calls to process.",
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
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of in-flight API requests at once.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each call within a worker. Better to leave at 0 in async mode; rely on --concurrency for rate control.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save partial results after every N completed calls.",
    )

    return parser.parse_args()


def build_client(args: argparse.Namespace):
    """Build the right async client based on the model name."""
    provider = get_provider(args.model)

    if provider == "openai":
        print(f"Using openai key for model: {args.model}")
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Pass --api-key or set OPENAI_API_KEY."
            )
        return AsyncOpenAI(api_key=api_key), provider

    elif provider == "anthropic":
        print(f"Using anthropic key for model: {args.model}")
        api_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. "
                "Pass --anthropic-api-key or set ANTHROPIC_API_KEY."
            )
        return AsyncAnthropic(api_key=api_key), provider

    else:
        raise ValueError(f"Unknown provider for model: {args.model}")


# =========================
# WORKER
# =========================

async def process_record(
    sem: asyncio.Semaphore,
    client,
    args: argparse.Namespace,
    provider: str,
    record: dict,
    call_index: int,
) -> tuple[int, dict]:
    """Process a single record under the concurrency semaphore."""
    row = {
        **record,
        "provider": provider,
        "model": args.model,
        "mode": args.mode,
        "call_index": call_index,
        "error": "",
        "model_output": "",
        "rating": "",
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
    }

    async with sem:
        try:
            if args.mode == "infer":
                api_result = await run_inference(
                    client, args.model, record, max_tokens=args.max_tokens
                )
                row.update(api_result)

            elif args.mode == "count":
                api_result = await count_tokens(client, args.model, record)
                row.update(api_result)

            else:
                raise ValueError(f"Unknown mode: {args.mode}")

        except Exception as e:
            row["error"] = repr(e)

        if args.sleep > 0:
            await asyncio.sleep(args.sleep)

    return call_index, row


# =========================
# MAIN (async)
# =========================

async def main_async() -> None:
    args = parse_args()
    start_time = time.time() 

    client, provider = build_client(args)

    if ".bib" in args.input_csv:
        venues = [v.strip() for v in args.venues.split(",") if v.strip()]
        matcher = load_dataset.build_venue_matcher(venues, args.include_findings)
        print("Venue filter: %s (findings=%s)", venues, args.include_findings)
        papers = load_dataset.parse_bib(Path(args.input_csv), matcher, limit=args.limit)

    elif ".csv" in args.input_csv:
        papers = load_dataset.load_csv(args.input_csv, limit=args.limit)

    else:
        raise ValueError(
            f"Unsupported input file (must be .csv or .bib): {args.input_csv}"
        )

    if not papers:
        print("No papers with abstracts after filtering. Aborting.")
        raise SystemExit(1)

    records = build_prompt_records(papers)

    if args.max_calls is not None:
        records = records[:args.max_calls]

    print(f"Prepared {len(records)} prompt calls.")
    print(f"Mode: {args.mode}")
    # print(f"Prompt version: {args.prompt_version}")
    # print(f"Provider: {provider}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")

    bib_stem = _bib_stem(Path(args.input_csv))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{bib_stem}_per_paper_judge.csv"
    print(f"Output file path: {output}")

    # Pre-allocate results slot per call so final output is in original order
    results: list[dict | None] = [None] * len(records)

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            process_record(sem, client, args, provider, record, i)
        )
        for i, record in enumerate(records, start=1)
    ]

    completed = 0
    start_time = time.time()

    try:
        for fut in asyncio.as_completed(tasks):
            call_index, row = await fut
            results[call_index - 1] = row
            completed += 1

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (len(records) - completed) / rate if rate > 0 else float("inf")

            print(
                f"[{completed}/{len(records)}] "
                f"{row.get('paper_key', '')} | {row.get('axis_key', '')} | "
                f"tokens={row.get('input_tokens', '')} | "
                f"rating={row.get('rating', '')} | "
                f"error={bool(row['error'])} | "
                f"rate={rate:.2f}/s eta={eta:.0f}s"
            )

            if completed % args.save_every == 0:
                save_jsonl(results, output)
                save_wide_ratings_csv(
                    results, str(output).replace(".csv", "_wide.csv")
                )
    finally:
        # Make sure pending tasks are cancelled 
        for t in tasks:
            if not t.done():
                t.cancel()
        # Close the underlying HTTP session
        await client.close()

    # Final save
    
    save_jsonl(results, output)
    save_wide_ratings_csv(results, str(output).replace(".csv", "_wide.csv"))

    elapsed = time.time() - start_time   # ← total elapsed time
    print("Done.")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved JSONL to: {Path(output).with_suffix('.jsonl')}")
    print(f"Saved wide ratings CSV to: {str(output).replace('.csv', '_wide.csv')}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()