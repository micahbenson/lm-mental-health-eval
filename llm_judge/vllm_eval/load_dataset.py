import pandas as pd
from pathlib import Path


def load_responses_csv(data_file: str | Path, limit: int | None = None) -> list[dict]:
    """Load a CSV with 'id' and 'response' columns."""
    df = pd.read_csv(data_file)

    if "id" not in df.columns or "response" not in df.columns:
        raise ValueError(
            f"Input CSV must have 'id' and 'response' columns. "
            f"Found: {list(df.columns)}"
        )

    if limit is not None:
        df = df.head(limit)

    items = []
    n_missing = 0
    for _, row in df.iterrows():
        response = str(row["response"]) if pd.notna(row["response"]) else ""
        if not response.strip():
            n_missing += 1
            continue
        items.append({
            "id": str(row["id"]),
            "response": response,
        })

    print(f"Loaded {len(items)} responses ({n_missing} skipped: empty response).")
    return items
