import bibtexparser
import numpy as np
import pandas as pd

from bibtexparser.bparser import BibTexParser
from pathlib import Path
import gzip
import re
import unicodedata

PAPER_ENTRY_TYPES = {"article", "inproceedings", "incollection", "conference"}

VENUE_PATTERNS = {
    "acl":   [r"\bacl-(long|short|main)\b", r"\b[Pp]\d{2}-\d"],
    "naacl": [r"\bnaacl-(long|short|main)\b", r"\b[Nn]\d{2}-\d"],
    "emnlp": [r"\bemnlp-(long|short|main)\b", r"\b[Dd]\d{2}-\d"],
    "eacl":  [r"\beacl-(long|short|main)\b", r"\b[Ee]\d{2}-\d"],
    "aacl":  [r"\baacl-(long|short|main)\b", r"\b[Oo]\d{2}-\d"],
    "tacl":  [r"\btacl-\d+\b", r"\b[Qq]\d{2}-\d"],
}
FINDINGS_PATTERNS = {
    "acl":   [r"\bfindings-acl\b"],
    "naacl": [r"\bfindings-naacl\b"],
    "emnlp": [r"\bfindings-emnlp\b"],
    "eacl":  [r"\bfindings-eacl\b"],
    "aacl":  [r"\bfindings-aacl\b"],
    "tacl":  [],  # journal; no Findings track
}

def clean_latex(s: str) -> str:
    """Strip LaTeX accents and braces; normalize unicode and whitespace."""
    if not s:
        return ""
    s = re.sub(r"\\[`'^\"~=.]\{?([A-Za-z])\}?", r"\1", s)  # \'e -> e
    s = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", s)         # \emph{x} -> x
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()

def parse_bib(path: Path, venue_matcher) -> list[dict]:
    """Parse a (possibly gzipped) anthology+abstracts bib file."""

    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
   
    print("loading .gz")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        db = bibtexparser.load(f, parser=parser)

    papers, n_no_abstract = [], 0
    count = 0
    print("parsing bib entries")
    for e in db.entries:
        count += 1
        if count % 100 == 0:
            print(count)
        if e.get("ENTRYTYPE", "").lower() not in PAPER_ENTRY_TYPES:
            continue
        # Venue filter checks URL slug, bibkey, and booktitle in turn.
        if venue_matcher is not None:
            haystacks = [str(e.get(k, "")) for k in ("url", "ID", "booktitle")]
            if not any(venue_matcher.search(h) for h in haystacks):
                continue

        try:
            year = int(str(e.get("year", "")).strip().strip('"'))
        except ValueError:
            continue

        abstract = clean_latex(e.get("abstract", ""))
        if not abstract:
            n_no_abstract += 1
            continue

        papers.append({
            "key": e.get("ID", ""),
            "year": year,
            "title": clean_latex(e.get("title", "")),
            "abstract": abstract,
        })

    # log.info("Parsed %d papers with abstracts (%d skipped: no abstract)", len(papers), n_no_abstract)
    return papers

def build_venue_matcher(venues: list[str], include_findings: bool):
    pats = []
    if len(venues) == 0:
        return None
    for v in venues:
        if v not in VENUE_PATTERNS:
            raise ValueError(f"Unknown venue '{v}'. Choices: {list(VENUE_PATTERNS)}")
        pats.extend(VENUE_PATTERNS[v])
        if include_findings:
            pats.extend(FINDINGS_PATTERNS[v])
    return re.compile("|".join(pats), re.IGNORECASE) if pats else None


def load_csv(data_file):
    data = pd.read_csv(data_file)
    papers = []
    n_no_abstract = 0
    for _, e in data.iterrows():
        abstract = clean_latex(e.get("Abstract", ""))
        if not abstract:
            n_no_abstract += 1
            continue

        try:
            year = 0
            # year = int(e.get("Year"))
        except ValueError:
            continue

        papers.append({
            "key": e.get("ID", ""),
            "year": year,
            "title": clean_latex(e.get("Title", "")),
            "abstract": abstract,
            "venue": e.get("Venue", ""),
            "award": e.get("Award", "")
        })
    return papers


def load_responses_csv(data_file, limit=None):
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