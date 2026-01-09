
#!/usr/bin/env python3
"""
Pulls all of the papers published in Nature Communications from the previous month based on the date the script is run.

Outputs: ncomms_YYYY-MM.csv
Columns: publication_date, title, abstract, doi, url, last_author, pre-defined tags

Coding strategy:
- Crossref REST API for prior-month articles and core metadata
- User-defined KEYWORDS list; if any keyword appears in the abstract (case-insensitive),
  include it in the 'tags' column (semicolon-joined).

Required dependencies:
  pip install requests pandas
"""

import os
import re
import html
import time
from datetime import date, timedelta
from typing import List

import requests
import pandas as pd


# ----------------------------- Config -----------------------------
CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "email@help.com") #include email when making large fetches, like 10k+ to prevent getting locked out
JOURNAL_NAME    = "Nature Communications"
ROWS_PER_PAGE   = 100
MAX_RETRIES     = 5
BACKOFF_SECONDS = 3
HTTP_TIMEOUT    = 20

# >>> Define keywords here (examples shown) <<<
KEYWORDS: List[str] = [
    # not case-sensitive
    "gene editing",
    "CRISPR",
    "prime editing",
    "base editing",
    "gene therapy",
    "DNA",
    "genes", 
    "nucleotides", 
    "CRISPR", 
    "TALENs", 
    "vectors", 
    "transduction", 
    "gene addition", 
    "somatic", 
    "germline", 
    "in-vivo", 
    "ex-vivo", 
    "mutation", 
    "therapeutic protein", 
    "gene silencing", 
    "transgene",
    "ZFNs", 
    "TALENs", 
    "insertion", 
    "deletion",
    "gene regulatory network",
    "transcriptomics",
    "epigenetics",
    "epigenomics",
    "chromatin",
    "methylation",
    "histone modification",
    "non-coding RNA",
    "long non-coding RNA",
    "lncRNA",
    "microRNA",
    "miRNA",
    "CRISPR interference",
    "CRISPR activation",
    "3D genome organization",
    "ATAC-seq",
    "ChIP-seq",
    "Hi-C",
    "single-cell sequencing",
    "multi-omics",
    "spatial transcriptomics",
    "single-cell RNA sequencing",
    "gene regulatory network",
    "splicing",
    "proteomics",
    
]
# set to True to only match whole words (e.g., "lipoprotein" but not "lipoproteins").
WHOLE_WORDS = False
# ------------------------------------------------------------------


# ------------------------- Date utilities -------------------------
def previous_month_window(today: date):
    """
    Return the previous calendar month window and label.
    e.g., run on 2026-01-07 → window 2025-12-01..2025-12-31, label '2025-12'
    """
    first_this_month = date(today.year, today.month, 1)
    last_prev_month  = first_this_month - timedelta(days=1)
    first_prev_month = date(last_prev_month.year, last_prev_month.month, 1)
    label            = f"{last_prev_month.year}-{last_prev_month.month:02d}"
    return first_prev_month, last_prev_month, label


def iso_date_from_crossref_parts(parts: List[List[int]]) -> str:
    """Convert Crossref 'date-parts' ([[YYYY,MM,DD]]) to ISO 'YYYY-MM-DD'."""
    y = parts[0][0]
    m = parts[0][1] if len(parts[0]) >= 2 else 1
    d = parts[0][2] if len(parts[0]) >= 3 else 1
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_pub_date(item: dict) -> str:
    """Prefer published-online → published-print → issued → created.date-time."""
    for key in ("published-online", "published-print", "issued"):
        if key in item and "date-parts" in item[key] and item[key]["date-parts"]:
            return iso_date_from_crossref_parts(item[key]["date-parts"])
    if "created" in item and "date-time" in item["created"]:
        return item["created"]["date-time"][:10]
    return ""
# ------------------------------------------------------------------


# ---------------------- Text/author utilities ---------------------
def clean_abstract(raw: str) -> str:
    """Strip JATS/HTML tags & entities → plain text."""
    if not raw:
        return ""
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def format_author(auth_obj: dict) -> str:
    """Return 'given family' or fallbacks for Crossref author objects."""
    given   = auth_obj.get("given", "")
    family  = auth_obj.get("family", "")
    literal = auth_obj.get("literal", "")
    name    = auth_obj.get("name", "")
    if given or family:
        return " ".join([given, family]).strip()
    if name:
        return name.strip()
    if literal:
        return literal.strip()
    parts = [p for p in (auth_obj.get("prefix", ""), given, family, auth_obj.get("suffix", "")) if p]
    return " ".join(parts).strip()
# ------------------------------------------------------------------


# --------------------- keyword logic ----------------------
def build_keyword_patterns(keywords: List[str], whole_words: bool) -> List[re.Pattern]:
    """
    Precompile regex patterns for keywords.
    whole_words=True → wrap with \b boundaries (best for single words).
    whole_words=False → simple case-insensitive substring pattern.
    """
    patterns = []
    for kw in keywords:
        if not kw:
            continue
        escaped = re.escape(kw.strip())
        if whole_words:
            pat = re.compile(rf"\b{escaped}\b", re.I)
        else:
            pat = re.compile(escaped, re.I)
        patterns.append(pat)
    return patterns


def tags_from_abstract(abstract: str, patterns: List[re.Pattern], keywords: List[str]) -> List[str]:
    """
    Return a deduplicated list of keywords that appear in the abstract.
    """
    if not abstract:
        return []
    hits = []
    for kw, pat in zip(keywords, patterns):
        if pat.search(abstract):
            hits.append(kw)
    # Deduplicate while preserving order (case-insensitive key)
    seen = set()
    out = []
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out
# ------------------------------------------------------------------


# ------------------------ Crossref retrieval ----------------------
def fetch_crossref_page(params: dict, headers: dict, rows: int, offset: int) -> dict:
    """Fetch one page from Crossref with retry/backoff."""
    url = "https://api.crossref.org/works"
    q = params.copy()
    q["rows"]   = rows
    q["offset"] = offset
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, params=q, headers=headers, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 502, 503, 504):
            time.sleep(BACKOFF_SECONDS * attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch Crossref page after {MAX_RETRIES} attempts.")


def get_crossref_records(prev_start: date, prev_end: date) -> list:
    """
    Pull Nature Communications journal-articles published in the previous month
    with abstracts present.
    """
    params = {
        "filter": (
            f"container-title:{JOURNAL_NAME},"
            f"type:journal-article,"
            f"from-pub-date:{prev_start.isoformat()},"
            f"until-pub-date:{prev_end.isoformat()},"
            f"has-abstract:true"
        ),
        "mailto": CROSSREF_MAILTO,
        "sort": "published",
        "order": "asc",
    }
    headers = {"User-Agent": f"NCommsPrevMonth/2.1 (mailto:{CROSSREF_MAILTO})"}

    records = []
    offset  = 0

    while True:
        payload = fetch_crossref_page(params, headers, ROWS_PER_PAGE, offset)
        items   = payload.get("message", {}).get("items", [])
        if not items:
            break

        for it in items:
            title     = (it.get("title") or [""])[0]
            abstract  = clean_abstract(it.get("abstract", ""))
            doi       = it.get("DOI", "")
            url       = it.get("URL", "")
            pub_date  = extract_pub_date(it)
            authors   = it.get("author") or []
            last_auth = format_author(authors[-1]) if authors else ""

            records.append(
                {
                    "publication_date": pub_date,
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                    "url": url,
                    "last_author": last_auth,
                }
            )

        offset += ROWS_PER_PAGE

    return records
# ------------------------------------------------------------------


# ------------------------------- Main -----------------------------
def main():
    prev_start, prev_end, label = previous_month_window(date.today())
    csv_name = f"ncomms_{label}.csv"

    # 1) Crossref records for previous month
    records = get_crossref_records(prev_start, prev_end)

    # 2) Build keyword regex patterns
    patterns = build_keyword_patterns(KEYWORDS, WHOLE_WORDS)

    # 3) Tag abstracts
    for rec in records:
        rec["tags"] = "; ".join(tags_from_abstract(rec["abstract"], patterns, KEYWORDS))

    # 4) Export CSV
    df = pd.DataFrame(records)
    if "publication_date" in df.columns:
        df = df.sort_values(by="publication_date", kind="stable")

    df.to_csv(csv_name, index=False)
    print(f"Saved {len(df)} records to {csv_name}")


if __name__ == "__main__":
    main()
