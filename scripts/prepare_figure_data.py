"""Prepare data used by repository figures without heavyweight dependencies."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/data.csv"
OUTPUT_PATH = ROOT / "images/data_distribution_data.json"


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip()).lower()


def read_rows() -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            row.pop("Unnamed: 0", None)
            rows.append(row)
    return rows


def build_distribution() -> dict[str, object]:
    rows = [
        row
        for row in read_rows()
        if row.get("description") not in (None, "")
        and row.get("WikiData") not in (None, "")
    ]
    raw_valid_rows = len(rows)

    for row in rows:
        for column in ["description", "en_txt", "Name", "City"]:
            row[column] = normalize_text(row.get(column, ""))
        row["place_hints"] = normalize_text(f"{row.get('Name', '')} {row.get('City', '')}")
        row["text"] = normalize_text(
            f"{row['place_hints']} {row.get('description', '')} {row.get('en_txt', '')}"
        )

    seen_captions: set[str] = set()
    deduped_rows: list[dict[str, str]] = []
    for row in rows:
        caption = row.get("en_txt", "")
        if caption in seen_captions:
            continue
        seen_captions.add(caption)
        deduped_rows.append(row)

    wikidata_to_place_hints: dict[str, list[str]] = defaultdict(list)
    for row in deduped_rows:
        wikidata_to_place_hints[row["WikiData"]].append(row["place_hints"])

    canonical_by_wikidata = {
        wikidata: max(set(place_hints), key=len)
        for wikidata, place_hints in wikidata_to_place_hints.items()
    }
    for row in deduped_rows:
        row["place_hints"] = canonical_by_wikidata[row["WikiData"]]

    place_counts = Counter(row["place_hints"] for row in deduped_rows)
    counts = sorted(place_counts.values())
    bins = list(range(1, 125, 5))
    hist = []
    for start in bins[:-1]:
        end = start + 5
        hist.append(
            {
                "start": start,
                "end": end,
                "count": sum(1 for value in counts if start <= value < end),
            }
        )

    return {
        "raw_valid_rows": raw_valid_rows,
        "after_caption_dedup": len(deduped_rows),
        "unique_wikidata": len(wikidata_to_place_hints),
        "unique_place_hints": len(place_counts),
        "duplicates_total": len(deduped_rows) - len(place_counts),
        "max_duplicates": max(counts),
        "mean_duplicates": round(mean(counts), 2),
        "median_duplicates": float(median(counts)),
        "histogram": hist,
    }


def main() -> None:
    OUTPUT_PATH.write_text(
        json.dumps(build_distribution(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
