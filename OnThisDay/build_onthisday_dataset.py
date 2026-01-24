#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a day-level historical events dataset using Wikimedia "On this day" API.

Examples:
  python build_onthisday_dataset.py --start-year 1900 --end-year 2025 --out events.jsonl --format jsonl
  python build_onthisday_dataset.py --start-year 1950 --end-year 1960 --out events.csv --format csv --lang en
"""

import argparse
import csv
import json
import random
import time
from datetime import date
from typing import Dict, List, Tuple, Optional

import requests


API_URL_TEMPLATE = "https://api.wikimedia.org/feed/v1/wikipedia/{lang}/onthisday/all/{mm}/{dd}"


def month_day_iter():
    """Iterate over all valid month/day pairs for a non-leap year baseline (we query by mm/dd anyway)."""
    # Use 2001 as a non-leap year baseline for valid dates
    for m in range(1, 13):
        for d in range(1, 32):
            try:
                _ = date(2001, m, d)
            except ValueError:
                continue
            yield m, d


def request_with_retry(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    max_retries: int = 5,
    base_sleep: float = 0.8,
    timeout: int = 20,
) -> Optional[dict]:
    """GET JSON with retries (handles 429/5xx)."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                # backoff
                sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.3
                time.sleep(sleep_s)
                continue
            # other errors: print and stop retrying
            print(f"[WARN] HTTP {r.status_code} for {url}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.3
            print(f"[WARN] Request error ({e}) for {url}. sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)
    print(f"[ERROR] Failed after retries: {url}")
    return None


def one_sentence_event(item: dict) -> str:
    """
    Convert a Wikimedia onthisday 'event' item to a single-sentence string.
    Typical fields: 'year', 'text', 'pages' (optional).
    """
    year = item.get("year")
    text = (item.get("text") or "").strip()

    # (optional) add one page title as anchor if available
    pages = item.get("pages") or []
    title = None
    if pages:
        # pick first page title
        title = pages[0].get("normalizedtitle") or pages[0].get("title")

    if year is not None and text:
        if title and title.lower() not in text.lower():
            return f"{year}: {text} ({title})"
        return f"{year}: {text}"

    # fallback
    return text or json.dumps(item, ensure_ascii=False)


def filter_events_by_year(events: List[dict], start_year: int, end_year: int) -> List[dict]:
    """Keep only event items whose 'year' within [start_year, end_year]."""
    out = []
    for e in events:
        y = e.get("year")
        if isinstance(y, int) and start_year <= y <= end_year:
            out.append(e)
    return out


def build_records_for_day(payload: dict, mm: int, dd: int, start_year: int, end_year: int) -> List[dict]:
    """
    Build records for a specific month/day.
    Record schema:
      {
        "date": "YYYY-MM-DD",
        "month": mm,
        "day": dd,
        "event_year": yyyy,
        "event": "yyyy: one sentence ..."
      }
    """
    events = payload.get("events") or []
    events = filter_events_by_year(events, start_year, end_year)

    records = []
    for e in events:
        event_year = e.get("year")
        sentence = one_sentence_event(e)
        # Create a full date using the event year + mm/dd if valid. If invalid (Feb 29 non-leap), skip.
        try:
            full_date = date(event_year, mm, dd).isoformat()
        except Exception:
            # Most commonly Feb 29 for non-leap years
            continue
        records.append(
            {
                "date": full_date,
                "month": mm,
                "day": dd,
                "event_year": event_year,
                "event": sentence,
            }
        )
    return records


def write_jsonl(path: str, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, records: List[dict]):
    fieldnames = ["date", "month", "day", "event_year", "event"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, required=True, help="inclusive, e.g. 1900")
    ap.add_argument("--end-year", type=int, required=True, help="inclusive, e.g. 2025")
    ap.add_argument("--lang", type=str, default="en", help="wikipedia language, e.g. en, zh, ja")
    ap.add_argument("--out", type=str, required=True, help="output file path")
    ap.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    ap.add_argument("--sleep", type=float, default=0.25, help="sleep seconds between API calls")
    ap.add_argument(
        "--user-agent",
        type=str,
        default="onthisday-dataset-builder/1.0 (contact: youremail@example.com)",
        help="set a respectful UA string",
    )
    args = ap.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("start-year must be <= end-year")

    headers = {
        "User-Agent": args.user_agent,
        "Accept": "application/json",
    }

    session = requests.Session()
    all_records: List[dict] = []

    days = list(month_day_iter())
    total = len(days)

    for idx, (mm, dd) in enumerate(days, start=1):
        url = API_URL_TEMPLATE.format(lang=args.lang, mm=f"{mm:02d}", dd=f"{dd:02d}")
        payload = request_with_retry(session, url, headers=headers)
        if payload is None:
            print(f"[WARN] skip {mm:02d}-{dd:02d}")
            time.sleep(args.sleep)
            continue

        recs = build_records_for_day(payload, mm, dd, args.start_year, args.end_year)
        all_records.extend(recs)
        print(all_records[-1])

        print(f"[{idx}/{total}] {mm:02d}-{dd:02d} -> +{len(recs)} records (total={len(all_records)})")
        time.sleep(args.sleep)

    # sort for nice output
    all_records.sort(key=lambda r: (r["date"], r["event"]))

    if args.format == "jsonl":
        write_jsonl(args.out, all_records)
    else:
        write_csv(args.out, all_records)

    print(f"[DONE] wrote {len(all_records)} records to {args.out}")


if __name__ == "__main__":
    main()
