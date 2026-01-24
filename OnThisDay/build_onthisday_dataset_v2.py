#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a day-level historical events dataset using Wikimedia "On this day" API
with resume + daily checkpointing.

Examples:
  python build_onthisday_dataset_resume.py --start-year 1900 --end-year 2025 --out events.jsonl --format jsonl
  python build_onthisday_dataset_resume.py --start-year 1950 --end-year 1960 --out events.csv --format csv --lang en
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import requests


API_URL_TEMPLATE = "https://api.wikimedia.org/feed/v1/wikipedia/{lang}/onthisday/all/{mm}/{dd}"


def month_day_iter():
    """Iterate over all valid month/day pairs for a non-leap year baseline."""
    for m in range(1, 13):
        for d in range(1, 32):
            try:
                _ = date(2001, m, d)  # non-leap year baseline
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
                sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.3
                time.sleep(sleep_s)
                continue
            print(f"[WARN] HTTP {r.status_code} for {url}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.3
            print(f"[WARN] Request error ({e}) for {url}. sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)
    print(f"[ERROR] Failed after retries: {url}")
    return None


def one_sentence_event(item: dict) -> str:
    """Convert a Wikimedia onthisday 'event' item to a single-sentence string."""
    year = item.get("year")
    text = (item.get("text") or "").strip()

    pages = item.get("pages") or []
    title = None
    if pages:
        title = pages[0].get("normalizedtitle") or pages[0].get("title")

    # if year is not None and text:
    #     if title and title.lower() not in text.lower():
    #         return f"{year}: {text} ({title})"
    #     return f"{year}: {text}"
    return text or json.dumps(item, ensure_ascii=False)


def filter_events_by_year(events: List[dict], start_year: int, end_year: int) -> List[dict]:
    out = []
    for e in events:
        y = e.get("year")
        if isinstance(y, int) and start_year <= y <= end_year:
            out.append(e)
    return out


def build_records_for_day(payload: dict, mm: int, dd: int, start_year: int, end_year: int) -> List[dict]:
    """
    Record schema:
      {
        "date": "YYYY-MM-DD",
        "month": mm,
        "day": dd,
        "event_year": yyyy,
        "event": "yyyy: one sentence ..."
      }
    """
    events = filter_events_by_year(payload.get("events") or [], start_year, end_year)

    records = []
    for e in events:
        event_year = e.get("year")
        sentence = one_sentence_event(e)

        # For Feb 29 on non-leap years, skip
        try:
            full_date = date(event_year, mm, dd).isoformat()
        except Exception:
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


# -------------------------
# Resume helpers
# -------------------------

def load_done_dates_jsonl(path: str) -> Set[str]:
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                d = obj.get("date")
                if isinstance(d, str) and len(d) == 10:
                    done.add(d)
            except json.JSONDecodeError:
                # ignore partial/corrupted line (e.g., crash mid-write)
                continue
    return done


def load_done_dates_csv(path: str) -> Set[str]:
    done = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("date")
            if isinstance(d, str) and len(d) == 10:
                done.add(d)
    return done


def load_done_dates(path: str, fmt: str) -> Set[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    if fmt == "jsonl":
        return load_done_dates_jsonl(path)
    return load_done_dates_csv(path)


def append_jsonl(path: str, records: List[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_csv(path: str, records: List[dict]):
    fieldnames = ["date", "month", "day", "event_year", "event"]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())


def append_records(path: str, fmt: str, records: List[dict]):
    if not records:
        return
    # stable order within a batch
    records.sort(key=lambda r: (r["date"], r["event"]))
    if fmt == "jsonl":
        append_jsonl(path, records)
    else:
        append_csv(path, records)


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
        default="onthisday-dataset-builder/1.1 (contact: youremail@example.com)",
        help="set a respectful UA string",
    )
    args = ap.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("start-year must be <= end-year")

    headers = {
        "User-Agent": args.user_agent,
        "Accept": "application/json",
    }

    # Resume: load already written dates
    done_dates = load_done_dates(args.out, args.format)
    print(f"[RESUME] loaded {len(done_dates)} done dates from {args.out}")

    session = requests.Session()
    days = list(month_day_iter())
    total = len(days)

    # For each mm-dd, fetch once, then write all years' events for that day
    for idx, (mm, dd) in enumerate(days, start=1):
        # Decide whether this mm-dd is fully done for the whole year range.
        # We consider it done if ALL dates within [start_year, end_year] for this mm-dd are present.
        expected_dates = []
        for y in range(args.start_year, args.end_year + 1):
            try:
                expected_dates.append(date(y, mm, dd).isoformat())
            except Exception:
                continue  # Feb 29 for non-leap year, etc.

        if expected_dates and all(d in done_dates for d in expected_dates):
            print(f"[{idx}/{total}] {mm:02d}-{dd:02d} already done -> skip")
            continue

        url = API_URL_TEMPLATE.format(lang=args.lang, mm=f"{mm:02d}", dd=f"{dd:02d}")
        payload = request_with_retry(session, url, headers=headers)
        if payload is None:
            print(f"[WARN] skip fetch {mm:02d}-{dd:02d}")
            time.sleep(args.sleep)
            continue

        # build records for this day across years, then filter out already written ones
        recs = build_records_for_day(payload, mm, dd, args.start_year, args.end_year)
        recs = [r for r in recs if r["date"] not in done_dates]

        append_records(args.out, args.format, recs)

        # update done_dates for resume safety
        for r in recs:
            done_dates.add(r["date"])

        print(f"[{idx}/{total}] {mm:02d}-{dd:02d} -> wrote +{len(recs)} records (done={len(done_dates)})")
        time.sleep(args.sleep)

    print(f"[DONE] output: {args.out} (unique dates written={len(done_dates)})")


if __name__ == "__main__":
    main()

