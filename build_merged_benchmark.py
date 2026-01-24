#!/usr/bin/env python3
import argparse
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

EVENT_TYPE_PRIORITY = {
    "start": 0,
    "end": 1,
}

QA_SUB_ID_MAP = {
    "generic": 0,
    "rephrased_1": 1,
    "rephrased_2": 2,
    "rephrased_3": 3,
}

MONTH_MAP = {
    "January": "01",
    "February": "02",
    "March": "03",
    "April": "04",
    "May": "05",
    "June": "06",
    "July": "07",
    "August": "08",
    "September": "09",
    "October": "10",
    "November": "11",
    "December": "12",
}


@dataclass(frozen=True)
class ParsedAnswer:
    name: str
    start: str
    end: Optional[str]


def parse_answer_span(answer: str) -> ParsedAnswer:
    parts = [part.strip() for part in answer.split("|")]
    name = parts[0]
    start = None
    end = None
    for part in parts[1:]:
        if part.startswith("S:"):
            start = part.replace("S:", "", 1).strip()
        elif part.startswith("E:"):
            end = part.replace("E:", "", 1).strip()
    if start is None:
        raise ValueError(f"Missing start date in answer: {answer}")
    return ParsedAnswer(name=name, start=start, end=end)


def normalize_date(date_str: str) -> str:
    cleaned = date_str.lstrip("+")
    return cleaned.split("T")[0]


def build_role(category: str, element: str, attribute: Optional[str], answer_name: str) -> str:
    if category == "countries_byGDP":
        return attribute
    if category == "organizations":
        return f"{attribute} of {element}"
    if category == "companies_byRevenue":
        return f"Chief Executive Officer of {element}"
    if category == "athletes_byPayment":
        return f"played for {answer_name}"
    raise ValueError(f"Unknown category: {category}")


def build_fact_text(
    category: str,
    element: str,
    attribute: Optional[str],
    answer_name: str,
    role: str,
    date: str,
    event_type: str,
) -> str:
    if category == "athletes_byPayment":
        subject = element
        if event_type == "start":
            return f"{subject} {role} on {date}."
        return f"{subject} stopped playing for {answer_name} on {date}."

    subject = answer_name
    if event_type == "start":
        return f"{subject} served as {role} on {date}."
    return f"{subject} ceased serving as {role} on {date}."


def iter_entries(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for category, elements in data.items():
        for element, payload in elements.items():
            if category in {"countries_byGDP", "organizations"}:
                for attribute, entry in payload.items():
                    yield {
                        "category": category,
                        "element": element,
                        "attribute": attribute,
                        "entry": entry,
                    }
            else:
                yield {
                    "category": category,
                    "element": element,
                    "attribute": None,
                    "entry": payload,
                }


def sort_key_for_date(date_str: str, event_type: str) -> tuple:
    if "00" in date_str:
        base_date = datetime.strptime(f"{date_str[:4]}-01-01", "%Y-%m-%d")
    else:
        base_date = datetime.strptime(date_str, "%Y-%m-%d")
    return (base_date, EVENT_TYPE_PRIORITY[event_type])


def build_event_stream(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for item in iter_entries(data):
        category = item["category"]
        element = item["element"]
        attribute = item["attribute"]
        answers = item["entry"].get("answers", [])
        for answer in answers:
            parsed = parse_answer_span(answer)
            start_date = normalize_date(parsed.start)
            role = build_role(category, element, attribute, parsed.name)
            facts.append(
                {
                    "text": build_fact_text(
                        category,
                        element,
                        attribute,
                        parsed.name,
                        role,
                        start_date,
                        "start",
                    ),
                    "metadata": {
                        "category": category,
                        "element": element,
                        "attribute": attribute,
                        "answer": parsed.name,
                        "role": role,
                        "date": start_date,
                        "event_type": "start",
                    },
                }
            )
            if parsed.end:
                end_date = normalize_date(parsed.end)
                facts.append(
                    {
                        "text": build_fact_text(
                            category,
                            element,
                            attribute,
                            parsed.name,
                            role,
                            end_date,
                            "end",
                        ),
                        "metadata": {
                            "category": category,
                            "element": element,
                            "attribute": attribute,
                            "answer": parsed.name,
                            "role": role,
                            "date": end_date,
                            "event_type": "end",
                        },
                    }
                )
    facts.sort(key=lambda item: sort_key_for_date(item["metadata"]["date"], item["metadata"]["event_type"]))
    return facts


def load_on_this_day_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            events.append(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": payload["date"],
                    "content": payload["event"],
                    "prev_event_ids": [],
                    "next_event_ids": [],
                    "metadata": {
                        "source": "on_this_day",
                        "event_year": payload.get("event_year"),
                        "month": payload.get("month"),
                        "day": payload.get("day"),
                    },
                }
            )
    return events


def build_reasoning_events(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in iter_entries(data):
        category = item["category"]
        element = item["element"]
        attribute = item["attribute"]
        answers = item["entry"].get("answers", [])
        parsed_answers = []
        for answer in answers:
            parsed = parse_answer_span(answer)
            parsed_answers.append(
                {
                    "parsed": parsed,
                    "start_date": normalize_date(parsed.start),
                    "end_date": normalize_date(parsed.end) if parsed.end else None,
                }
            )
        parsed_answers.sort(key=lambda item: sort_key_for_date(item["start_date"], "start"))

        item_entries: List[Dict[str, Any]] = []
        for parsed_item in parsed_answers:
            parsed = parsed_item["parsed"]
            start_date = parsed_item["start_date"]
            end_date = parsed_item["end_date"]
            role = build_role(category, element, attribute, parsed.name)
            item_entries.append(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": start_date,
                    "content": build_fact_text(
                        category,
                        element,
                        attribute,
                        parsed.name,
                        role,
                        start_date,
                        "start",
                    ),
                    "prev_event_ids": [],
                    "next_event_ids": [],
                    "metadata": {
                        "category": category,
                        "element": element,
                        "attribute": attribute,
                        "answer": parsed.name,
                        "role": role,
                        "date": start_date,
                        "event_type": "start",
                        "source": "reasoning_event_stream",
                    },
                }
            )
            if end_date:
                item_entries.append(
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": end_date,
                        "content": build_fact_text(
                            category,
                            element,
                            attribute,
                            parsed.name,
                            role,
                            end_date,
                            "end",
                        ),
                        "prev_event_ids": [],
                        "next_event_ids": [],
                        "metadata": {
                            "category": category,
                            "element": element,
                            "attribute": attribute,
                            "answer": parsed.name,
                            "role": role,
                            "date": end_date,
                            "event_type": "end",
                            "source": "reasoning_event_stream",
                        },
                    }
                )

        for index, entry in enumerate(item_entries):
            if index > 0:
                entry["prev_event_ids"].append(item_entries[index - 1]["id"])
            if index < len(item_entries) - 1:
                entry["next_event_ids"].append(item_entries[index + 1]["id"])
        entries.extend(item_entries)

    return entries


def parse_question_date(question: str) -> Optional[str]:
    match = re.search(r"On\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", question)
    if not match:
        return None
    day = int(match.group(1))
    month_name = match.group(2)
    year = match.group(3)
    month = MONTH_MAP.get(month_name)
    if not month:
        return None
    return f"{year}-{month}-{day:02d}"


def build_qa_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    qa_entries: List[Dict[str, Any]] = []
    for item in iter_entries(data):
        category = item["category"]
        element = item["element"]
        attribute = item["attribute"]
        entry = item["entry"]
        ground_truth = entry.get("ground_truth")
        for qa_type in ("ranking_qa", "accumulate_qa"):
            qa_block = entry.get(qa_type)
            if not qa_block:
                continue
            question_id = "::".join(
                part
                for part in [category, element, attribute or "", qa_type]
                if part != ""
            )
            for key, question in qa_block.items():
                sub_id = QA_SUB_ID_MAP.get(key)
                if sub_id is None:
                    continue
                reference_date = parse_question_date(question)
                qa_entries.append(
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": reference_date,
                        "content": question,
                        "prev_event_ids": [],
                        "next_event_ids": [],
                        "metadata": {
                            "source": "reasoning_qa",
                            "question_id": question_id,
                            "sub_id": sub_id,
                            "qa_type": qa_type,
                            "category": category,
                            "element": element,
                            "attribute": attribute,
                            "answer": ground_truth,
                        },
                    }
                )
    return qa_entries


def sort_event_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(entry: Dict[str, Any]) -> tuple:
        timestamp = entry.get("timestamp")
        if not timestamp:
            return (1, datetime.max, 0)
        event_type = entry.get("metadata", {}).get("event_type", "start")
        if "00" in timestamp:
            base_date = datetime.strptime(f"{timestamp[:4]}-01-01", "%Y-%m-%d")
        else:
            base_date = datetime.strptime(timestamp, "%Y-%m-%d")
        priority = EVENT_TYPE_PRIORITY.get(event_type, 0)
        return (0, base_date, priority)

    return sorted(entries, key=sort_key)


def sort_qa_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(entry: Dict[str, Any]) -> tuple:
        timestamp = entry.get("timestamp")
        if not timestamp:
            return (1, datetime.max, "", 0)
        base_date = datetime.strptime(timestamp, "%Y-%m-%d")
        metadata = entry.get("metadata", {})
        return (0, base_date, metadata.get("question_id", ""), metadata.get("sub_id", 0))

    return sorted(entries, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge OnThisDay events with reasoning event streams and QA pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--on-this-day",
        default="OnThisDay/events.jsonl",
        help="Path to the OnThisDay events.jsonl file.",
    )
    parser.add_argument(
        "--reasoning-qa",
        default="Reasoning/reasoning_qa_20250210.json",
        help="Path to the reasoning QA JSON file.",
    )
    parser.add_argument(
        "--output",
        default="merged_benchmark.json",
        help="Destination path for the merged benchmark JSON file.",
    )
    args = parser.parse_args()

    on_this_day_path = Path(args.on_this_day)
    reasoning_path = Path(args.reasoning_qa)
    output_path = Path(args.output)

    with reasoning_path.open("r", encoding="utf-8") as handle:
        reasoning_data = json.load(handle)

    on_this_day_entries = load_on_this_day_events(on_this_day_path)
    reasoning_event_entries = build_reasoning_events(reasoning_data)
    qa_entries = build_qa_entries(reasoning_data)

    merged = {
        "event": sort_event_entries(on_this_day_entries + reasoning_event_entries),
        "qa": sort_qa_entries(qa_entries),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
