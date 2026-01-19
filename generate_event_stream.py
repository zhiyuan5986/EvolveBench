import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EVENT_TYPE_PRIORITY = {
    "start": 0,
    "end": 1,
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
    # if category in {"countries_byGDP", "organizations"}:
    #     if not attribute:
    #         raise ValueError(f"Missing attribute for category {category} element {element}")
    #     return f"{attribute} of {element}"
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


def build_event_facts(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for item in iter_entries(data):
        category = item["category"]
        element = item["element"]
        attribute = item["attribute"]
        answers = item["entry"]["answers"]
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
    facts.sort(
        key=lambda item: (
            datetime.strptime(item["metadata"]["date"], "%Y-%m-%d") if "00" not in item["metadata"]["date"] else datetime.strptime(item["metadata"]["date"][:4]+"-01-01", "%Y-%m-%d"),
            EVENT_TYPE_PRIORITY[item["metadata"]["event_type"]],
        )
    )
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a time-ordered event stream from QA answer lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="Reasoning/reasoning_qa_20250210.json",
        help="Path to a QA JSON that contains answer lists.",
    )
    parser.add_argument(
        "--output",
        default="event_stream.json",
        help="Destination path for the generated event stream JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    facts = build_event_facts(data)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(facts, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
