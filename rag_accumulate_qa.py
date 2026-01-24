import argparse
import json
import os
import time
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_TEMPLATE = (
    "Use the provided context to answer the question. "
    "Your answer must contain only the name, with no other words. "
    "If the answer is not present in the context, reply with 'Unknown'.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "Your answer:"
)


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m rag_accumulate_qa",
        description="RAG pipeline for Reasoning accumulate_qa using ChromaDB and DeepSeek.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--event-stream",
        type=str,
        default="Reasoning/event_stream.json",
        help="Path to event_stream.json.",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default="Reasoning/reasoning_qa_20250210.json",
        help="Path to accumulate_qa questions.",
    )
    parser.add_argument(
        "--task-data",
        type=str,
        default="Reasoning/reasoning_task_data.json",
        help="Path to reasoning task data with ground truth.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Reasoning/results/rag_accumulate_qa",
        help="Output directory for answers and metrics.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="Reasoning/.chroma_reasoning",
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="reasoning_event_stream",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of retrieved chunks per question.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="Model name for DeepSeek.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens for model completion.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep between API calls in seconds.",
    )
    return parser.parse_args()


def load_event_stream(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_or_load_collection(
    event_stream: List[Dict[str, object]],
    persist_dir: str,
    collection_name: str,
):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="../all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    if collection.count() == 0:
        documents = [item["text"] for item in event_stream]
        metadatas = [sanitize_metadata(item.get("metadata")) for item in event_stream]
        ids = [str(i) for i in range(len(documents))]
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection
def sanitize_metadata(metadata: Optional[object]) -> Dict[str, object]:
    if not isinstance(metadata, dict):
        return {}
    return {key: value for key, value in metadata.items() if value is not None}

def format_context(documents: List[str], metadatas: List[Dict[str, object]]) -> str:
    lines = []
    for idx, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        meta_bits = []
        for key in ("category", "element", "attribute", "answer", "date", "event_type"):
            if key in metadata and metadata[key] is not None:
                meta_bits.append(f"{key}={metadata[key]}")
        meta_str = "; ".join(meta_bits)
        if meta_str:
            lines.append(f"[{idx}] {doc} ({meta_str})")
        else:
            lines.append(f"[{idx}] {doc}")
    return "\n".join(lines)


def build_messages(context: str, question: str) -> List[Dict[str, str]]:
    user_prompt = DEFAULT_USER_TEMPLATE.format(context=context, question=question)
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_model(client: OpenAI, model: str, messages: List[Dict[str, str]], max_tokens: int) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().split()).lower()


def load_ground_truth(task_data: dict, category: str, element: str, attribute: Optional[str]) -> str:
    if category in ["countries_byGDP", "organizations"]:
        return task_data[category][element][attribute]["task_accumulate"]["ground_truth"]
    return task_data[category][element]["task_accumulate"]["ground_truth"]


def generate_for_questions(
    questions: Dict[str, str],
    collection,
    client: OpenAI,
    args: Namespace,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    answers = {}
    contexts = {}
    prompts = {}
    for qt, question in questions.items():
        retrieval = collection.query(
            query_texts=[question],
            n_results=args.top_k,
        )
        documents = retrieval["documents"][0]
        metadatas = retrieval["metadatas"][0]
        context = format_context(documents, metadatas)
        messages = build_messages(context, question)
        prompts[qt] = messages
        contexts[qt] = context
        answers[qt] = call_model(client, args.model, messages, args.max_tokens)
        if args.sleep:
            time.sleep(args.sleep)
    return prompts, contexts, answers


def main() -> None:
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    event_stream = load_event_stream(args.event_stream)
    collection = build_or_load_collection(event_stream, args.persist_dir, args.collection_name)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(args.qa_file, "r") as f:
        qa_data = json.load(f)
    with open(args.task_data, "r") as f:
        task_data = json.load(f)

    outputs = {}
    metrics = {
        "model": args.model,
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "by_question_type": {},
    }

    for category in qa_data:
        if category not in outputs:
            outputs[category] = {}
        for element in tqdm(qa_data[category], desc=category):
            if category in ["countries_byGDP", "organizations"]:
                if element not in outputs[category]:
                    outputs[category][element] = {}
                for attribute, payload in qa_data[category][element].items():
                    questions = payload["accumulate_qa"]
                    prompts, contexts, answers = generate_for_questions(
                        questions, collection, client, args
                    )
                    ground_truth = load_ground_truth(task_data, category, element, attribute)
                    outputs[category][element][attribute] = {
                        "questions": prompts,
                        "contexts": contexts,
                        "answers": answers,
                        "ground_truth": ground_truth,
                    }
                    for qt, answer in answers.items():
                        metrics["total"] += 1
                        metrics.setdefault("by_question_type", {}).setdefault(qt, {
                            "total": 0,
                            "correct": 0,
                        })
                        metrics["by_question_type"][qt]["total"] += 1
                        if normalize_answer(answer) == normalize_answer(ground_truth):
                            metrics["correct"] += 1
                            metrics["by_question_type"][qt]["correct"] += 1
            else:
                questions = qa_data[category][element]["accumulate_qa"]
                prompts, contexts, answers = generate_for_questions(
                    questions, collection, client, args
                )
                outputs[category][element] = {
                    "questions": prompts,
                    "contexts": contexts,
                    "answers": answers,
                }
                ground_truth = load_ground_truth(task_data, category, element, None)
                for qt, answer in answers.items():
                    metrics["total"] += 1
                    metrics.setdefault("by_question_type", {}).setdefault(qt, {
                        "total": 0,
                        "correct": 0,
                    })
                    metrics["by_question_type"][qt]["total"] += 1
                    if normalize_answer(answer) == normalize_answer(ground_truth):
                        metrics["correct"] += 1
                        metrics["by_question_type"][qt]["correct"] += 1

    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] else 0.0
    for qt, stats in metrics["by_question_type"].items():
        total = stats["total"]
        stats["accuracy"] = stats["correct"] / total if total else 0.0

    model_dir = os.path.join(args.out_dir, args.model.replace("/", "_"), f"_top{args.top_k}")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "answers.json"), "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
