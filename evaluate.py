"""Evaluate retrieval against a JSONL file of {query, relevant_chunk_ids}."""

import argparse
import json
from pathlib import Path

from retrieval import chunk_key


def evaluate(service, dataset):
    rows = [
        json.loads(line)
        for line in Path(dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hits, reciprocal, total = 0, 0.0, 0
    for row in rows:
        ranked, _trace = service.retrieve(row["query"])
        expected = set(row.get("relevant_chunk_ids", []))
        keys = [chunk_key(item.document.metadata) for item in ranked]
        if any(key in expected for key in keys):
            hits += 1
            reciprocal += 1 / (
                next(index for index, key in enumerate(keys, 1) if key in expected)
            )
        total += 1
    return {
        "queries": total,
        "recall_at_k": round(hits / total, 4) if total else 0,
        "mrr": round(reciprocal / total, 4) if total else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSONL evaluation dataset")
    args = parser.parse_args()
    from app import create_app

    application = create_app()
    print(
        json.dumps(
            evaluate(application.extensions["rag_service"], args.dataset),
            ensure_ascii=False,
            indent=2,
        )
    )
