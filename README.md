# DocMind: Explainable Multi-Document RAG

DocMind is a local document question-answering system built with Flask, LangChain, Chroma and Zhipu AI. It supports PDF, DOCX, Markdown and TXT ingestion, incremental indexing, persistent storage, multi-turn conversations, hybrid retrieval and traceable citations.

## Core capabilities

- Incremental ingestion without rebuilding the knowledge base.
- Hybrid retrieval: Chroma dense similarity plus in-process BM25, fused with weighted reciprocal rank fusion.
- Conversational search: follow-up questions are rewritten into standalone queries before retrieval.
- Diversity ranking to avoid adjacent chunks dominating the context.
- Relevance threshold, citation-only prompt and prompt-injection defense.
- Source metadata: filename, page, chunk, score and retrieval channel.
- Environment configuration, safe filenames, upload limits, health endpoint and timing trace.
- `evaluate.py` computes Recall@K and MRR from a reproducible JSONL dataset.

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-dev.txt
copy .env.example .env
python app.py
```

Set `ZHIPU_API_KEY` in `.env`, then open `http://127.0.0.1:5000`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness and knowledge-base readiness |
| GET | `/api/documents` | Indexed document list |
| POST | `/api/documents` | Multi-file incremental indexing (`files`) |
| DELETE | `/api/documents/<id>` | Delete document and vectors |
| POST | `/api/chat` | Answer with `question` and `history` |

## Evaluation

Create a UTF-8 JSONL file with `query` and `relevant_chunk_ids` (`document_id:chunk_index`). Run `python evaluate.py eval.jsonl`; report Recall@K, MRR, citation hit rate and latency. Do not claim an accuracy percentage without a fixed dataset.

## Tests

```bash
pytest
ruff check .
```
