import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


def tokenize(text):
    """Tokenize English words and Chinese unigrams/bigrams without extra dependencies."""
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    chinese = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index : index + 2] for index in range(len(run) - 1))
    return words + chinese


def chunk_key(metadata):
    return f"{metadata.get('document_id', '')}:{metadata.get('chunk_index', '')}"


@dataclass
class RankedChunk:
    document: object
    score: float
    channels: tuple
    dense_score: float | None = None
    lexical_score: float | None = None


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(doc.page_content) for doc in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(documents)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query, limit):
        query_tokens = tokenize(query)
        scores = []
        for index, tokens in enumerate(self.tokens):
            frequencies = Counter(tokens)
            length_ratio = self.lengths[index] / max(self.average_length, 1)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self.idf.get(token, 0) * numerator / denominator
            if score > 0:
                scores.append((self.documents[index], score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]


def reciprocal_rank_fusion(dense_results, lexical_results, top_k, rrf_k=60):
    """Fuse independent rankings, then discourage adjacent chunks from dominating."""
    documents = {}
    fused = defaultdict(float)
    channel_scores = defaultdict(dict)
    weights = {"dense": 0.65, "lexical": 0.35}
    for channel, results in (("dense", dense_results), ("lexical", lexical_results)):
        for rank, (document, raw_score) in enumerate(results, 1):
            key = chunk_key(document.metadata)
            documents[key] = document
            fused[key] += weights[channel] / (rrf_k + rank)
            channel_scores[key][channel] = float(raw_score)
    if not fused:
        return []
    maximum = max(fused.values())
    ranked = sorted(fused, key=fused.get, reverse=True)
    selected, deferred = [], []
    for key in ranked:
        document = documents[key]
        metadata = document.metadata
        adjacent = any(
            item.document.metadata.get("document_id") == metadata.get("document_id")
            and abs(
                int(item.document.metadata.get("chunk_index", -99))
                - int(metadata.get("chunk_index", 99))
            )
            <= 1
            for item in selected
        )
        channels = channel_scores[key]
        item = RankedChunk(
            document=document,
            score=fused[key] / maximum,
            channels=tuple(sorted(channels)),
            dense_score=channels.get("dense"),
            lexical_score=channels.get("lexical"),
        )
        (deferred if adjacent else selected).append(item)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        selected.extend(deferred[: top_k - len(selected)])
    return selected
