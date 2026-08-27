from types import SimpleNamespace

from retrieval import BM25Index, reciprocal_rank_fusion


def doc(text, document_id, chunk_index):
    return SimpleNamespace(
        page_content=text,
        metadata={"document_id": document_id, "chunk_index": chunk_index},
    )


def test_bm25_handles_terms():
    documents = [
        doc("deployment needs Redis", "a", 0),
        doc("users delete documents", "b", 0),
    ]
    assert BM25Index(documents).search("Redis", 1)[0][0] is documents[0]


def test_fusion_combines_channels():
    first, second = doc("alpha", "a", 0), doc("beta", "b", 0)
    results = reciprocal_rank_fusion([(first, 0.9), (second, 0.8)], [(second, 5)], 2)
    assert [item.document for item in results] == [second, first]
    assert results[0].channels == ("dense", "lexical")
