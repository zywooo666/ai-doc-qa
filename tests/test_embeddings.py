from types import SimpleNamespace

from rag_service import ZhipuEmbeddings


class FakeEmbeddingEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(text)]) for text in input]
        )


def test_document_embeddings_are_sent_in_bounded_batches():
    endpoint = FakeEmbeddingEndpoint()
    embeddings = ZhipuEmbeddings.__new__(ZhipuEmbeddings)
    embeddings.client = SimpleNamespace(embeddings=endpoint)
    embeddings.batch_size = 16

    result = embeddings.embed_documents([str(index) for index in range(35)])

    assert [len(call[1]) for call in endpoint.calls] == [16, 16, 3]
    assert len(result) == 35
    assert result[17] == [17.0]
