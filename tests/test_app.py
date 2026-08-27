import io

import pytest

from app import create_app


class FakeRAG:
    def __init__(self):
        self.items = []
        self.last_history = None

    @property
    def ready(self):
        return bool(self.items)

    def list_documents(self):
        return self.items

    def ingest(self, _path, name, document_id):
        item = {"id": document_id, "name": name, "chunks": 2}
        self.items.append(item)
        return item

    def delete_document(self, document_id):
        old = len(self.items)
        self.items = [item for item in self.items if item["id"] != document_id]
        return len(self.items) != old

    def answer(self, _question, history):
        self.last_history = history
        return {"answer": "answer [1]", "sources": [], "grounded": True}


class RejectingRAG(FakeRAG):
    def ingest(self, _path, _name, _document_id):
        raise ValueError("no text")


@pytest.fixture()
def fake():
    return FakeRAG()


@pytest.fixture()
def client(tmp_path, fake):
    return create_app(
        {"TESTING": True, "UPLOAD_FOLDER": tmp_path / "uploads"}, fake
    ).test_client()


def test_health_reports_readiness(client):
    assert client.get("/api/health").get_json()["ready"] is False


def test_browser_support_requests_do_not_become_server_errors(client):
    assert client.get("/favicon.ico").status_code == 204
    assert client.get("/missing-page").status_code == 404


def test_upload_chat_and_delete(client, fake):
    response = client.post(
        "/api/documents", data={"files": (io.BytesIO(b"knowledge"), "guide.txt")}
    )
    assert response.status_code == 201
    document_id = response.get_json()["documents"][0]["id"]
    response = client.post(
        "/api/chat",
        json={
            "question": "What?",
            "history": [{"role": "user", "content": "Previous"}],
        },
    )
    assert response.status_code == 200 and fake.last_history[0]["content"] == "Previous"
    assert client.delete(f"/api/documents/{document_id}").status_code == 204


def test_rejects_unsupported_extension(client):
    assert (
        client.post(
            "/api/documents", data={"files": (io.BytesIO(b"x"), "script.exe")}
        ).status_code
        == 415
    )


def test_reports_unreadable_documents(tmp_path):
    application = create_app(
        {"TESTING": True, "UPLOAD_FOLDER": tmp_path / "uploads"}, RejectingRAG()
    )
    response = application.test_client().post(
        "/api/documents", data={"files": (io.BytesIO(b"scan"), "scan.pdf")}
    )
    assert response.status_code == 422
    assert "Cannot extract text" in response.get_json()["error"]


def test_validates_chat_payload(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 400
    assert (
        client.post("/api/chat", json={"question": "ok", "history": "bad"}).status_code
        == 400
    )
