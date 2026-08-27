import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from rag_service import SUPPORTED_EXTENSIONS, RAGService

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config=None, rag_service=None):
    app = Flask(__name__)
    app.config.update(
        UPLOAD_FOLDER=os.getenv("UPLOAD_FOLDER", "data/uploads"),
        CHROMA_FOLDER=os.getenv("CHROMA_FOLDER", "data/chroma"),
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
        TOP_K=int(os.getenv("TOP_K", "4")),
        MAX_HISTORY_TURNS=int(os.getenv("MAX_HISTORY_TURNS", "6")),
    )
    if config:
        app.config.update(config)
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    service = rag_service or RAGService(
        app.config["CHROMA_FOLDER"],
        os.getenv("ZHIPU_API_KEY", ""),
        app.config["TOP_K"],
        app.config["MAX_HISTORY_TURNS"],
    )
    app.extensions["rag_service"] = service

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/api/health")
    def health():
        return jsonify(
            status="ok", ready=service.ready, documents=service.list_documents()
        )

    @app.get("/api/documents")
    def documents():
        return jsonify(documents=service.list_documents())

    @app.post("/api/documents")
    def upload_documents():
        files = request.files.getlist("files")
        if not files or all(not item.filename for item in files):
            return jsonify(error="Select at least one document."), 400
        uploaded = []
        for item in files:
            original_name = item.filename or ""
            extension = Path(original_name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                return jsonify(error=f"Unsupported file: {original_name}"), 415
            document_id = uuid.uuid4().hex
            safe_name = secure_filename(original_name) or f"document{extension}"
            path = upload_dir / f"{document_id}_{safe_name}"
            item.save(path)
            try:
                uploaded.append(service.ingest(path, original_name, document_id))
            except ValueError as error:
                path.unlink(missing_ok=True)
                logger.info("Document rejected: %s: %s", original_name, error)
                return jsonify(
                    error=(
                        f"Cannot extract text from {original_name}. "
                        "The file may be empty, encrypted, damaged, or a scanned image."
                    )
                ), 422
            except UnicodeError:
                path.unlink(missing_ok=True)
                logger.info("Document encoding is unsupported: %s", original_name)
                return jsonify(
                    error=f"Cannot read {original_name}. Save the text file as UTF-8 and retry."
                ), 422
            except Exception as error:
                path.unlink(missing_ok=True)
                logger.exception("Document processing failed: %s", original_name)
                error_name = type(error).__name__
                return jsonify(
                    error=f"Failed to process {original_name} ({error_name}). Check the server terminal for details."
                ), 502
        return jsonify(
            message=f"Indexed {len(uploaded)} document(s).", documents=uploaded
        ), 201

    @app.delete("/api/documents/<document_id>")
    def delete_document(document_id):
        if not service.delete_document(document_id):
            return jsonify(error="Document not found."), 404
        for path in upload_dir.glob(f"{document_id}_*"):
            path.unlink(missing_ok=True)
        return "", 204

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        history = payload.get("history", [])
        if not question or len(question) > 2000:
            return jsonify(error="Question must contain 1-2000 characters."), 400
        if not isinstance(history, list):
            return jsonify(error="History must be an array."), 400
        try:
            return jsonify(service.answer(question, history))
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        size = app.config["MAX_CONTENT_LENGTH"] // 1024 // 1024
        return jsonify(error=f"Uploads may not exceed {size} MB."), 413

    @app.errorhandler(Exception)
    def unexpected_error(error):
        if isinstance(error, HTTPException):
            return error
        logger.exception("Unhandled request error")
        if app.config.get("TESTING"):
            raise error
        return jsonify(error="Service temporarily unavailable."), 500

    return app


if __name__ == "__main__":
    create_app().run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
