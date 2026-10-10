"""Connection tests must not accept a reply from an unrelated model on the same URL."""

import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app.db import Database
from app.main import create_app
from app.secretbox import SecretBox
from app.settings import load_settings


@pytest.mark.parametrize("available,expected_ok", [
    (["GLM-5.3-Flash-EXL3"], False),
    (["deepseek-v4-flash-0731"], True),
    (None, False),
])
def test_connection_rejects_unavailable_model_before_chat(tmp_path, available, expected_ok):
    requests = []

    class FakeProvider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_json(self, status, data):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            requests.append(("GET", self.path, self.headers.get("Authorization")))
            if available is None:
                self.send_json(404, {"error": "model list unavailable"})
            else:
                self.send_json(200, {"data": [{"id": name} for name in available]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(("POST", self.path, self.headers.get("Authorization"), body["model"]))
            self.send_json(200, {"model": (available or ["GLM-5.3-Flash-EXL3"])[0], "choices": [{"message": {"content": "번역 결과"}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = replace(load_settings(), data_dir=tmp_path, host="127.0.0.1", password=None)
        db = Database(settings.db_path)
        db.upsert_provider({
            "id": "stale", "name": "stale", "kind": "openai_compatible", "auth": "api_key",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "deepseek-v4-flash-0731",
            "extra_body": {}, "vision": False,
            "secret": SecretBox(settings.data_dir / "secret.key").seal({"api_key": "fixture-key"}),
            "account": None, "created_at": time.time(),
        })
        with TestClient(create_app(settings)) as app:
            result = app.post("/api/providers/stale/test")
        assert result.status_code == 200
        assert result.json()["ok"] is expected_ok
        assert requests[0] == ("GET", "/v1/models", "Bearer fixture-key")
        if expected_ok:
            assert result.json()["reply"] == "번역 결과"
            assert requests[1] == ("POST", "/v1/chat/completions", "Bearer fixture-key", "deepseek-v4-flash-0731")
        else:
            assert "reply" not in result.json()
            assert "모델" in result.json()["error"]
            assert len(requests) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
