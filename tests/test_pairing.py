import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import pairing as pairing_module
from app.db import Database
from app.main import create_app
from app.settings import load_settings


@pytest.fixture
def server(tmp_path):
    settings = replace(load_settings(), data_dir=tmp_path, host="127.0.0.1", password="pw")
    with TestClient(create_app(settings)) as client:
        yield client, tmp_path


def login(client):
    client.cookies.clear()
    assert client.post("/api/login", json={"password": "pw"}).status_code == 200


def pair(client, name="Chrome · Windows"):
    client.cookies.clear()
    started = client.post("/api/pair/start", json={"name": name}).json()
    login(client)
    pending = client.get("/api/pair/requests").json()
    assert [(r["name"], r["code"]) for r in pending] == [(name, started["code"])]
    assert client.post(f"/api/pair/requests/{started['id']}/approve").status_code == 200
    client.cookies.clear()
    result = client.post("/api/pair/poll", json={"id": started["id"], "secret": started["secret"]}).json()
    assert result["status"] == "approved"
    return result["token"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_approved_request_hands_the_token_once_and_only_to_the_requester(server):
    client, _ = server
    started = client.post("/api/pair/start", json={"name": "Chrome"}).json()
    assert client.post("/api/pair/poll", json={"id": started["id"], "secret": started["secret"]}).json()["status"] == "pending"
    login(client)
    client.post(f"/api/pair/requests/{started['id']}/approve")
    client.cookies.clear()
    assert client.post("/api/pair/poll", json={"id": started["id"], "secret": "wrong"}).json() == {"status": "expired"}
    first = client.post("/api/pair/poll", json={"id": started["id"], "secret": started["secret"]}).json()
    assert first["status"] == "approved" and first["token"].startswith("ct_")
    assert client.post("/api/pair/poll", json={"id": started["id"], "secret": started["secret"]}).json() == {"status": "expired"}
    session = client.get("/api/session", headers=bearer(first["token"])).json()
    assert session["authenticated"] is True and session["device"] == "Chrome"


def test_denied_request_issues_nothing(server):
    client, _ = server
    started = client.post("/api/pair/start", json={"name": "x"}).json()
    login(client)
    client.post(f"/api/pair/requests/{started['id']}/deny")
    assert client.get("/api/devices").json() == []
    client.cookies.clear()
    assert client.post("/api/pair/poll", json={"id": started["id"], "secret": started["secret"]}).json() == {"status": "denied"}


def test_unclaimed_approval_is_revoked_when_it_expires(server, monkeypatch):
    client, _ = server
    started = client.post("/api/pair/start", json={"name": "x"}).json()
    login(client)
    client.post(f"/api/pair/requests/{started['id']}/approve")
    assert len(client.get("/api/devices").json()) == 1
    later = time.time() + pairing_module.TTL_SECONDS + 1
    monkeypatch.setattr(pairing_module.time, "time", lambda: later)
    assert client.get("/api/pair/requests").json() == []
    assert client.get("/api/devices").json() == []


def test_pending_requests_are_capped(server):
    client, _ = server
    for _ in range(pairing_module.MAX_PENDING):
        assert client.post("/api/pair/start", json={"name": "x"}).status_code == 200
    assert client.post("/api/pair/start", json={"name": "x"}).status_code == 429


def test_device_token_is_limited_to_extension_endpoints(server):
    client, tmp_path = server
    token = pair(client)
    allowed = client.get("/api/providers", headers=bearer(token))
    assert allowed.status_code == 200
    for method, path in [("GET", "/api/jobs"), ("GET", "/api/devices"), ("GET", "/api/pair/requests"),
                         ("GET", "/api/model-setup"), ("POST", "/api/providers")]:
        assert client.request(method, path, headers=bearer(token)).status_code == 403, path
    # Page images of batch jobs stay private. A live job is readable only by the device that owns it.
    db = Database(tmp_path / "app.db")
    for job_id, live in (("batch", 0), ("live", 1)):
        db.create_job({"id": job_id, "title": "t", "engine": "koharu", "provider_id": "p", "options": {},
                       "instructions": "", "live": live, "created_at": time.time()}, [])
    assert client.get("/api/jobs/batch/pages/1/output", headers=bearer(token)).status_code == 403
    # A live job with no owning device is not this extension's session.
    assert client.get("/api/jobs/live/pages/1/output", headers=bearer(token)).status_code == 403


def test_live_output_requires_the_owning_paired_device(server):
    client, tmp_path = server
    owner_token = pair(client, "owner")
    other_token = pair(client, "other")
    login(client)
    devices = {device["name"]: device["id"] for device in client.get("/api/devices").json()}
    client.cookies.clear()
    db = Database(tmp_path / "app.db")
    db.create_job({
        "id": "owned", "title": "t", "engine": "koharu", "provider_id": "p", "options": {},
        "instructions": "", "live": 1, "device_id": devices["owner"], "created_at": time.time(),
    }, [{"job_id": "owned", "idx": 1, "source_name": "page", "file": "0001.png", "sha256": "abc"}])
    db.set_page_output("owned", 1, "0001.png")
    output = tmp_path / "jobs" / "owned" / "output"
    output.mkdir(parents=True)
    (output / "0001.png").write_bytes(b"translated-png")

    owned = client.get("/api/jobs/owned/pages/1/output", headers=bearer(owner_token))
    assert owned.status_code == 200 and owned.content == b"translated-png"
    assert client.get("/api/jobs/owned/pages/1/output", headers=bearer(other_token)).status_code == 403
    assert client.get("/api/jobs/owned/pages/1/source", headers=bearer(owner_token)).status_code == 403
    assert client.get("/api/live/owned", headers=bearer(other_token)).status_code == 403
    assert client.post("/api/live/owned/discard", json={"pages": [1]}, headers=bearer(other_token)).status_code == 403
    assert client.get("/api/live/owned", headers=bearer(owner_token)).status_code == 200

    login(client)
    web = client.get("/api/jobs/owned/pages/1/output")
    assert web.status_code == 200 and web.content == b"translated-png"


def test_revoking_or_unpairing_disconnects_the_device(server):
    client, _ = server
    revoked, unpaired = pair(client, "revoked"), pair(client, "unpaired")
    login(client)
    device = next(d for d in client.get("/api/devices").json() if d["name"] == "revoked")
    client.delete(f"/api/devices/{device['id']}")
    client.cookies.clear()
    assert client.get("/api/providers", headers=bearer(revoked)).status_code == 401
    assert client.post("/api/pair/unpair", headers=bearer(unpaired)).status_code == 200
    assert client.get("/api/providers", headers=bearer(unpaired)).status_code == 401
    login(client)
    assert client.get("/api/devices").json() == []


def test_server_without_password_needs_no_pairing(tmp_path):
    settings = replace(load_settings(), data_dir=tmp_path, host="127.0.0.1", password=None)
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/pair/start", json={"name": "x"}).status_code == 400
        assert client.get("/api/session").json()["authenticated"] is True
