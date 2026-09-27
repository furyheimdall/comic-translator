"""Observable setup boundaries: no secret disclosure and no partial model readiness."""

import io
from dataclasses import replace
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.engines import build_engines
from app.db import Database
from app.main import create_app
from app import model_setup as models
from app.secretbox import SecretBox
from app.settings import load_settings


class DownloadResponse(io.BytesIO):
    def __init__(self, payload: bytes, expected_length: int):
        super().__init__(payload)
        self.headers = {"Content-Length": str(expected_length), "Content-Type": "application/octet-stream"}


def test_saved_token_is_private_and_clear_disables_inherited_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "legacy-host-token")
    legacy_file = tmp_path / "legacy-token"
    legacy_file.write_text("legacy-file-token")
    settings = replace(load_settings(), data_dir=tmp_path / "private-data", hf_token_file=legacy_file, password="test")
    cache = tmp_path / "model cache"
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/model-setup").status_code == 401
        assert client.post("/api/login", json={"password": "test"}).status_code == 200
        inherited = client.get("/api/model-setup").json()
        assert inherited["token_source"] == "env" and inherited["token_configured"]
        credential = "hf_" + "a" * 38
        response = client.put("/api/model-setup", json={"cache_dir": str(cache), "token": credential})
        assert response.status_code == 200
        payload = response.json()
        assert payload["cache_dir"] == str(cache) and payload["token_source"] == "saved"
        assert credential not in response.text
        row = Database(settings.db_path).get_model_setup()
        assert row is not None and credential.encode() not in row["token"]
        assert SecretBox(settings.data_dir / "secret.key").open(row["token"]) == {"token": credential}
        cleared = client.put("/api/model-setup", json={"clear_token": True}).json()
        assert not cleared["token_configured"] and cleared["token_source"] is None
        refreshed = client.get("/api/model-setup").json()
        assert not refreshed["token_configured"]
        resolved = models.ModelSetup(Database(settings.db_path), SecretBox(settings.data_dir / "secret.key"), settings, build_engines(settings))
        env = resolved.environment("mangatranslator")
        assert env["HF_TOKEN"] == "" and env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert env["HF_TOKEN_PATH"] == "/dev/null" and env["HUGGING_FACE_HUB_TOKEN"] == ""
        assert env["HF_HOME"] == str(cache / "huggingface") and env["XDG_CACHE_HOME"] == str(cache)


def test_incomplete_download_fails_then_resume_prepares_engine_consumed_file(tmp_path, monkeypatch):
    settings = replace(load_settings(), data_dir=tmp_path / "data", mt_dir=tmp_path / "mt")
    db = Database(settings.db_path)
    setup = models.ModelSetup(db, SecretBox(settings.data_dir / "secret.key"), settings, build_engines(settings))
    cache = tmp_path / "cache"
    setup.save({"cache_dir": str(cache)})
    artifact = models.Artifact("example/model", "a" * 40, "subdir/model.safetensors")
    profile = models.Profile("fixture-preparation", "koharu", "합성 모델", "격리 테스트", (artifact,), True)
    monkeypatch.setitem(models.PROFILE_BY_ID, profile.id, profile)
    target = models.artifact_path(artifact, cache, settings)
    monkeypatch.setattr(models._OPENER, "open", lambda *_args, **_kw: DownloadResponse(b"short", 20))
    assert setup.prepare(profile.id)["state"] == "running"
    setup._thread.join(timeout=5)
    assert setup._status["state"] == "failed" and not target.exists()
    assert not models._ready(target)

    payload = b"actual-isolated-model-contents"
    monkeypatch.setattr(models._OPENER, "open", lambda *_args, **_kw: DownloadResponse(payload, len(payload)))
    assert setup.prepare(profile.id)["state"] == "running"
    setup._thread.join(timeout=5)
    assert setup._status["state"] == "done"
    assert target.read_bytes() == payload and models._ready(target)
    assert target == cache / "koharu/packages/hugging-face/models/example--model/snapshots" / ("a" * 40) / "subdir/model.safetensors"
    assert target.with_name(target.name + ".ct-complete").is_file()


def test_new_koharu_job_requires_prepared_models_before_saving_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    settings = replace(load_settings(), data_dir=tmp_path / "data", koharu_bin=tmp_path / "not-installed")
    db = Database(settings.db_path)
    db.upsert_provider({"id": "provider", "name": "test", "kind": "openai_compatible", "auth": "none",
                        "base_url": "http://127.0.0.1:9999/v1", "model": "test-model", "extra_body": {},
                        "vision": 0, "secret": None, "account": None, "created_at": time.time()})
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image, "PNG")
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/jobs", data={"engine": "koharu", "provider_id": "provider"},
                               files={"files": ("test.png", image.getvalue(), "image/png")})
        assert response.status_code == 409 and "모델 준비" in response.json()["detail"]
        assert db.list_jobs() == []
        assert not settings.jobs_dir.exists()


def test_default_profile_recognizes_pinned_quantized_ocr_cache(tmp_path):
    settings = replace(load_settings(), data_dir=tmp_path / "data", mt_dir=tmp_path / "mt")
    setup = models.ModelSetup(Database(settings.db_path), SecretBox(settings.data_dir / "secret.key"), settings, build_engines(settings))
    cache = tmp_path / "cache"
    setup.save({"cache_dir": str(cache)})
    for artifact in (*models.DETECTION, *models.LAMA):
        file = models.artifact_path(artifact, cache, settings)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"cached-model")

    base = cache / "koharu/packages/hugging-face/models/PaddlePaddle--PaddleOCR-VL-1.6-GGUF/snapshots/511b09642bb324401f15f97cc23bc67e8f0a291d"
    for filename in ("PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"):
        base.mkdir(parents=True, exist_ok=True)
        (base / filename).write_bytes(b"cached-model")

    status = next(profile for profile in setup.public()["profiles"] if profile["id"] == "koharu-default")
    assert status["prepared"] and not status["missing"]
    setup.require_prepared_for_new_job({"ocr": "paddleocr-vl-1.6", "inpainting": "lama"})
    (base / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf").unlink()
    assert not next(profile for profile in setup.public()["profiles"] if profile["id"] == "koharu-default")["prepared"]
