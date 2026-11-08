"""Recommendations use real option resolution, cache readiness, and host support."""

from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from app import model_setup as models
from app.db import Database
from app.engines import build_engines
from app.main import create_app
from app.secretbox import SecretBox
from app.settings import load_settings


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    settings = replace(
        load_settings(), data_dir=tmp_path / "data", password="",
        koharu_bin=tmp_path / "koharu", mt_dir=tmp_path / "mt",
        mt_python=tmp_path / "mt-python", hf_token_file=tmp_path / "no-token",
    )
    settings.koharu_bin.write_text("isolated engine installation fixture")
    setup = models.ModelSetup(
        Database(settings.db_path), SecretBox(settings.data_dir / "secret.key"),
        settings, build_engines(settings),
    )
    setup.save({"cache_dir": str(tmp_path / "cache")})
    return setup, settings


def set_host(monkeypatch, os_name, machine):
    monkeypatch.setattr(models.sys, "platform", os_name)
    monkeypatch.setattr(models.platform, "machine", lambda: machine)
    monkeypatch.setattr(models.platform, "system", lambda: {"darwin": "Darwin", "linux": "Linux"}.get(os_name, os_name))


def cache_artifacts(setup, artifacts):
    for artifact in artifacts:
        path = models.artifact_path(artifact, setup.cache_dir(), setup.settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"isolated model cache fixture")


@pytest.mark.parametrize(
    ("os_name", "machine", "accelerator", "supported"),
    [
        ("darwin", "arm64", "metal", {"koharu"}),
        ("linux", "x86_64", "cuda", {"koharu", "mangatranslator"}),
    ],
)
def test_api_recommendations_follow_server_setup_path_and_real_resolver(
    preparation, monkeypatch, os_name, machine, accelerator, supported,
):
    setup, settings = preparation
    set_host(monkeypatch, os_name, machine)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/model-setup")
    assert response.status_code == 200
    payload = response.json()
    assert payload["platform"]["accelerator"] == accelerator
    assert set(payload["platform"]["supported_engines"]) == supported
    for recommendation in payload["recommendations"]:
        assert recommendation["available"] and not recommendation["reason"]
        assert not recommendation["prepared"]
        engine = setup.engines[recommendation["engine"]]
        options = engine.resolve_options(recommendation["options"])
        profile = models.PROFILE_BY_ID[recommendation["profile_id"]]
        with pytest.raises(RuntimeError):
            setup.require_prepared_for_new_job(options)
        cache_artifacts(setup, profile.files)
        setup.require_prepared_for_new_job(options)
    refreshed = setup.public()
    assert all(item["prepared"] for item in refreshed["recommendations"])
    if os_name == "darwin":
        supplemental = [item for item in payload["profiles"] if item["category"] == "supplemental"]
        assert supplemental and all(not item["available"] and item["reason"] for item in supplemental)


def test_recommended_restoration_requires_complete_extra_bundle_and_shared_ocr(preparation, monkeypatch):
    setup, _settings = preparation
    set_host(monkeypatch, "linux", "x86_64")
    base = models.PROFILE_BY_ID["koharu-default"]
    restoration = models.PROFILE_BY_ID["koharu-inpainting-flux2-klein"]
    extra = [artifact for artifact in restoration.files if artifact not in base.files]
    cache_artifacts(setup, (*base.files, *extra[:-1]))
    recommendations = {item["id"]: item for item in setup.public()["recommendations"]}
    assert recommendations["koharu-standard"]["prepared"]
    assert not recommendations["koharu-restoration"]["prepared"]
    assert all(item["available"] for item in recommendations.values())
    options = setup.engines["koharu"].resolve_options(recommendations["koharu-restoration"]["options"])
    with pytest.raises(RuntimeError):
        setup.require_prepared_for_new_job(options)

    missing = models.artifact_path(extra[-1], setup.cache_dir(), setup.settings)
    cache_artifacts(setup, [extra[-1]])
    # A completion marker must agree with the file size, not just exist.
    marker = missing.with_name(missing.name + ".ct-complete")
    marker.write_text('{"size": 999}')
    assert not next(item for item in setup.public()["recommendations"] if item["id"] == "koharu-restoration")["prepared"]
    marker.unlink()
    assert all(item["prepared"] for item in setup.public()["recommendations"])
    setup.require_prepared_for_new_job(options)

    models.artifact_path(models.PADDLE[-1], setup.cache_dir(), setup.settings).unlink()
    assert not any(item["prepared"] for item in setup.public()["recommendations"])
    with pytest.raises(RuntimeError):
        setup.require_prepared_for_new_job(options)


@pytest.mark.parametrize("absence", ["not-installed", "not-registered"])
def test_ready_models_do_not_make_missing_engine_available(preparation, monkeypatch, absence):
    setup, settings = preparation
    set_host(monkeypatch, "linux", "x86_64")
    cache_artifacts(setup, models.PROFILE_BY_ID["koharu-inpainting-flux2-klein"].files)
    assert all(item["available"] and item["prepared"] for item in setup.public()["recommendations"])
    if absence == "not-installed":
        settings.koharu_bin.unlink()
    else:
        del setup.engines["koharu"]
    payload = setup.public()
    assert all(item["prepared"] and not item["available"] and item["reason"] for item in payload["recommendations"])
    profiles = [item for item in payload["profiles"] if item["engine"] == "koharu"]
    assert all(not item["engine_installed"] and not item["available"] for item in profiles)


@pytest.mark.parametrize(
    ("os_name", "machine"),
    [("darwin", "x86_64"), ("win32", "ARM64"), ("freebsd14", "x86_64")],
    ids=["intel-or-rosetta-mac", "windows", "other-os"],
)
def test_unsupported_server_never_offers_runnable_recommendation(preparation, monkeypatch, os_name, machine):
    setup, _settings = preparation
    set_host(monkeypatch, os_name, machine)
    cache_artifacts(setup, models.PROFILE_BY_ID["koharu-inpainting-flux2-klein"].files)
    payload = setup.public()
    assert payload["platform"]["supported_engines"] == []
    assert payload["platform"]["accelerator"] == "none"
    assert all(item["prepared"] and not item["available"] and item["reason"] for item in payload["recommendations"])
    assert {item["id"] for item in payload["profiles"]} == set(models.PROFILE_BY_ID)
    assert all(not item["available"] and item["reason"] for item in payload["profiles"])
    assert {item["category"] for item in payload["profiles"]} == {"base", "ocr", "inpainting", "supplemental"}
