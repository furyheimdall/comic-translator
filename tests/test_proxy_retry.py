"""Observable retry and cancellation boundaries of the internal LLM proxy."""

import time
from dataclasses import replace

import httpx
import pytest

import app.main as main
from app.db import Database
from app.providers import ProviderError
from app.settings import load_settings


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    settings = replace(load_settings(), data_dir=tmp_path)
    app = main.create_app(settings)
    db = Database(settings.db_path)
    db.upsert_provider({
        "id": "provider", "name": "test", "kind": "openai_compatible", "auth": "none",
        "base_url": "http://upstream.test/v1", "model": "model", "extra_body": {},
        "vision": 0, "secret": None, "account": None, "created_at": time.time(),
    })
    db.create_job({
        "id": "work", "title": "book", "engine": "koharu", "provider_id": "provider",
        "options": {}, "instructions": "", "created_at": time.time(),
    }, [])
    db.set_job_status("work", "running", progress=0.4)
    (settings.jobs_dir / "work").mkdir(parents=True)

    class Clock:
        now = 0.0
        sleeps = []

        def monotonic(self):
            return self.now

        async def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(main.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main.asyncio, "sleep", clock.sleep)
    headers = {"Authorization": f"Bearer {(tmp_path / 'proxy.token').read_text()}"}
    return app, db, clock, headers, settings.jobs_dir / "work" / "job.log"


async def request_proxy(app, headers):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        return await client.post("/llm/v1/chat/completions", headers=headers, json={
            "model": "job:work", "messages": [{"role": "user", "content": "translate"}],
        })


@pytest.mark.parametrize("error", [
    ProviderError("HTTP 429 rate limited", 429),
    ProviderError("upstream HTTP 503 unavailable", 503),
    ProviderError("upstream HTTP 502 gateway", 502),
    httpx.ConnectError("network unreachable"),
])
async def test_transient_exhausts_exactly_five_retries_and_preserves_last_error(proxy, monkeypatch, error):
    app, db, clock, headers, log_path = proxy
    calls = 0

    async def fails(*_args):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(main, "chat", fails)
    response = await request_proxy(app, headers)
    assert response.status_code == (error.status if isinstance(error, ProviderError) else 502)
    assert response.json()["error"]["message"] == str(error)
    assert calls == 6  # original + five retries, never a seventh
    assert sum(clock.sleeps) == 135
    assert "재시도 5/5 (60초 후)" in log_path.read_text()
    assert "재시도 5회" in log_path.read_text()
    assert "5/5" in db.get_job("work")["message"]
    assert db.get_job("work")["progress"] == 0.4


@pytest.mark.parametrize("error,status", [
    (ProviderError("bad request", 400), 400),
    (ProviderError("invalid key", 401), 401),
    (ProviderError("empty model response"), 502),
])
async def test_nonretryable_provider_failure_returns_immediately(proxy, monkeypatch, error, status):
    app, _db, clock, headers, log_path = proxy
    calls = 0

    async def fails(*_args):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(main, "chat", fails)
    response = await request_proxy(app, headers)
    assert response.status_code == status
    assert response.json()["error"]["message"] == str(error)
    assert calls == 1
    assert clock.sleeps == []
    assert "재시도 0회" in log_path.read_text()


async def test_success_after_transient_failure_returns_original_completion(proxy, monkeypatch):
    app, db, clock, headers, log_path = proxy
    calls = 0

    async def recovers(*_args):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError("HTTP 500 temporarily unavailable", 500)
        return "한국어"

    monkeypatch.setattr(main, "chat", recovers)
    response = await request_proxy(app, headers)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "한국어"
    assert calls == 3
    assert sum(clock.sleeps) == 15
    assert "성공 (재시도 2회)" in log_path.read_text()
    assert db.get_job("work")["progress"] == 0.4


@pytest.mark.parametrize("restart", [False, True])
async def test_cancelled_job_does_not_make_next_upstream_attempt(proxy, monkeypatch, restart):
    app, db, clock, headers, _log_path = proxy
    calls = 0

    async def fails(*_args):
        nonlocal calls
        calls += 1
        raise ProviderError("HTTP 429", 429)

    original_sleep = clock.sleep

    async def cancel_during_backoff(seconds):
        await original_sleep(seconds)
        db.set_job_status("work", "cancelled")
        if restart:
            db.update_job("work", status="running", started_at=db.get_job("work")["started_at"] + 1)

    monkeypatch.setattr(main, "chat", fails)
    monkeypatch.setattr(main.asyncio, "sleep", cancel_during_backoff)
    response = await request_proxy(app, headers)
    assert response.status_code == 409
    assert calls == 1
    assert sum(clock.sleeps) == 1
    assert db.get_job("work")["status"] == ("running" if restart else "cancelled")
