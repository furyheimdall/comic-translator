"""FastAPI application: web UI, REST API, and the internal LLM proxy."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import oauth
from .db import Database
from .engines import build_engines
from .jobs import IngestError, JobManager, collect_images
from .model_setup import ModelSetup
from .prompting import build_instructions
from .providers import (
    KINDS, PRESETS, REASONING_LABELS, ChatRequest, ProviderConfig, ProviderError, chat, completion_envelope,
    list_models, reasoning_levels, with_overrides,
)
from .secretbox import SecretBox, read_or_create_token
from .settings import ROOT, Settings, load_settings

SESSION_COOKIE = "ct_session"
SESSION_TTL = 30 * 24 * 3600


class NoCacheStaticFiles(StaticFiles):
    """UI assets change with deploys; force revalidation (ETag → 304) instead of heuristic caching."""

    def file_response(self, *args: Any, **kwargs: Any):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    if not settings.is_loopback_only() and not settings.password:
        raise SystemExit(f"CT_HOST={settings.host} 로 외부에 공개하려면 CT_PASSWORD 를 설정해야 합니다.")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    box = SecretBox(settings.data_dir / "secret.key")
    proxy_token = read_or_create_token(settings.data_dir / "proxy.token")
    engines = build_engines(settings)
    model_setup = ModelSetup(db, box, settings, engines)
    for instance in engines.values():
        instance.model_environment = model_setup.environment
    jobs = JobManager(
        db, settings.jobs_dir, engines, lambda: (settings.llm_base_url, proxy_token),
        live_idle_seconds=settings.live_idle_seconds,
    )
    sessions: dict[str, float] = {}

    async def save_oauth(provider_id: str, tokens: dict[str, Any], account: str | None) -> None:
        db.set_provider_secret(provider_id, box.seal({"oauth": tokens}), account)

    logins = oauth.LoginManager(save_oauth)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        jobs.start()
        yield
        jobs.stop()

    app = FastAPI(title="comic-translator", lifespan=lifespan, docs_url=None, redoc_url=None)

    # ------------------------------------------------------------------ auth
    def authenticated(request: Request) -> bool:
        if not settings.password:
            return True
        header = request.headers.get("authorization", "")
        if header.startswith("Bearer "):
            # Extension API token; only its hash is stored.
            return db.use_api_token(hashlib.sha256(header[7:].strip().encode("utf-8")).hexdigest())
        token = request.cookies.get(SESSION_COOKIE)
        expires = sessions.get(token or "")
        return bool(expires and expires > time.time())

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith("/llm/"):
            header = request.headers.get("authorization", "")
            if not hmac.compare_digest(header, f"Bearer {proxy_token}"):
                return JSONResponse({"error": {"message": "invalid proxy token"}}, status_code=401)
        elif path.startswith("/api/") and path not in ("/api/login", "/api/session") and not authenticated(request):
            return JSONResponse({"detail": "로그인이 필요합니다."}, status_code=401)
        return await call_next(request)

    @app.get("/api/session")
    async def session(request: Request) -> dict[str, Any]:
        return {"authenticated": authenticated(request), "password_required": bool(settings.password)}

    @app.post("/api/login")
    async def login(payload: dict[str, str]) -> JSONResponse:
        if not settings.password or not hmac.compare_digest(
            payload.get("password", "").encode("utf-8"),
            settings.password.encode("utf-8"),
        ):
            await asyncio.sleep(1)
            raise HTTPException(401, "비밀번호가 올바르지 않습니다.")
        token = secrets.token_urlsafe(32)
        sessions[token] = time.time() + SESSION_TTL
        response = JSONResponse({"ok": True})
        response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL, httponly=True, samesite="strict")
        return response

    # ------------------------------------------------------------------ extension API tokens
    def require_web_session(request: Request) -> None:
        # An extension token must not be able to mint or revoke tokens.
        if request.headers.get("authorization"):
            raise HTTPException(403, "토큰 관리는 웹 화면에서만 할 수 있습니다.")

    @app.get("/api/tokens")
    async def list_tokens(request: Request) -> list[dict[str, Any]]:
        require_web_session(request)
        return db.list_api_tokens()

    @app.post("/api/tokens")
    async def create_token(request: Request, payload: dict[str, str]) -> dict[str, Any]:
        require_web_session(request)
        name = str(payload.get("name", "")).strip()[:100] or "브라우저 확장"
        token = "ct_" + secrets.token_urlsafe(32)
        token_id = uuid.uuid4().hex[:12]
        db.create_api_token(token_id, name, hashlib.sha256(token.encode("utf-8")).hexdigest())
        # The plain token is returned once and never stored.
        return {"id": token_id, "name": name, "token": token}

    @app.delete("/api/tokens/{token_id}")
    async def delete_token(request: Request, token_id: str) -> dict[str, bool]:
        require_web_session(request)
        db.delete_api_token(token_id)
        return {"ok": True}

    # ------------------------------------------------------------------ meta
    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        return {
            "engines": [engine.public() for engine in engines.values()],
            "provider_kinds": [{"kind": kind, **PRESETS[kind]} for kind in KINDS],
            "default_instructions": build_instructions(""),
        }
    # -------------------------------------------------------------- model setup
    @app.get("/api/model-setup")
    async def get_model_setup() -> dict[str, Any]:
        return await asyncio.to_thread(model_setup.public)

    @app.put("/api/model-setup")
    async def save_model_setup(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(model_setup.save, payload)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/model-setup/prepare", status_code=202)
    async def prepare_models(payload: dict[str, str]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(model_setup.prepare, payload.get("profile_id", ""))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


    # ------------------------------------------------------------------ providers
    def provider_config(row: dict[str, Any]) -> ProviderConfig:
        sealed = box.open(row["secret"]) or {}
        return ProviderConfig(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            auth=row["auth"],
            base_url=row["base_url"],
            model=row["model"],
            extra_body=json.loads(row["extra_body"] or "{}"),
            vision=bool(row["vision"]),
            secret=sealed.get("oauth") if row["auth"] == "oauth" else sealed,
        )

    def provider_public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "auth": row["auth"],
            "base_url": row["base_url"],
            "model": row["model"],
            "extra_body": json.loads(row["extra_body"] or "{}"),
            "vision": bool(row["vision"]),
            "connected": bool(row["secret"]) or row["auth"] == "none",
            "account": row["account"],
        }

    def require_provider(provider_id: str) -> dict[str, Any]:
        row = db.get_provider(provider_id)
        if not row:
            raise HTTPException(404, "제공자를 찾을 수 없습니다.")
        return row

    @app.get("/api/providers")
    async def list_providers() -> list[dict[str, Any]]:
        return [provider_public(row) for row in db.list_providers()]

    model_lists: dict[str, tuple[float, list[str], str | None]] = {}

    async def provider_models(row: dict[str, Any]) -> tuple[list[str], str | None]:
        """Models the provider reports (cached 10 min); falls back to the configured model."""
        key = f"{row['id']}:{row['model']}:{row['base_url']}"
        cached = model_lists.get(key)
        if cached and cached[0] > time.time():
            return cached[1], cached[2]
        try:
            models, error = await list_models(provider_config(row)), None
        except (ProviderError, httpx.HTTPError, ValueError) as exc:
            models, error = [row["model"]], f"모델 목록을 가져오지 못해 설정된 모델만 표시합니다: {exc}"
        model_lists[key] = (time.time() + 600, models, error)
        return models, error

    @app.get("/api/providers/{provider_id}/options")
    async def provider_options(provider_id: str) -> dict[str, Any]:
        row = require_provider(provider_id)
        models, error = await provider_models(row)
        return {
            "models": models,
            "default_model": row["model"],
            "models_error": error,
            "reasoning_levels": [
                {"value": level, "label": REASONING_LABELS[level]} for level in reasoning_levels(provider_config(row))
            ],
        }

    @app.post("/api/providers")
    async def save_provider(payload: dict[str, Any]) -> dict[str, Any]:
        kind = payload.get("kind")
        if kind not in KINDS:
            raise HTTPException(400, "알 수 없는 제공자 종류입니다.")
        auth = payload.get("auth") or PRESETS[kind]["auth"][0]
        if auth not in PRESETS[kind]["auth"]:
            raise HTTPException(400, f"{PRESETS[kind]['label']}에서 지원하지 않는 인증 방식입니다: {auth}")
        extra_body = payload.get("extra_body") or {}
        if isinstance(extra_body, str):
            try:
                extra_body = json.loads(extra_body or "{}")
            except json.JSONDecodeError as exc:
                raise HTTPException(400, f"추가 요청 JSON 오류: {exc}") from exc
        if not isinstance(extra_body, dict):
            raise HTTPException(400, "추가 요청 값은 JSON 객체여야 합니다.")
        model = (payload.get("model") or "").strip()
        if not model:
            raise HTTPException(400, "모델 이름이 필요합니다.")
        base_url = (payload.get("base_url") or "").strip() or None
        if kind == "openai_compatible" and not base_url:
            raise HTTPException(400, "OpenAI 호환 제공자에는 base URL이 필요합니다.")
        existing = db.get_provider(payload["id"]) if payload.get("id") else None
        provider_id = existing["id"] if existing else uuid.uuid4().hex[:10]
        keep_secret = existing and existing["kind"] == kind and existing["auth"] == auth
        secret = existing["secret"] if keep_secret else None
        account = existing["account"] if keep_secret else None
        if auth == "api_key" and payload.get("api_key"):
            secret, account = box.seal({"api_key": payload["api_key"].strip()}), None
        db.upsert_provider(
            {
                "id": provider_id,
                "name": (payload.get("name") or PRESETS[kind]["label"]).strip(),
                "kind": kind,
                "auth": auth,
                "base_url": base_url,
                "model": model,
                "extra_body": extra_body,
                "vision": 1 if payload.get("vision") else 0,
                "secret": secret,
                "account": account,
                "created_at": existing["created_at"] if existing else time.time(),
            }
        )
        return provider_public(require_provider(provider_id))

    @app.delete("/api/providers/{provider_id}")
    async def delete_provider(provider_id: str) -> dict[str, bool]:
        require_provider(provider_id)
        if any(job["provider_id"] == provider_id and job["status"] in ("queued", "running") for job in db.list_jobs()):
            raise HTTPException(409, "이 제공자를 쓰는 대기/실행 중 작업이 있습니다.")
        db.delete_provider(provider_id)
        return {"ok": True}

    @app.post("/api/providers/{provider_id}/logout")
    async def logout_provider(provider_id: str) -> dict[str, Any]:
        require_provider(provider_id)
        db.set_provider_secret(provider_id, None, None)
        return provider_public(require_provider(provider_id))

    @app.post("/api/providers/{provider_id}/oauth")
    async def start_oauth(provider_id: str) -> dict[str, Any]:
        row = require_provider(provider_id)
        if row["auth"] != "oauth":
            raise HTTPException(400, "OAuth 인증을 쓰는 제공자가 아닙니다.")
        return logins.start(provider_id, row["kind"]).public()

    @app.get("/api/oauth/{login_id}")
    async def oauth_status(login_id: str) -> dict[str, Any]:
        session_ = logins.get(login_id)
        if not session_:
            raise HTTPException(404, "로그인 세션이 없습니다.")
        return session_.public()

    @app.post("/api/oauth/{login_id}/cancel")
    async def oauth_cancel(login_id: str) -> dict[str, bool]:
        logins.cancel(login_id)
        return {"ok": True}

    @app.post("/api/providers/{provider_id}/test")
    async def test_provider(provider_id: str) -> dict[str, Any]:
        config = provider_config(require_provider(provider_id))
        started = time.time()
        try:
            text = await chat(
                config,
                ChatRequest(
                    messages=[
                        {"role": "system", "content": "Translate the Japanese line into natural Korean. Reply with the translation only."},
                        {"role": "user", "content": "先輩、まだ残ってたんですか？"},
                    ],
                    max_tokens=2048,
                ),
                lambda tokens: save_oauth(provider_id, tokens, db.get_provider(provider_id)["account"]),
            )
        except (ProviderError, oauth.OAuthError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "reply": text.strip(), "seconds": round(time.time() - started, 1)}

    # ------------------------------------------------------------------ jobs
    def job_public(job: dict[str, Any], with_pages: bool = False) -> dict[str, Any]:
        pages = db.list_pages(job["id"])
        data = {
            **{key: job[key] for key in ("id", "title", "engine", "provider_id", "status", "progress", "message", "error", "created_at", "started_at", "finished_at", "instructions")},
            "live": bool(job["live"]),
            "llm": json.loads(job["llm"] or "{}"),
            "options": json.loads(job["options"]),
            "page_count": len(pages),
            "translated_count": sum(1 for page in pages if page["output"]),
        }
        if with_pages:
            data["pages"] = [page_public(page) for page in pages]
        return data

    def page_public(page: dict[str, Any]) -> dict[str, Any]:
        return {"idx": page["idx"], "source_name": page["source_name"], "translated": bool(page["output"]), "error": page["error"]}

    def require_job(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "작업을 찾을 수 없습니다.")
        return job

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        return [job_public(job) for job in db.list_jobs()]

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile] = File(...),
        engine: str = Form(...),
        provider_id: str = Form(...),
        title: str = Form(""),
        options: str = Form("{}"),
        instructions: str = Form(""),
    ) -> dict[str, Any]:
        require_provider(provider_id)
        try:
            parsed_options = json.loads(options or "{}")
            if engine == "koharu":
                try:
                    model_setup.require_prepared_for_new_job(engines[engine].resolve_options(parsed_options))
                except RuntimeError as exc:
                    raise HTTPException(409, str(exc)) from exc
            images = await asyncio.to_thread(collect_images, [(f.filename or "upload", f.file) for f in files])
            job_id = await asyncio.to_thread(
                jobs.create,
                title=title.strip() or (files[0].filename or "작업"),
                engine_id=engine,
                provider_id=provider_id,
                options=parsed_options,
                instructions=instructions,
                images=images,
            )
        except (IngestError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return job_public(require_job(job_id), with_pages=True)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return job_public(require_job(job_id), with_pages=True)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, bool]:
        require_job(job_id)
        jobs.cancel(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry")
    async def retry_job(job_id: str, payload: dict[str, str] | None = None) -> dict[str, Any]:
        job = require_job(job_id)
        provider_id = (payload or {}).get("provider_id", job["provider_id"])
        provider = require_provider(provider_id)
        if not provider_public(provider)["connected"]:
            raise HTTPException(400, "선택한 LLM 제공자를 먼저 연결하세요.")
        try:
            jobs.retry(job_id, provider_id=provider_id)
        except IngestError as exc:
            raise HTTPException(409, str(exc)) from exc
        return job_public(require_job(job_id), with_pages=True)

    @app.post("/api/jobs/{job_id}/retry-failed")
    async def retry_failed_pages(job_id: str, payload: dict[str, str] | None = None) -> dict[str, Any]:
        job = require_job(job_id)
        provider_id = (payload or {}).get("provider_id", job["provider_id"])
        provider = require_provider(provider_id)
        if not provider_public(provider)["connected"]:
            raise HTTPException(400, "선택한 LLM 제공자를 먼저 연결하세요.")
        try:
            jobs.retry(job_id, failed_only=True, provider_id=provider_id)
        except IngestError as exc:
            raise HTTPException(409, str(exc)) from exc
        return job_public(require_job(job_id), with_pages=True)

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, bool]:
        require_job(job_id)
        try:
            jobs.delete(job_id)
        except IngestError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True}

    def page_row(job_id: str, idx: int) -> dict[str, Any]:
        for page in db.list_pages(job_id):
            if page["idx"] == idx:
                return page
        raise HTTPException(404, "페이지를 찾을 수 없습니다.")

    @app.get("/api/jobs/{job_id}/pages/{idx}/{kind}")
    async def page_image(job_id: str, idx: int, kind: str) -> FileResponse:
        page = page_row(job_id, idx)
        path = jobs.input_path(job_id, page) if kind == "source" else jobs.output_path(job_id, page)
        if kind not in ("source", "output") or not path or not path.exists():
            raise HTTPException(404, "이미지가 없습니다.")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/download")
    async def download_job(job_id: str) -> FileResponse:
        job = require_job(job_id)
        try:
            path = await asyncio.to_thread(jobs.archive, job_id)
        except IngestError as exc:
            raise HTTPException(409, str(exc)) from exc
        safe = "".join(ch for ch in job["title"] if ch not in '\\/:*?"<>|').strip() or job_id
        return FileResponse(path, media_type="application/vnd.comicbook+zip", filename=f"{safe} (한국어).cbz")

    @app.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str) -> PlainTextResponse:
        require_job(job_id)
        path = jobs.log_path(job_id)
        if not path.exists():
            return PlainTextResponse("")
        data = path.read_bytes()[-200_000:]
        return PlainTextResponse(data.decode("utf-8", errors="replace"))

    # ------------------------------------------------------------------ live sessions (browser extension)
    def require_live(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if not job or not job["live"]:
            raise HTTPException(404, "실시간 번역 세션을 찾을 수 없습니다.")
        return job

    @app.post("/api/live")
    async def open_live(payload: dict[str, Any]) -> dict[str, Any]:
        provider_id = str(payload.get("provider_id", ""))
        engine = str(payload.get("engine", "koharu"))
        row = require_provider(provider_id)
        if not provider_public(row)["connected"]:
            raise HTTPException(400, "선택한 LLM 제공자를 먼저 연결하세요.")
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise HTTPException(400, "options 는 객체여야 합니다.")
        # Model and reasoning level may only be chosen from what the server offers for this provider.
        llm: dict[str, str] = {}
        model = str(payload.get("model") or row["model"])
        if model != row["model"]:
            models, _error = await provider_models(row)
            if model not in models:
                raise HTTPException(400, f"{row['name']}에서 제공하지 않는 모델입니다: {model}")
            llm["model"] = model
        reasoning = str(payload.get("reasoning") or "default")
        if reasoning not in reasoning_levels(provider_config(row)):
            raise HTTPException(400, f"{row['name']}에서 지원하지 않는 추론 수준입니다: {reasoning}")
        if reasoning != "default":
            llm["reasoning"] = reasoning
        try:
            if engine == "koharu":
                try:
                    model_setup.require_prepared_for_new_job(engines[engine].resolve_options(options))
                except RuntimeError as exc:
                    raise HTTPException(409, str(exc)) from exc
            job_id = await asyncio.to_thread(
                jobs.open_live,
                engine_id=engine,
                provider_id=provider_id,
                options=options,
                instructions=str(payload.get("instructions", "")),
                title=str(payload.get("title", "")).strip()[:200] or time.strftime("웹 실시간 번역 %Y-%m-%d"),
                llm=llm,
            )
        except (IngestError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**job_public(require_live(job_id)), "pages": []}

    @app.get("/api/live/{job_id}")
    async def get_live(job_id: str, pages: str = "") -> dict[str, Any]:
        job = require_live(job_id)
        try:
            wanted = {int(part) for part in pages.split(",") if part.strip()}
        except ValueError as exc:
            raise HTTPException(400, "pages 는 쉼표로 구분한 번호여야 합니다.") from exc
        rows = [page_public(page) for page in db.list_pages(job_id) if page["idx"] in wanted]
        return {**job_public(job), "pages": rows}

    @app.post("/api/live/{job_id}/pages")
    async def add_live_page(job_id: str, file: UploadFile = File(...), source_name: str = Form("")) -> dict[str, Any]:
        require_live(job_id)
        data = await file.read()
        try:
            return await asyncio.to_thread(jobs.add_live_page, job_id, source_name or file.filename or "image", data)
        except KeyError as exc:
            raise HTTPException(404, "실시간 번역 세션을 찾을 수 없습니다.") from exc
        except IngestError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/live/{job_id}/discard")
    async def discard_live_pages(job_id: str, payload: dict[str, Any]) -> dict[str, bool]:
        require_live(job_id)
        try:
            indices = {int(idx) for idx in payload.get("pages") or []}
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "pages 는 번호 목록이어야 합니다.") from exc
        await asyncio.to_thread(jobs.discard_live_pages, job_id, indices)
        return {"ok": True}

    # ------------------------------------------------------------------ LLM proxy (engines only)
    @app.get("/llm/v1/models")
    async def proxy_models() -> dict[str, Any]:
        data = [{"id": f"job:{job['id']}", "object": "model"} for job in db.list_jobs() if job["status"] == "running"]
        return {"object": "list", "data": data}

    @app.post("/llm/v1/chat/completions")
    async def proxy_chat(payload: dict[str, Any]) -> JSONResponse:
        model = str(payload.get("model", ""))
        if not model.startswith("job:"):
            return _proxy_error("model must be job:<id>", 400)
        job = db.get_job(model[4:])
        if not job:
            return _proxy_error("unknown job", 404)
        row = db.get_provider(job["provider_id"])
        if not row:
            return _proxy_error("job provider was deleted", 409)
        if payload.get("stream"):
            return _proxy_error("streaming is not supported", 400)
        messages = list(payload.get("messages") or [])
        if job["engine"] == "mangatranslator":
            messages = _inject_system(messages, build_instructions(job["instructions"]))
        request = ChatRequest(
            messages=messages,
            max_tokens=payload.get("max_tokens") or payload.get("max_completion_tokens"),
            temperature=payload.get("temperature"),
            response_format=payload.get("response_format"),
        )
        # The MangaTranslator proxy caller waits 3780s. Leave the provider's
        # 600s read timeout intact while bounding the entire proxy request.
        deadline = time.monotonic() + 3735
        backoff = (5, 10, 20, 40, 60)
        retries = 0
        while True:
            current = db.get_job(job["id"])
            if not current or current["status"] != "running" or current["started_at"] != job["started_at"]:
                return _proxy_error("job is no longer running", 409)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _proxy_error("upstream request timed out", 504)
            try:
                async with asyncio.timeout(remaining):
                    config = provider_config(row)
                    llm_choice = json.loads(job["llm"] or "{}")
                    if llm_choice:
                        config = with_overrides(config, llm_choice.get("model"), llm_choice.get("reasoning"))
                    text = await chat(
                        config,
                        request,
                        lambda tokens: save_oauth(row["id"], tokens, row["account"]),
                    )
            except (ProviderError, oauth.OAuthError, httpx.RequestError, TimeoutError) as exc:
                if isinstance(exc, ProviderError):
                    status = exc.status
                    # ProviderError defaults to 502 even for malformed/empty
                    # replies; retry that status only for upstream HTTP 502.
                    transient = status == 429 or (
                        500 <= status < 600 and (status != 502 or "HTTP 502" in str(exc))
                    )
                elif isinstance(exc, oauth.OAuthError):
                    status, transient = 401, False
                else:
                    status, transient = (504 if isinstance(exc, TimeoutError) else 502), True
                error_message = str(exc) or "upstream request timed out"
                delay = backoff[retries] if retries < len(backoff) else None
                if transient and delay is not None and deadline - time.monotonic() > delay:
                    retries += 1
                    db.update_job(job["id"], message=f"LLM {row['name']} 재시도 {retries}/{len(backoff)} ({delay}초 후)")
                    with open(jobs.log_path(job["id"]), "a", encoding="utf-8") as log:
                        log.write(f"[proxy] {row['name']} 재시도 {retries}/{len(backoff)} ({delay}초 후): {error_message}\n")
                    wake_at = time.monotonic() + delay
                    while time.monotonic() < wake_at:
                        current = db.get_job(job["id"])
                        if not current or current["status"] != "running" or current["started_at"] != job["started_at"]:
                            return _proxy_error("job is no longer running", 409)
                        await asyncio.sleep(min(1, wake_at - time.monotonic()))
                    continue
                with open(jobs.log_path(job["id"]), "a", encoding="utf-8") as log:
                    log.write(f"[proxy] {row['name']} 오류 (재시도 {retries}회): {error_message}\n")
                return _proxy_error(error_message, status if 400 <= status < 600 else 502)
            current = db.get_job(job["id"])
            if not current or current["status"] != "running" or current["started_at"] != job["started_at"]:
                return _proxy_error("job is no longer running", 409)
            if retries:
                db.update_job(job["id"], message=f"LLM {row['name']} 성공 (재시도 {retries}회)")
                with open(jobs.log_path(job["id"]), "a", encoding="utf-8") as log:
                    log.write(f"[proxy] {row['name']} 성공 (재시도 {retries}회)\n")
            return JSONResponse(completion_envelope(model, text))

    # ------------------------------------------------------------------ static web UI
    web_dir = ROOT / "web"
    app.mount("/", NoCacheStaticFiles(directory=web_dir, html=True), name="web")
    return app


def _inject_system(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, list):
                content = [*content, {"type": "text", "text": text}]
            else:
                content = f"{content or ''}\n\n{text}"
            return [*messages[:index], {**message, "content": content}, *messages[index + 1 :]]
    return [{"role": "system", "content": text}, *messages]


def _proxy_error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "proxy_error"}}, status_code=status)


def run() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
