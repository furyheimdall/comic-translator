"""Device-code OAuth logins for ChatGPT (Codex) and xAI (SuperGrok).

Both flows avoid a localhost callback so they work when the browser is on a
different machine than the server.

- ChatGPT/Codex: unofficial — mirrors the Codex CLI device flow; OpenAI can
  change or block it at any time.
- xAI: official OAuth 2.0 device authorization (RFC 8628) against auth.x.ai.

Anthropic is intentionally absent: its terms forbid offering Claude.ai login in
third-party applications (API keys only).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

SecretSaver = Callable[[dict[str, Any]], Awaitable[None]]

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
CODEX_DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
CODEX_DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
CODEX_DEVICE_PAGE = "https://auth.openai.com/codex/device"
CODEX_JWT_AUTH_CLAIM = "https://api.openai.com/auth"

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_ISSUER = "https://auth.x.ai"
XAI_DEVICE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_SCOPES = "openid profile email offline_access grok-cli:access api:access"

REFRESH_SKEW_SECONDS = 120
LOGIN_TIMEOUT_SECONDS = 15 * 60


class OAuthError(RuntimeError):
    pass


def decode_jwt(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError):
        return {}


# --------------------------------------------------------------------------- login sessions


@dataclass
class LoginSession:
    id: str
    provider_id: str
    kind: str
    status: str = "starting"  # starting | pending | done | error | cancelled
    user_code: str | None = None
    verification_url: str | None = None
    message: str = ""
    account: str | None = None
    created_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "status": self.status,
            "user_code": self.user_code,
            "verification_url": self.verification_url,
            "message": self.message,
            "account": self.account,
        }


class LoginManager:
    def __init__(self, on_success: Callable[[str, dict[str, Any], str | None], Awaitable[None]]) -> None:
        self._sessions: dict[str, LoginSession] = {}
        self._on_success = on_success

    def start(self, provider_id: str, kind: str) -> LoginSession:
        if kind not in ("openai", "xai"):
            raise OAuthError("이 제공자는 OAuth 로그인을 지원하지 않습니다.")
        session = LoginSession(id=uuid.uuid4().hex, provider_id=provider_id, kind=kind)
        self._sessions[session.id] = session
        session.task = asyncio.create_task(self._run(session))
        return session

    def get(self, login_id: str) -> LoginSession | None:
        return self._sessions.get(login_id)

    def cancel(self, login_id: str) -> None:
        session = self._sessions.get(login_id)
        if session and session.task and not session.task.done():
            session.task.cancel()
            session.status = "cancelled"

    async def _run(self, session: LoginSession) -> None:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
                flow = _codex_device_login if session.kind == "openai" else _xai_device_login
                tokens = await asyncio.wait_for(flow(client, session), LOGIN_TIMEOUT_SECONDS)
            await self._on_success(session.provider_id, tokens, tokens.get("email"))
            session.account = tokens.get("email")
            session.status = "done"
            session.message = "로그인 완료"
        except asyncio.CancelledError:
            session.status = "cancelled"
        except asyncio.TimeoutError:
            session.status = "error"
            session.message = "로그인 시간이 초과되었습니다."
        except Exception as exc:  # surfaced to the UI verbatim
            session.status = "error"
            session.message = str(exc)


# --------------------------------------------------------------------------- ChatGPT / Codex


def _codex_tokens(data: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    access = data["access_token"]
    id_token = data.get("id_token") or (previous or {}).get("id_token")
    claims = {**decode_jwt(id_token or ""), **decode_jwt(access)}
    auth = claims.get(CODEX_JWT_AUTH_CLAIM) or {}
    profile = claims.get("https://api.openai.com/profile") or {}
    return {
        "access_token": access,
        "refresh_token": data.get("refresh_token") or (previous or {}).get("refresh_token"),
        "id_token": id_token,
        "expires_at": time.time() + int(data.get("expires_in", 3600)),
        "account_id": auth.get("chatgpt_account_id") or (previous or {}).get("account_id"),
        "email": claims.get("email") or profile.get("email") or (previous or {}).get("email"),
    }


async def _codex_device_login(client: httpx.AsyncClient, session: LoginSession) -> dict[str, Any]:
    init = await client.post(CODEX_DEVICE_USERCODE_URL, json={"client_id": CODEX_CLIENT_ID})
    if init.status_code >= 400:
        raise OAuthError(f"ChatGPT 기기 인증 시작 실패: HTTP {init.status_code} {init.text[:300]}")
    data = init.json()
    session.user_code = data["user_code"]
    session.verification_url = CODEX_DEVICE_PAGE
    session.status = "pending"
    session.message = "브라우저에서 코드를 입력하고 승인하세요."
    interval = float(data.get("interval") or 5) + 1
    while True:
        await asyncio.sleep(interval)
        poll = await client.post(
            CODEX_DEVICE_TOKEN_URL, json={"device_auth_id": data["device_auth_id"], "user_code": data["user_code"]}
        )
        if poll.status_code in (403, 404):
            continue
        if poll.status_code >= 400:
            raise OAuthError(f"ChatGPT 기기 인증 확인 실패: HTTP {poll.status_code} {poll.text[:300]}")
        grant = poll.json()
        exchange = await client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CODEX_CLIENT_ID,
                "code": grant["authorization_code"],
                "code_verifier": grant["code_verifier"],
                "redirect_uri": CODEX_DEVICE_REDIRECT_URI,
            },
        )
        if exchange.status_code >= 400:
            raise OAuthError(f"ChatGPT 토큰 교환 실패: HTTP {exchange.status_code} {exchange.text[:300]}")
        tokens = _codex_tokens(exchange.json())
        if not tokens["account_id"]:
            raise OAuthError("토큰에 ChatGPT 계정 ID가 없습니다 (Plus/Pro 등 유료 구독 필요).")
        return tokens


async def codex_valid_tokens(
    client: httpx.AsyncClient, tokens: dict[str, Any] | None, save: SecretSaver
) -> dict[str, Any]:
    if not tokens or not tokens.get("access_token"):
        raise OAuthError("ChatGPT 로그인이 필요합니다.")
    if tokens.get("expires_at", 0) - REFRESH_SKEW_SECONDS > time.time():
        return tokens
    response = await client.post(
        CODEX_TOKEN_URL,
        json={
            "client_id": CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": tokens.get("refresh_token"),
            "scope": "openid profile email",
        },
    )
    if response.status_code >= 400:
        raise OAuthError(f"ChatGPT 토큰 갱신 실패 (다시 로그인하세요): HTTP {response.status_code} {response.text[:300]}")
    refreshed = _codex_tokens(response.json(), tokens)
    await save(refreshed)
    return refreshed


# --------------------------------------------------------------------------- xAI


async def _xai_token_endpoint(client: httpx.AsyncClient) -> str:
    response = await client.get(f"{XAI_ISSUER}/.well-known/openid-configuration")
    response.raise_for_status()
    endpoint = response.json()["token_endpoint"]
    host = httpx.URL(endpoint).host
    if not endpoint.startswith("https://") or not (host == "x.ai" or host.endswith(".x.ai")):
        raise OAuthError(f"예상치 못한 xAI 토큰 엔드포인트: {endpoint}")
    return endpoint


def _xai_tokens(data: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    access = data["access_token"]
    claims = decode_jwt(access)
    return {
        "access_token": access,
        "refresh_token": data.get("refresh_token") or (previous or {}).get("refresh_token"),
        "expires_at": time.time() + int(data.get("expires_in", 3600)),
        "subject": claims.get("sub") or (previous or {}).get("subject"),
        "email": claims.get("email") or (previous or {}).get("email"),
    }


async def _xai_device_login(client: httpx.AsyncClient, session: LoginSession) -> dict[str, Any]:
    token_url = await _xai_token_endpoint(client)
    init = await client.post(
        XAI_DEVICE_URL,
        data={"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPES},
        headers={"Accept": "application/json"},
    )
    if init.status_code >= 400:
        raise OAuthError(f"xAI 기기 인증 시작 실패: HTTP {init.status_code} {init.text[:300]}")
    data = init.json()
    session.user_code = data["user_code"]
    session.verification_url = data.get("verification_uri_complete") or data["verification_uri"]
    session.status = "pending"
    session.message = "브라우저에서 xAI 계정으로 승인하세요."
    interval = float(data.get("interval") or 5)
    while True:
        await asyncio.sleep(interval)
        poll = await client.post(
            token_url,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": data["device_code"],
                "client_id": XAI_CLIENT_ID,
            },
            headers={"Accept": "application/json"},
        )
        if poll.status_code == 200:
            tokens = _xai_tokens(poll.json())
            userinfo = await client.get(
                f"{XAI_ISSUER}/oauth2/userinfo", headers={"Authorization": f"Bearer {tokens['access_token']}"}
            )
            if userinfo.status_code == 200:
                tokens["email"] = userinfo.json().get("email") or tokens.get("email")
            return tokens
        error = poll.json().get("error") if poll.headers.get("content-type", "").startswith("application/json") else None
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        raise OAuthError(f"xAI 기기 인증 실패: HTTP {poll.status_code} {poll.text[:300]}")


async def xai_valid_tokens(client: httpx.AsyncClient, tokens: dict[str, Any] | None, save: SecretSaver) -> dict[str, Any]:
    if not tokens or not tokens.get("access_token"):
        raise OAuthError("xAI 로그인이 필요합니다.")
    if tokens.get("expires_at", 0) - REFRESH_SKEW_SECONDS > time.time():
        return tokens
    token_url = await _xai_token_endpoint(client)
    response = await client.post(
        token_url,
        data={"grant_type": "refresh_token", "refresh_token": tokens.get("refresh_token"), "client_id": XAI_CLIENT_ID},
        headers={"Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise OAuthError(f"xAI 토큰 갱신 실패 (다시 로그인하세요): HTTP {response.status_code} {response.text[:300]}")
    refreshed = _xai_tokens(response.json(), tokens)
    await save(refreshed)
    return refreshed
