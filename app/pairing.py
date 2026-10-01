"""Browser-extension pairing: the extension asks, a logged-in user approves, a device token is issued.

Requests live in memory only (a restart simply drops pending requests). The
extension proves it started a request with a random secret; the short code is
only for the user to compare the request in the web UI with the one shown in
the extension before approving.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

TTL_SECONDS = 300
MAX_PENDING = 5


class PairingError(RuntimeError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class _Request:
    id: str
    name: str
    code: str
    secret_hash: str
    expires_at: float
    status: str = "pending"  # pending | approved | denied
    device_id: str | None = None
    token: str | None = None


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Pairing:
    def __init__(self, issue: Callable[[str], tuple[str, str]], revoke: Callable[[str], None]) -> None:
        """`issue(name)` creates a device and returns (device id, plain token); `revoke(device id)` removes it."""
        self._issue = issue
        self._revoke = revoke
        self._requests: dict[str, _Request] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.time()
        for request in [r for r in self._requests.values() if r.expires_at <= now]:
            del self._requests[request.id]
            if request.status == "approved" and request.device_id:
                # Approved but never collected: do not leave an unused credential behind.
                self._revoke(request.device_id)

    def start(self, name: str) -> dict[str, Any]:
        name = " ".join(name.split())[:100] or "브라우저 확장"
        with self._lock:
            self._prune()
            if sum(1 for r in self._requests.values() if r.status == "pending") >= MAX_PENDING:
                raise PairingError("대기 중인 연결 요청이 너무 많습니다. 잠시 후 다시 시도하세요.", 429)
            secret = secrets.token_urlsafe(32)
            request = _Request(
                id=uuid.uuid4().hex,
                name=name,
                code=f"{secrets.randbelow(10**6):06d}",
                secret_hash=_hash(secret),
                expires_at=time.time() + TTL_SECONDS,
            )
            self._requests[request.id] = request
        return {"id": request.id, "code": request.code, "secret": secret, "expires_in": TTL_SECONDS}

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            self._prune()
            return [
                {"id": r.id, "name": r.name, "code": r.code, "expires_at": r.expires_at}
                for r in self._requests.values()
                if r.status == "pending"
            ]

    def _pending_request(self, request_id: str) -> _Request:
        self._prune()
        request = self._requests.get(request_id)
        if not request or request.status != "pending":
            raise PairingError("연결 요청이 없거나 만료되었습니다.", 404)
        return request

    def approve(self, request_id: str) -> str:
        with self._lock:
            request = self._pending_request(request_id)
            request.device_id, request.token = self._issue(request.name)
            request.status = "approved"
            return request.name

    def deny(self, request_id: str) -> None:
        with self._lock:
            self._pending_request(request_id).status = "denied"

    def poll(self, request_id: str, secret: str) -> dict[str, Any]:
        with self._lock:
            self._prune()
            request = self._requests.get(request_id)
            if not request or not hmac.compare_digest(request.secret_hash, _hash(secret)):
                return {"status": "expired"}
            if request.status == "pending":
                return {"status": "pending", "expires_at": request.expires_at}
            # Approved or denied is reported exactly once; the token leaves memory here.
            del self._requests[request_id]
            if request.status == "denied":
                return {"status": "denied"}
            return {"status": "approved", "token": request.token, "name": request.name}
