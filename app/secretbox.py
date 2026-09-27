"""At-rest encryption for provider API keys and OAuth tokens."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


class SecretBox:
    def __init__(self, key_path: Path) -> None:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if not key_path.exists():
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(Fernet.generate_key())
        self._fernet = Fernet(key_path.read_bytes().strip())

    def seal(self, value: dict[str, Any]) -> bytes:
        return self._fernet.encrypt(json.dumps(value).encode())

    def open(self, blob: bytes | None) -> dict[str, Any] | None:
        if not blob:
            return None
        return json.loads(self._fernet.decrypt(blob))


def read_or_create_token(path: Path) -> str:
    """Random bearer token persisted with 0600 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text().strip()
    token = os.urandom(24).hex()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    return token
