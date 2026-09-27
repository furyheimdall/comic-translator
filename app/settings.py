"""Runtime paths and environment configuration."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    host: str
    port: int
    password: str | None
    public_url: str
    mt_dir: Path
    mt_python: Path
    koharu_bin: Path
    koharu_font_family: str
    font_dir: Path
    hf_token_file: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def llm_base_url(self) -> str:
        """Base URL the engines use to reach the internal OpenAI-compatible proxy."""
        return f"http://127.0.0.1:{self.port}/llm/v1"

    def is_loopback_only(self) -> bool:
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return self.host == "localhost"


def load_settings() -> Settings:
    data_dir = _path("CT_DATA_DIR", ROOT / "data")
    port = int(os.environ.get("CT_PORT", "8710"))
    return Settings(
        data_dir=data_dir,
        host=os.environ.get("CT_HOST", "127.0.0.1"),
        port=port,
        password=os.environ.get("CT_PASSWORD") or None,
        public_url=os.environ.get("CT_PUBLIC_URL", f"http://127.0.0.1:{port}"),
        mt_dir=_path("CT_MT_DIR", ROOT / "engines" / "mangatranslator"),
        mt_python=_path("CT_MT_PYTHON", ROOT / "engines" / "mangatranslator" / ".venv" / "bin" / "python"),
        koharu_bin=_path("CT_KOHARU_BIN", ROOT / "engines" / "koharu-target" / "release" / "ct_batch"),
        koharu_font_family=os.environ.get("CT_KOHARU_FONT", "Noto Sans CJK KR"),
        font_dir=_path("CT_FONT_DIR", ROOT / "fonts" / "korean"),
        hf_token_file=_path(
            "CT_HF_TOKEN_FILE",
            Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser() / "token",
        ),
    )
