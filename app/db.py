"""SQLite persistence for providers, jobs, and pages."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    auth TEXT NOT NULL,
    base_url TEXT,
    model TEXT NOT NULL,
    extra_body TEXT NOT NULL DEFAULT '{}',
    vision INTEGER NOT NULL DEFAULT 0,
    secret BLOB,
    account TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    engine TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    options TEXT NOT NULL DEFAULT '{}',
    instructions TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS pages (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    file TEXT NOT NULL,
    output TEXT,
    PRIMARY KEY (job_id, idx)
);
CREATE TABLE IF NOT EXISTS model_setup (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cache_dir TEXT,
    token BLOB
);
CREATE TABLE IF NOT EXISTS api_tokens (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    last_used_at REAL
);
"""

# Columns added after the first release; applied to existing databases on open.
MIGRATIONS = (
    ("jobs", "live", "INTEGER NOT NULL DEFAULT 0"),
    ("pages", "sha256", "TEXT"),
    ("pages", "error", "TEXT"),
    ("jobs", "llm", "TEXT NOT NULL DEFAULT '{}'"),
)

JOB_STATUSES = ("queued", "running", "done", "failed", "cancelled")


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        for table, column, definition in MIGRATIONS:
            columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self._conn.execute("CREATE INDEX IF NOT EXISTS pages_sha256 ON pages (sha256) WHERE output IS NOT NULL")
        self._lock = threading.Lock()

    def _exec(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _all(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # Model setup is one private, durable application preference.
    def get_model_setup(self) -> dict[str, Any] | None:
        return self._one("SELECT cache_dir, token FROM model_setup WHERE id = 1")

    def save_model_setup(self, cache_dir: str | None, token: bytes | None) -> None:
        self._exec(
            """INSERT INTO model_setup (id, cache_dir, token) VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE SET cache_dir=excluded.cache_dir, token=excluded.token""",
            (cache_dir, token),
        )

    # providers -------------------------------------------------------------
    def list_providers(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM providers ORDER BY created_at")

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM providers WHERE id = ?", (provider_id,))

    def upsert_provider(self, row: dict[str, Any]) -> None:
        row = {**row, "extra_body": json.dumps(row.get("extra_body") or {})}
        self._exec(
            """INSERT INTO providers (id, name, kind, auth, base_url, model, extra_body, vision, secret, account, created_at)
               VALUES (:id, :name, :kind, :auth, :base_url, :model, :extra_body, :vision, :secret, :account, :created_at)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, kind=excluded.kind, auth=excluded.auth,
                 base_url=excluded.base_url, model=excluded.model, extra_body=excluded.extra_body,
                 vision=excluded.vision, secret=excluded.secret, account=excluded.account""",
            row,
        )

    def set_provider_secret(self, provider_id: str, secret: bytes | None, account: str | None) -> None:
        self._exec("UPDATE providers SET secret = ?, account = ? WHERE id = ?", (secret, account, provider_id))

    def delete_provider(self, provider_id: str) -> None:
        self._exec("DELETE FROM providers WHERE id = ?", (provider_id,))
    # extension API tokens (only SHA-256 hashes are stored) ---------------------
    def list_api_tokens(self) -> list[dict[str, Any]]:
        return self._all("SELECT id, name, created_at, last_used_at FROM api_tokens ORDER BY created_at")

    def create_api_token(self, token_id: str, name: str, token_hash: str) -> None:
        self._exec(
            "INSERT INTO api_tokens (id, name, token_hash, created_at) VALUES (?, ?, ?, ?)",
            (token_id, name, token_hash, time.time()),
        )

    def use_api_token(self, token_hash: str) -> bool:
        return self._exec(
            "UPDATE api_tokens SET last_used_at = ? WHERE token_hash = ?", (time.time(), token_hash)
        ).rowcount > 0

    def delete_api_token(self, token_id: str) -> None:
        self._exec("DELETE FROM api_tokens WHERE id = ?", (token_id,))

    # jobs ------------------------------------------------------------------
    def create_job(self, job: dict[str, Any], pages: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    """INSERT INTO jobs (id, title, engine, provider_id, options, instructions, status, message, live, llm, created_at)
                       VALUES (:id, :title, :engine, :provider_id, :options, :instructions, :status, :message, :live, :llm, :created_at)""",
                    {"status": "queued", "message": "", "live": 0, **job, "options": json.dumps(job["options"]), "llm": json.dumps(job.get("llm") or {}, sort_keys=True)},
                )
                self._conn.executemany(
                    "INSERT INTO pages (job_id, idx, source_name, file, sha256) VALUES (:job_id, :idx, :source_name, :file, :sha256)",
                    [{"sha256": None, **page} for page in pages],
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM jobs ORDER BY created_at DESC")

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))

    def next_queued_job(self) -> dict[str, Any] | None:
        # Live (browser extension) sessions wait on a reader, so they go first.
        return self._one("SELECT * FROM jobs WHERE status = 'queued' ORDER BY live DESC, created_at LIMIT 1")

    def find_live_job(self, engine: str, provider_id: str, options: str, instructions: str, llm: str, since: float) -> dict[str, Any] | None:
        return self._one(
            """SELECT * FROM jobs WHERE live = 1 AND engine = ? AND provider_id = ? AND options = ? AND instructions = ?
               AND llm = ? AND created_at >= ? ORDER BY created_at DESC LIMIT 1""",
            (engine, provider_id, options, instructions, llm, since),
        )

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = :{key}" for key in fields)
        self._exec(f"UPDATE jobs SET {assignments} WHERE id = :id", {**fields, "id": job_id})

    def set_job_status(self, job_id: str, status: str, **fields: Any) -> None:
        assert status in JOB_STATUSES
        now = time.time()
        if status == "running":
            fields.setdefault("started_at", now)
        if status in ("done", "failed", "cancelled"):
            fields.setdefault("finished_at", now)
        self.update_job(job_id, status=status, **fields)

    def reset_interrupted_jobs(self) -> None:
        """Jobs left running by a previous process are failed, not silently resumed.

        Live sessions have a reader waiting per page, so their unfinished pages
        get an explicit error instead of hanging until the next upload.
        """
        with self._lock:
            self._conn.execute(
                """UPDATE pages SET error = '서버가 재시작되어 번역이 중단되었습니다.'
                   WHERE output IS NULL AND error IS NULL
                   AND job_id IN (SELECT id FROM jobs WHERE live = 1 AND status IN ('queued', 'running'))"""
            )
            self._conn.execute(
                "UPDATE jobs SET status='done', message='서버 재시작', finished_at=? WHERE live = 1 AND status = 'queued'",
                (time.time(),),
            )
            self._conn.execute(
                "UPDATE jobs SET status='failed', error='서버가 재시작되어 작업이 중단되었습니다.', finished_at=? WHERE status='running'",
                (time.time(),),
            )

    def delete_job(self, job_id: str) -> None:
        self._exec("DELETE FROM jobs WHERE id = ?", (job_id,))

    def list_pages(self, job_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM pages WHERE job_id = ? ORDER BY idx", (job_id,))

    def set_page_output(self, job_id: str, idx: int, output: str | None) -> None:
        self._exec("UPDATE pages SET output = ? WHERE job_id = ? AND idx = ?", (output, job_id, idx))

    def add_page(self, page: dict[str, Any]) -> None:
        self._exec(
            """INSERT INTO pages (job_id, idx, source_name, file, sha256, output, error)
               VALUES (:job_id, :idx, :source_name, :file, :sha256, :output, :error)""",
            {"output": None, "error": None, **page},
        )

    def next_page_idx(self, job_id: str) -> int:
        row = self._one("SELECT COALESCE(MAX(idx), 0) + 1 AS idx FROM pages WHERE job_id = ?", (job_id,))
        assert row is not None
        return int(row["idx"])

    def find_translated_page(self, sha256: str, provider_id: str, llm: str) -> dict[str, Any] | None:
        return self._one(
            """SELECT pages.* FROM pages JOIN jobs ON jobs.id = pages.job_id
               WHERE pages.sha256 = ? AND pages.output IS NOT NULL AND jobs.provider_id = ? AND jobs.llm = ?
               ORDER BY pages.rowid DESC LIMIT 1""",
            (sha256, provider_id, llm),
        )

    def set_page_error(self, job_id: str, idx: int, error: str | None) -> None:
        self._exec("UPDATE pages SET error = ? WHERE job_id = ? AND idx = ?", (error, job_id, idx))
