"""Job ingestion, the single GPU worker, and result packaging."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import threading
import tempfile
import time
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from .db import Database
from .engines import Engine, EngineCancelled, EngineRun
from .prompting import build_instructions

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ARCHIVE_EXTENSIONS = {".zip", ".cbz"}
PDF_RENDER_MAX_SIDE = 3000
MAX_ARCHIVE_MEMBERS = 5000
MAX_IMAGE_BYTES = 80 * 1024 * 1024


class IngestError(ValueError):
    pass


def natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def collect_images(uploads: list[tuple[str, BinaryIO]]) -> list[tuple[str, bytes]]:
    """Flatten uploaded images and ZIP/CBZ archives into (display name, bytes), natural-sorted.

    Uploads are ordered by file name; archive members keep their in-archive
    order by natural path sort. Non-image members are ignored.
    """
    images: list[tuple[str, bytes]] = []
    for filename, stream in sorted(uploads, key=lambda item: natural_key(item[0])):
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix in ARCHIVE_EXTENSIONS:
            images.extend(_archive_images(filename, stream.read()))
        elif suffix == ".pdf":
            images.extend(_pdf_images(filename, stream.read()))
        elif suffix in IMAGE_EXTENSIONS:
            images.append((PurePosixPath(filename).name, _checked_image(filename, stream.read())))
        else:
            raise IngestError(f"지원하지 않는 파일 형식입니다: {filename}")
    if not images:
        raise IngestError("이미지가 없습니다.")
    return images


def _archive_images(filename: str, data: bytes) -> list[tuple[str, bytes]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise IngestError(f"손상된 압축 파일입니다: {filename}") from exc
    members = [info for info in archive.infolist() if not info.is_dir()]
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise IngestError(f"압축 파일 항목이 너무 많습니다: {filename}")
    result: list[tuple[str, bytes]] = []
    for info in sorted(members, key=lambda item: natural_key(item.filename)):
        path = PurePosixPath(info.filename)
        if path.name.startswith(".") or "__MACOSX" in path.parts or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if info.file_size > MAX_IMAGE_BYTES:
            raise IngestError(f"이미지가 너무 큽니다: {info.filename}")
        result.append((str(path), _checked_image(info.filename, archive.read(info))))
    return result


def _pdf_images(filename: str, data: bytes) -> list[tuple[str, bytes]]:
    """One PNG per PDF page.

    Scanned manga PDFs usually hold one full-page raster per page; that image is
    taken at its native resolution (no resampling). Other pages are rendered at
    300 DPI, capped at PDF_RENDER_MAX_SIDE pixels on the long side.
    """
    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise IngestError(f"PDF를 열 수 없습니다: {filename} ({exc})") from exc
    stem = PurePosixPath(filename).stem
    result: list[tuple[str, bytes]] = []
    try:
        if len(document) > MAX_ARCHIVE_MEMBERS:
            raise IngestError(f"PDF 쪽수가 너무 많습니다: {filename}")
        for index in range(len(document)):
            page = document[index]
            try:
                image = _pdf_page_raster(page) or _pdf_page_render(page)
            finally:
                page.close()
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "PNG")
            result.append((f"{stem}/p{index + 1:04d}.png", buffer.getvalue()))
    finally:
        document.close()
    return result


def _pdf_page_raster(page: "pdfium.PdfPage") -> Image.Image | None:
    width, height = page.get_size()
    images = list(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=2))
    if len(images) != 1:
        return None
    left, bottom, right, top = images[0].get_bounds()
    if (right - left) * (top - bottom) < 0.9 * width * height:
        return None
    try:
        return images[0].get_bitmap(render=False).to_pil()
    except pdfium.PdfiumError:
        return None


def _pdf_page_render(page: "pdfium.PdfPage") -> Image.Image:
    width, height = page.get_size()
    scale = min(300 / 72, PDF_RENDER_MAX_SIDE / max(width, height))
    return page.render(scale=scale).to_pil()


def _checked_image(name: str, data: bytes) -> bytes:
    if len(data) > MAX_IMAGE_BYTES:
        raise IngestError(f"이미지가 너무 큽니다: {name}")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise IngestError(f"이미지를 읽을 수 없습니다: {name}") from exc
    return data


def image_suffix(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        return {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(image.format or "", ".png")


class JobManager:
    """Owns job directories and runs one job at a time on a background thread."""

    def __init__(
        self,
        db: Database,
        jobs_dir: Path,
        engines: dict[str, Engine],
        llm_endpoint: Callable[[], tuple[str, str]],
        live_idle_seconds: int = 180,
    ) -> None:
        self.db = db
        self.jobs_dir = jobs_dir
        self.engines = engines
        self._llm_endpoint = llm_endpoint
        self.live_idle_seconds = live_idle_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._current: tuple[str, Engine] | None = None
        # Input folder and run of the live session the engine is watching now.
        self._live: tuple[Path, EngineRun] | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)

    # lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        self.db.reset_interrupted_jobs()
        self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            if self._current:
                self._current[1].cancel()

    # paths ---------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def input_path(self, job_id: str, page: dict[str, Any]) -> Path:
        return self.job_dir(job_id) / "input" / page["file"]

    def output_path(self, job_id: str, page: dict[str, Any]) -> Path | None:
        return self.job_dir(job_id) / "output" / page["output"] if page.get("output") else None

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    # commands ------------------------------------------------------------------
    def create(
        self,
        *,
        title: str,
        engine_id: str,
        provider_id: str,
        options: dict[str, Any],
        instructions: str,
        images: list[tuple[str, bytes]],
    ) -> str:
        engine = self.engines.get(engine_id)
        if not engine:
            raise IngestError(f"알 수 없는 엔진: {engine_id}")
        available, reason = engine.availability()
        if not available:
            raise IngestError(f"{engine.name} 엔진을 사용할 수 없습니다: {reason}")
        resolved = engine.resolve_options(options)
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        input_dir = self.job_dir(job_id) / "input"
        input_dir.mkdir(parents=True)
        pages = []
        for index, (name, data) in enumerate(images, start=1):
            file = f"{index:04d}{image_suffix(data)}"
            (input_dir / file).write_bytes(data)
            pages.append({"job_id": job_id, "idx": index, "source_name": name, "file": file})
        self.db.create_job(
            {
                "id": job_id,
                "title": title,
                "engine": engine_id,
                "provider_id": provider_id,
                "options": resolved,
                "instructions": instructions,
                "created_at": time.time(),
            },
            pages,
        )
        with self._lock:
            self._release_idle_live()
        self._wake.set()
        return job_id

    # live sessions (browser extension) -------------------------------------------
    def open_live(
        self, *, engine_id: str, provider_id: str, options: dict[str, Any], instructions: str, title: str,
        llm: dict[str, str] | None = None,
    ) -> str:
        """Reuse today's live session for the same settings, so the engine stays warm across tabs.

        `llm` holds the session's model / reasoning-level choice ({} = provider defaults).
        """
        engine = self.engines.get(engine_id)
        if not engine:
            raise IngestError(f"알 수 없는 엔진: {engine_id}")
        if not engine.supports_live:
            raise IngestError(f"{engine.name} 엔진은 실시간 번역을 지원하지 않습니다.")
        available, reason = engine.availability()
        if not available:
            raise IngestError(f"{engine.name} 엔진을 사용할 수 없습니다: {reason}")
        resolved = engine.resolve_options(options)
        today = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        with self._lock:
            existing = self.db.find_live_job(
                engine_id, provider_id, json.dumps(resolved), instructions, json.dumps(llm or {}, sort_keys=True), today
            )
            if existing:
                return existing["id"]
            job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            (self.job_dir(job_id) / "input").mkdir(parents=True)
            self.db.create_job(
                {
                    "id": job_id,
                    "title": title,
                    "engine": engine_id,
                    "provider_id": provider_id,
                    "options": resolved,
                    "instructions": instructions,
                    "status": "done",
                    "message": "페이지 대기 중",
                    "live": 1,
                    "llm": llm or {},
                    "created_at": time.time(),
                },
                [],
            )
        return job_id

    def add_live_page(self, job_id: str, name: str, data: bytes) -> dict[str, Any]:
        """Append one page; reuse an earlier translation of identical bytes, else feed the engine."""
        _checked_image(name, data)
        try:
            suffix = image_suffix(data)
        except Exception as exc:
            raise IngestError(f"이미지를 읽을 수 없습니다: {name}") from exc
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            job = self.db.get_job(job_id)
            if not job or not job["live"]:
                raise KeyError(job_id)
            idx = self.db.next_page_idx(job_id)
            file = f"{idx:04d}{suffix}"
            (self.job_dir(job_id) / "input" / file).write_bytes(data)
            output = None
            # Reuse only translations made with the same provider and model/reasoning choice.
            cached = self.db.find_translated_page(digest, job["provider_id"], job["llm"])
            source = self.output_path(cached["job_id"], cached) if cached else None
            if source and source.exists():
                output_dir = self.job_dir(job_id) / "output"
                output_dir.mkdir(exist_ok=True)
                output = f"{idx:04d}{source.suffix}"
                shutil.copyfile(source, output_dir / output)
            page = {"job_id": job_id, "idx": idx, "source_name": name[:500], "file": file, "sha256": digest, "output": output}
            self.db.add_page(page)
            if output is None:
                if self._current and self._current[0] == job_id:
                    # Running: link into the watched folder. Outside that window
                    # the page is listed at start or requeued by _finish_live.
                    if self._live:
                        (self._live[0] / file).symlink_to(self.input_path(job_id, page).resolve())
                else:
                    # Another session's engine may be idling in watch mode; let it exit now.
                    self._release_idle_live()
                    if job["status"] != "queued":
                        self.db.set_job_status(job_id, "queued", message="대기 중", error=None)
                        self._wake.set()
        return {"idx": idx, "translated": output is not None, "error": None}

    def _release_idle_live(self) -> None:
        """Ask the watching engine to exit once its pending pages are done (caller holds the lock).

        ct_batch only checks the marker when it has nothing left, so queued pages still finish.
        """
        if self._live:
            (self._live[0] / ".closed").touch()

    def discard_live_pages(self, job_id: str, indices: set[int]) -> None:
        """Drop pages nobody waits for any more, unless the engine already started them."""
        with self._lock:
            live = self._live if self._current and self._current[0] == job_id else None
            for page in self.db.list_pages(job_id):
                if page["idx"] not in indices or page["output"] or page["error"]:
                    continue
                if live:
                    if page["file"] in live[1].started_files:
                        continue
                    (live[0] / page["file"]).unlink(missing_ok=True)
                self.db.set_page_error(job_id, page["idx"], "취소됨")

    def cancel(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        if job["status"] == "queued":
            self.db.set_job_status(job_id, "cancelled", message="취소됨")
            return
        with self._lock:
            if self._current and self._current[0] == job_id:
                self._current[1].cancel()

    def retry(self, job_id: str, *, failed_only: bool = False, provider_id: str | None = None) -> None:
        job = self.db.get_job(job_id)
        if not job or job["status"] not in ("failed", "cancelled", "done"):
            raise IngestError("대기 중이거나 실행 중인 작업은 다시 실행할 수 없습니다.")
        if provider_id is not None and not self.db.get_provider(provider_id):
            raise IngestError("선택한 LLM 제공자를 찾을 수 없습니다.")
        output_dir = self.job_dir(job_id) / "output"
        if failed_only:
            done = self._collect_outputs(job_id, output_dir)
            total = len(self.db.list_pages(job_id))
            if done == total:
                raise IngestError("재시도할 실패·미완료 페이지가 없습니다.")
        else:
            shutil.rmtree(output_dir, ignore_errors=True)
            for page in self.db.list_pages(job_id):
                self.db.set_page_output(job_id, page["idx"], None)
            done = 0
            total = len(self.db.list_pages(job_id))
        (self.job_dir(job_id) / "translated.cbz").unlink(missing_ok=True)
        for page in self.db.list_pages(job_id):
            if not page["output"] and page["error"]:
                self.db.set_page_error(job_id, page["idx"], None)
        switched = provider_id is not None and provider_id != job["provider_id"]
        self.db.update_job(
            job_id, status="queued", progress=done / max(total, 1),
            message=f"완료 {done}쪽 보존 · {total - done}쪽 재시도 대기",
            error=None, started_at=None, finished_at=None,
            provider_id=provider_id if provider_id is not None else job["provider_id"],
            # A model/reasoning choice belongs to the old provider.
            llm="{}" if switched else job["llm"],
        )
        self._wake.set()

    def delete(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job and job["status"] == "running":
            raise IngestError("실행 중인 작업은 먼저 취소하세요.")
        self.db.delete_job(job_id)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def archive(self, job_id: str) -> Path:
        """CBZ of translated pages in reading order (built on demand, cached)."""
        pages = self.db.list_pages(job_id)
        target = self.job_dir(job_id) / "translated.cbz"
        outputs = [(page, self.output_path(job_id, page)) for page in pages]
        missing = [page["source_name"] for page, path in outputs if not path or not path.exists()]
        if missing:
            raise IngestError(f"번역되지 않은 페이지가 있습니다: {', '.join(missing[:5])}")
        newest = max(path.stat().st_mtime for _, path in outputs if path)
        if not target.exists() or target.stat().st_mtime < newest:
            tmp = target.with_suffix(".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
                for page, path in outputs:
                    assert path is not None
                    archive.write(path, f"{page['idx']:04d}{path.suffix}")
            tmp.replace(target)
        return target

    # worker --------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=5)
            self._wake.clear()
            while not self._stop.is_set():
                job = self.db.next_queued_job()
                if not job:
                    break
                self._run(job)

    def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        live = bool(job["live"])
        engine = self.engines[job["engine"]]
        with self._lock:
            self._current = (job_id, engine)
        self.db.set_job_status(job_id, "running", progress=0.0, message="시작 중", error=None)
        output_dir = self.job_dir(job_id) / "output"
        base_url, api_key = self._llm_endpoint()
        run: EngineRun | None = None
        try:
            done = self._collect_outputs(job_id, output_dir)
            # Both engines keep input stems in their output names. Only expose
            # missing pages, so retries cannot overwrite successful results.
            with tempfile.TemporaryDirectory(prefix="pending-", dir=self.job_dir(job_id)) as directory:
                input_dir = Path(directory)
                pages = self.db.list_pages(job_id)
                run = EngineRun(
                    job_id=job_id,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    log_path=self.log_path(job_id),
                    llm_base_url=base_url,
                    llm_api_key=api_key,
                    llm_model=f"job:{job_id}",
                    instructions=build_instructions(job["instructions"]),
                    options=engine.resolve_options(json.loads(job["options"])),
                    report=lambda progress, message: self._report(
                        job_id, output_dir,
                        None if live else (done + progress * len(pending)) / max(len(pages), 1), message,
                    ),
                    watch_idle_seconds=self.live_idle_seconds if live else None,
                )
                # Under the lock, so a live page added concurrently is either
                # listed here or linked into input_dir by add_live_page.
                with self._lock:
                    pages = self.db.list_pages(job_id)
                    pending = [page for page in pages if not page["output"] and not (live and page["error"])]
                    for page in pending:
                        (input_dir / page["file"]).symlink_to(self.input_path(job_id, page).resolve())
                    if live:
                        self._live = (input_dir, run)
                try:
                    if pending:
                        engine.run(run)
                finally:
                    with self._lock:
                        self._live = None
            done = self._collect_outputs(job_id, output_dir)
            if live:
                self._finish_live(job_id, run)
                return
            total = len(self.db.list_pages(job_id))
            if done < total:
                detail = "; ".join(run.page_errors[:3])
                raise RuntimeError(f"{total}쪽 중 {done}쪽만 번역되었습니다. {detail}".strip())
            self.db.set_job_status(job_id, "done", progress=1.0, message=f"{done}쪽 완료")
        except EngineCancelled:
            self._collect_outputs(job_id, output_dir)
            if live:
                self._fail_unfinished(job_id, "취소됨")
            self.db.set_job_status(job_id, "cancelled", message="취소됨")
        except Exception as exc:
            done = self._collect_outputs(job_id, output_dir)
            total = len(self.db.list_pages(job_id))
            with open(self.log_path(job_id), "a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
            if live:
                self._fail_unfinished(job_id, str(exc))
            self.db.set_job_status(
                job_id, "failed", error=str(exc),
                message=f"{done}/{total}쪽 완료 · 실패·미완료 페이지를 재시도할 수 있습니다.",
            )
        finally:
            with self._lock:
                self._current = None

    def _finish_live(self, job_id: str, run: EngineRun) -> None:
        """After the engine went idle: record page errors, requeue pages it never saw."""
        unseen = 0
        with self._lock:
            for page in self.db.list_pages(job_id):
                if page["output"] or page["error"]:
                    continue
                if page["file"] in run.failed_files:
                    self.db.set_page_error(job_id, page["idx"], run.failed_files[page["file"]][:500] or "번역 실패")
                elif page["file"] in run.started_files:
                    self.db.set_page_error(job_id, page["idx"], "엔진이 결과 이미지를 만들지 못했습니다.")
                else:
                    # Linked after the engine's last scan of its input folder.
                    unseen += 1
            pages = self.db.list_pages(job_id)
            translated = sum(1 for page in pages if page["output"])
            if unseen:
                self.db.set_job_status(job_id, "queued", message=f"대기 중 · {unseen}쪽")
                self._wake.set()
            else:
                self.db.set_job_status(job_id, "done", progress=1.0, message=f"대기 중 · {translated}/{len(pages)}쪽 완료")

    def _fail_unfinished(self, job_id: str, error: str) -> None:
        for page in self.db.list_pages(job_id):
            if not page["output"] and not page["error"]:
                self.db.set_page_error(job_id, page["idx"], error[:500])

    def _report(self, job_id: str, output_dir: Path, progress: float | None, message: str) -> None:
        # Engines write pages as they finish; reflect them so the UI shows
        # translated pages during long jobs instead of only at the end.
        self._collect_outputs(job_id, output_dir)
        if progress is None:
            self.db.update_job(job_id, message=message[:500])
        else:
            self.db.update_job(job_id, progress=max(0.0, min(1.0, progress)), message=message[:500])

    def _collect_outputs(self, job_id: str, output_dir: Path) -> int:
        """Map engine outputs (`0001.png` or `0001_translated.png`) back to pages."""
        by_stem: dict[str, str] = {}
        for path in output_dir.iterdir() if output_dir.exists() else ():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                by_stem[path.stem.removesuffix("_translated")] = path.name
        done = 0
        for page in self.db.list_pages(job_id):
            name = by_stem.get(Path(page["file"]).stem)
            self.db.set_page_output(job_id, page["idx"], name)
            done += name is not None
        return done
