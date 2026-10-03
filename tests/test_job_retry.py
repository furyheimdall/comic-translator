import io
import zipfile
import shutil

import pytest
from PIL import Image

from app.db import Database
from app.jobs import IngestError, JobManager
from engines.mt_cache import _Page


def image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, "PNG")
    return buffer.getvalue()


class SelectiveEngine:
    name = "test"

    def __init__(self):
        self.fail = {"0002", "0004"}

    def availability(self):
        return True, ""

    def resolve_options(self, options):
        return options

    def run(self, run):
        run.output_dir.mkdir(exist_ok=True)
        for source in sorted(run.input_dir.iterdir()):
            if source.stem in self.fail:
                continue
            # Existing outputs must never even be presented to the engine.
            output = run.output_dir / f"{source.stem}_translated.png"
            assert not output.exists(), f"Completed page was resubmitted: {source.name}"
            output.write_bytes(source.read_bytes())
        run.report(1, "processed")
        if self.fail:
            raise RuntimeError("page translation failed")


@pytest.fixture
def job(tmp_path):
    db = Database(tmp_path / "app.db")
    engine = SelectiveEngine()
    manager = JobManager(db, tmp_path / "jobs", {"test": engine}, lambda: ("http://unused", "secret"))
    job_id = manager.create(title="retry", engine_id="test", provider_id="test", options={}, instructions="", images=[
        (f"page{i}.png", image_bytes(color)) for i, color in enumerate(["red", "green", "blue", "white"], 1)
    ])
    manager._run(db.get_job(job_id))
    yield db, engine, manager, job_id
    db._conn.close()


def test_partial_retry_preserves_successes_across_repeated_failures(job):
    db, engine, manager, job_id = job
    assert db.get_job(job_id)["status"] == "failed"
    output_dir = manager.job_dir(job_id) / "output"
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output_dir.iterdir()}
    assert set(before) == {"0001_translated.png", "0003_translated.png"}
    engine.fail = {"0004"}
    manager.retry(job_id, failed_only=True)
    manager._run(db.get_job(job_id))
    assert db.get_job(job_id)["status"] == "failed"
    assert [p["idx"] for p in db.list_pages(job_id) if not p["output"]] == [4]
    engine.fail = set()
    manager.retry(job_id, failed_only=True)
    manager._run(db.get_job(job_id))
    assert db.get_job(job_id)["status"] == "done"
    for name, original in before.items():
        path = output_dir / name
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    with zipfile.ZipFile(manager.archive(job_id)) as archive:
        assert archive.namelist() == ["0001.png", "0002.png", "0003.png", "0004.png"]
        for index, page in enumerate(db.list_pages(job_id), 1):
            assert archive.read(f"{index:04}.png") == manager.input_path(job_id, page).read_bytes()
    with pytest.raises(IngestError, match="없습니다"):
        manager.retry(job_id, failed_only=True)


def test_partial_retry_rejects_active_job_without_deleting_outputs(job):
    db, engine, manager, job_id = job
    db.set_job_status(job_id, "running")
    with pytest.raises(IngestError):
        manager.retry(job_id, failed_only=True)
    assert (manager.job_dir(job_id) / "output" / "0001_translated.png").exists()
    assert db.get_job(job_id)["status"] == "running"


@pytest.mark.parametrize("entire_directory", [False, True])
def test_partial_retry_recovers_missing_output_file(job, entire_directory):
    db, engine, manager, job_id = job
    engine.fail = set()
    manager.retry(job_id, failed_only=True)
    manager._run(db.get_job(job_id))
    page = db.list_pages(job_id)[0]
    if entire_directory:
        shutil.rmtree(manager.job_dir(job_id) / "output")
    else:
        manager.output_path(job_id, page).unlink()
    manager.retry(job_id, failed_only=True)
    assert db.list_pages(job_id)[0]["output"] is None
    manager._run(db.get_job(job_id))
    assert db.get_job(job_id)["status"] == "done"
    assert manager.output_path(job_id, db.list_pages(job_id)[0]).exists()


def test_full_restart_still_discards_old_results_and_archive(job):
    db, engine, manager, job_id = job
    engine.fail = set()
    manager.retry(job_id, failed_only=True)
    manager._run(db.get_job(job_id))
    archive = manager.archive(job_id)
    manager.retry(job_id)
    assert not archive.exists()
    assert not (manager.job_dir(job_id) / "output").exists()
    assert all(page["output"] is None for page in db.list_pages(job_id))
    manager._run(db.get_job(job_id))
    assert db.get_job(job_id)["status"] == "done"


@pytest.mark.parametrize("failed_only", [False, True])
def test_retry_switches_provider_before_queueing_and_respects_output_mode(job, failed_only):
    db, engine, manager, job_id = job
    db.upsert_provider({
        "id": "replacement", "name": "new LLM", "kind": "openai_compatible",
        "auth": "none", "base_url": "http://unused", "model": "new-model",
        "extra_body": {}, "vision": False, "secret": None, "account": None, "created_at": 0,
    })
    output = manager.job_dir(job_id) / "output" / "0001_translated.png"
    original = output.read_bytes()
    manager.retry(job_id, failed_only=failed_only, provider_id="replacement")
    queued = db.next_queued_job()
    assert queued["id"] == job_id
    assert queued["provider_id"] == "replacement"
    if failed_only:
        assert output.read_bytes() == original
    else:
        assert not output.exists()
    engine.fail = set()
    manager._run(queued)
    assert db.get_job(job_id)["status"] == "done"
    assert db.get_job(job_id)["provider_id"] == "replacement"


@pytest.mark.parametrize("failed_only", [False, True])
def test_invalid_replacement_provider_does_not_destroy_results_or_queue(job, failed_only):
    db, _engine, manager, job_id = job
    before = db.get_job(job_id)
    output = manager.job_dir(job_id) / "output" / "0001_translated.png"
    original = output.read_bytes()
    with pytest.raises(IngestError, match="LLM"):
        manager.retry(job_id, failed_only=failed_only, provider_id="missing")
    assert db.get_job(job_id) == before
    assert output.read_bytes() == original


@pytest.mark.parametrize("engine_id,payload_name", [
    ("koharu", "session/erased.png"),
    ("mangatranslator", "payload.zip"),
])
def test_identical_uploaded_page_reuses_private_cache_without_exporting_it(tmp_path, engine_id, payload_name):
    db = Database(tmp_path / "app.db")
    engine = SelectiveEngine()
    engine.fail = set()
    manager = JobManager(db, tmp_path / "jobs", {engine_id: engine}, lambda: ("http://unused", "secret"))
    image = image_bytes("white")

    def create(name, content, options=None):
        return manager.create(
            title=name, engine_id=engine_id, provider_id=name,
            options=options or {"ocr": "paddleocr-vl-1.6"}, instructions="",
            images=[(name + ".png", content)],
        )

    first = create("first-model", image)
    first_page = db.list_pages(first)[0]
    first_cache = manager.job_dir(first) / "private-cache" / "0001"
    payload = first_cache / payload_name
    payload.parent.mkdir(parents=True)
    payload.write_bytes(image_bytes("blue"))
    (first_cache / "manifest.json").write_text("{}")

    replacement = create("second-model", image)
    page = db.list_pages(replacement)[0]
    assert page["sha256"] == first_page["sha256"]
    manager._run(db.get_job(replacement))
    own_cache = manager.job_dir(replacement) / "private-cache" / "0001"
    assert (own_cache / payload_name).read_bytes() == payload.read_bytes()

    changed = create("changed-image", image + b"trailing bytes")
    manager._run(db.get_job(changed))
    assert not (manager.job_dir(changed) / "private-cache" / "0001").exists()
    other_ocr = create("other-ocr", image, {"ocr": "manga-ocr"})
    manager._run(db.get_job(other_ocr))
    assert not (manager.job_dir(other_ocr) / "private-cache" / "0001").exists()

    with zipfile.ZipFile(manager.archive(replacement)) as archive:
        assert archive.namelist() == ["0001.png"]
    manager.delete(first)
    assert (own_cache / payload_name).exists()
    manager.retry(replacement)
    assert own_cache.exists()  # A full retranslation drops output, not image preprocessing.
    manager.delete(replacement)
    assert not manager.job_dir(replacement).exists()
    db._conn.close()


def test_delete_keeps_job_visible_when_private_cache_cannot_be_removed(job, monkeypatch):
    db, _engine, manager, job_id = job
    cache = manager.job_dir(job_id) / "private-cache"
    cache.mkdir()

    def refuse_remove(_directory):
        raise PermissionError("read-only volume")

    monkeypatch.setattr("app.jobs.shutil.rmtree", refuse_remove)
    with pytest.raises(IngestError, match="삭제하지 못했습니다"):
        manager.delete(job_id)
    assert db.get_job(job_id) is not None
    assert cache.exists()


def test_transient_ocr_failure_is_not_kept_as_a_retranslation_checkpoint(tmp_path):
    source = tmp_path / "0001.png"
    source.write_bytes(image_bytes("white"))
    cache = tmp_path / "private-cache"
    checkpoint = _Page(cache, source, "same-preprocessing")
    try:
        text = checkpoint.stage("_perform_manga_ocr", lambda _images: ["[OCR FAILED]"], (["crop"],), {})
        checkpoint.publish()
        assert text == ["[OCR FAILED]"]  # Preserve upstream's error handling.
        assert not (cache / "0001" / "manifest.json").exists()
    finally:
        checkpoint.close()
