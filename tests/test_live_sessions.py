import io
import json
import sqlite3

import pytest
from PIL import Image

from app.db import SCHEMA, Database
from app.engines import Engine
from app.jobs import JobManager


def image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, "PNG")
    return buffer.getvalue()


class WatchingEngine:
    """Mimics ct_batch --watch: scans its input folder until nothing new is left."""

    name = "watch"
    supports_live = True

    def __init__(self):
        self.during_page = None  # callback(file name) while a page is being processed
        self.after_last_scan = None  # callback() after the final empty scan, before exit
        self.fail = set()
        self.before_start = None  # callback before stdout reports the selected page
        self.processed = []

    def availability(self):
        return True, ""

    def resolve_options(self, options):
        return options

    def run(self, run):
        assert run.watch_idle_seconds is not None
        run.output_dir.mkdir(exist_ok=True)
        seen = set()
        while True:
            pending = sorted(p for p in run.input_dir.iterdir() if p.name not in seen and not p.name.startswith("."))
            if not pending:
                if self.after_last_scan:
                    self.after_last_scan()
                    self.after_last_scan = None
                return
            source = pending[0]
            seen.add(source.name)
            if self.before_start:
                self.before_start(source.name)
            Engine._handle_event(run, json.dumps({"page": source.name, "message": "processing"}))
            self.processed.append((run.job_id, source.name))
            if self.during_page:
                self.during_page(source.name)
            if source.stem in self.fail:
                Engine._handle_event(run, json.dumps({"page_error": source.name, "message": "boom"}))
                continue
            (run.output_dir / f"{source.stem}.png").write_bytes(source.read_bytes())
            Engine._handle_event(run, json.dumps({"progress": 1, "message": f"{source.name} 완료"}))


@pytest.fixture
def live(tmp_path):
    db = Database(tmp_path / "app.db")
    engine = WatchingEngine()
    manager = JobManager(db, tmp_path / "jobs", {"watch": engine}, lambda: ("http://unused", "secret"))
    job_id = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="live")
    yield db, engine, manager, job_id
    db._conn.close()


def pages(db, job_id):
    return {page["idx"]: page for page in db.list_pages(job_id)}


def test_same_settings_reuse_one_session_and_other_settings_do_not(live):
    db, _engine, manager, job_id = live
    assert manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x") == job_id
    assert manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="용어", title="x") != job_id
    fast = {"model": "fast", "reasoning": "off"}
    fast_id = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", llm=fast)
    assert fast_id != job_id
    assert manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", llm=dict(reversed(fast.items()))) == fast_id


def test_same_settings_stay_on_the_device_that_opened_them(live):
    db, _engine, manager, job_id = live
    owned = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", device_id="dev-a")
    assert owned != job_id
    assert db.get_job(owned)["device_id"] == "dev-a"
    assert manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="again", device_id="dev-a") == owned
    other = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", device_id="dev-b")
    assert other != owned and db.get_job(other)["device_id"] == "dev-b"


def test_cached_translation_is_not_reused_across_model_choices(live):
    db, _engine, manager, job_id = live
    manager.add_live_page(job_id, "a", image_bytes("red"))
    manager._run(db.get_job(job_id))
    other = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", llm={"reasoning": "off"})
    assert manager.add_live_page(other, "a", image_bytes("red"))["translated"] is False
    assert manager.add_live_page(job_id, "a", image_bytes("red"))["translated"] is True


def test_live_page_does_not_reuse_a_regular_jobs_output(live):
    db, _engine, manager, live_id = live
    original = image_bytes("red")
    regular = manager.create(
        title="regular", engine_id="watch", provider_id="p",
        options={}, instructions="", images=[("page.png", original)],
    )
    output = manager.job_dir(regular) / "output"
    output.mkdir()
    (output / "0001.png").write_bytes(image_bytes("blue"))
    db.set_page_output(regular, 1, "0001.png")

    result = manager.add_live_page(live_id, "same source", original)
    assert result["translated"] is False
    assert pages(db, live_id)[result["idx"]]["output"] is None


def test_new_work_for_another_session_asks_the_idle_engine_to_exit(live):
    db, engine, manager, job_id = live
    other = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", llm={"reasoning": "off"})
    manager.add_live_page(job_id, "a", image_bytes("red"))
    markers = []

    def queue_other(_name):
        watched = manager._live[0]
        manager.add_live_page(job_id, "same session", image_bytes("blue"))
        markers.append((watched / ".closed").exists())
        manager.add_live_page(other, "other session", image_bytes("green"))
        markers.append((watched / ".closed").exists())

    engine.during_page = lambda name: markers or queue_other(name)
    engine.after_last_scan = lambda: markers.append((manager._live[0] / ".closed").exists())
    manager._run(db.get_job(job_id))
    assert markers == [False, False, True]
    # Pages already queued for the watched session still finish before it exits.
    assert all(page["output"] for page in pages(db, job_id).values())
    assert db.get_job(other)["status"] == "queued"


def test_page_added_while_engine_watches_is_translated_in_the_same_run(live):
    db, engine, manager, job_id = live
    manager.add_live_page(job_id, "a", image_bytes("red"))
    added = []
    engine.during_page = lambda name: added or added.append(manager.add_live_page(job_id, "b", image_bytes("green")))
    manager._run(db.get_job(job_id))
    assert all(page["output"] for page in pages(db, job_id).values())
    assert db.get_job(job_id)["status"] == "done"


def test_page_added_after_last_scan_is_requeued_not_lost(live):
    db, engine, manager, job_id = live
    manager.add_live_page(job_id, "a", image_bytes("red"))
    engine.after_last_scan = lambda: manager.add_live_page(job_id, "late", image_bytes("green"))
    manager._run(db.get_job(job_id))
    assert pages(db, job_id)[2]["output"] is None and pages(db, job_id)[2]["error"] is None
    assert db.get_job(job_id)["status"] == "queued"
    manager._run(db.get_job(job_id))
    assert pages(db, job_id)[2]["output"]


def test_identical_image_is_served_from_cache_without_engine(live):
    db, _engine, manager, job_id = live
    manager.add_live_page(job_id, "a", image_bytes("red"))
    manager._run(db.get_job(job_id))
    result = manager.add_live_page(job_id, "again", image_bytes("red"))
    assert result["translated"] is True
    assert db.get_job(job_id)["status"] == "done"
    assert manager.output_path(job_id, pages(db, job_id)[result["idx"]]).exists()


def test_failed_and_discarded_pages_report_errors_instead_of_hanging(live):
    db, engine, manager, job_id = live
    engine.fail = {"0001"}
    manager.add_live_page(job_id, "bad", image_bytes("red"))
    manager.add_live_page(job_id, "dropped", image_bytes("green"))
    manager.discard_live_pages(job_id, {2})
    manager._run(db.get_job(job_id))
    assert pages(db, job_id)[1]["error"] == "boom"
    assert pages(db, job_id)[2]["error"] == "취소됨"
    assert db.get_job(job_id)["status"] == "done"
    # A new upload restarts the idle session.
    manager.add_live_page(job_id, "next", image_bytes("blue"))
    assert db.get_job(job_id)["status"] == "queued"


def test_restart_fails_waiting_live_pages(live, tmp_path):
    db, _engine, manager, job_id = live
    manager.add_live_page(job_id, "a", image_bytes("red"))
    db.reset_interrupted_jobs()
    assert pages(db, job_id)[1]["error"]
    assert db.get_job(job_id)["status"] == "done"


def priority_session(manager, **settings):
    return manager.open_live(
        engine_id="watch", provider_id="p", options={}, instructions="", title="single",
        priority="single", **settings,
    )


def test_priority_sessions_reuse_settings_but_not_normal_or_other_devices(live):
    db, _engine, manager, normal = live
    single = priority_session(manager)
    assert single != normal and priority_session(manager) == single
    assert db.get_job(normal)["priority"] == "normal"
    assert db.get_job(single)["priority"] == "single"
    owner = priority_session(manager, device_id="owner")
    assert owner != single and priority_session(manager, device_id="owner") == owner
    assert priority_session(manager, device_id="other") != owner
    assert priority_session(manager, llm={"model": "fast"}) != single


@pytest.mark.parametrize("fail_first", [False, True])
def test_priority_yields_after_current_page_before_pending_live_and_batch(live, fail_first):
    db, engine, manager, normal = live
    single = priority_session(manager)
    batch = manager.create(
        title="batch", engine_id="watch", provider_id="p", options={}, instructions="",
        images=[("batch", image_bytes("black"))],
    )
    manager.add_live_page(normal, "first", image_bytes("red"))
    manager.add_live_page(normal, "already pending", image_bytes("green"))
    if fail_first:
        engine.fail.add("0001")

    def interrupt(name):
        if len(engine.processed) != 1:
            return
        assert name == "0001.png"
        manager.add_live_page(single, "selected", image_bytes("blue"))
        manager.add_live_page(normal, "late normal", image_bytes("yellow"))
        watched = manager._live[0]
        assert (watched / name).exists()
        assert not (watched / "0002.png").exists()
        assert not (watched / "0003.png").exists()

    engine.during_page = interrupt
    manager._run(db.get_job(normal))
    assert engine.processed == [(normal, "0001.png")]
    first, second, third = pages(db, normal).values()
    assert bool(first["output"]) is not fail_first
    assert bool(first["error"]) is fail_first
    assert second["output"] is None and second["error"] is None
    assert third["output"] is None and third["error"] is None
    assert db.get_job(normal)["status"] == "queued"
    assert db.next_queued_job()["id"] == single
    engine.fail.clear()
    manager._run(db.next_queued_job())
    assert pages(db, single)[1]["output"]
    assert db.next_queued_job()["id"] == normal
    manager._run(db.next_queued_job())
    assert pages(db, normal)[2]["output"] and pages(db, normal)[3]["output"]
    assert engine.processed.count((normal, "0001.png")) == 1
    assert db.next_queued_job()["id"] == batch


def test_priority_page_fifo_across_reused_sessions(live):
    db, engine, manager, normal = live
    older = priority_session(manager)
    newer = priority_session(manager, device_id="other")
    manager.add_live_page(normal, "normal", image_bytes("black"))
    # A newer session's earlier image must beat an older session's later image.
    manager.add_live_page(newer, "one", image_bytes("red"))
    manager.add_live_page(older, "two", image_bytes("green"))
    manager.add_live_page(newer, "three", image_bytes("blue"))
    assert db.next_queued_job()["id"] == newer
    manager._run(db.next_queued_job())
    assert engine.processed == [(newer, "0001.png")]
    assert db.next_queued_job()["id"] == older
    manager._run(db.next_queued_job())
    assert db.next_queued_job()["id"] == newer
    manager._run(db.next_queued_job())
    assert engine.processed == [(newer, "0001.png"), (older, "0001.png"), (newer, "0002.png")]
    assert db.next_queued_job()["id"] == normal


def test_priority_arrival_before_start_event_never_unlinks_selected_page(live):
    db, engine, manager, normal = live
    single = priority_session(manager)
    manager.add_live_page(normal, "first", image_bytes("red"))
    manager.add_live_page(normal, "second", image_bytes("green"))

    def selected(name):
        assert name not in manager._live[1].started_files
        manager.add_live_page(single, "selected", image_bytes("blue"))
        manager.discard_live_pages(normal, {1})
        assert (manager._live[0] / name).exists()

    engine.before_start = selected
    manager._run(db.get_job(normal))
    assert pages(db, normal)[1]["output"]
    assert pages(db, normal)[1]["error"] is None
    assert pages(db, normal)[2]["output"] is None
    assert pages(db, normal)[2]["error"] is None
    assert db.next_queued_job()["id"] == single


def test_error_boundary_admits_next_page_without_waiting_for_idle_timeout(live):
    db, engine, manager, normal = live
    manager.add_live_page(normal, "bad", image_bytes("red"))
    manager.add_live_page(normal, "good", image_bytes("green"))
    engine.fail.add("0001")
    manager._run(db.get_job(normal))
    assert pages(db, normal)[1]["error"] == "boom"
    assert pages(db, normal)[2]["output"]


def test_priority_queued_before_live_admission_prevents_normal_page_start(live):
    db, engine, manager, normal = live
    manager.add_live_page(normal, "normal", image_bytes("red"))
    single = priority_session(manager)
    manager.add_live_page(single, "selected", image_bytes("blue"))
    # Worker selected normal just before the request arrived.
    manager._run(db.get_job(normal))
    assert not engine.processed
    assert pages(db, normal)[1]["error"] is None
    assert db.get_job(normal)["status"] == "queued"
    assert db.next_queued_job()["id"] == single


def test_priority_and_fifo_order_persist_when_database_reopens(live, tmp_path):
    db, _engine, manager, normal = live
    first = priority_session(manager)
    second = priority_session(manager, device_id="other")
    manager.add_live_page(normal, "normal", image_bytes("black"))
    manager.add_live_page(second, "first", image_bytes("red"))
    manager.add_live_page(first, "second", image_bytes("green"))
    reopened = Database(tmp_path / "app.db")
    try:
        assert reopened.get_job(normal)["priority"] == "normal"
        assert reopened.get_job(first)["priority"] == "single"
        assert reopened.next_queued_job()["id"] == second
        # Server restart retains classification and the existing live-page error policy.
        reopened.reset_interrupted_jobs()
        assert reopened.get_job(first)["priority"] == "single"
        assert pages(reopened, first)[1]["error"]
    finally:
        reopened._conn.close()


def test_priority_migration_preserves_legacy_device_ownership(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.execute("ALTER TABLE jobs ADD COLUMN device_id TEXT")
    connection.execute(
        """INSERT INTO jobs (id, title, engine, provider_id, status, created_at, device_id)
           VALUES ('legacy', 'old', 'watch', 'p', 'done', 1, 'owner')"""
    )
    connection.commit()
    connection.close()
    db = Database(path)
    try:
        row = db.get_job("legacy")
        assert row["priority"] == "normal"
        assert row["device_id"] == "owner"
    finally:
        db._conn.close()


def test_late_normal_upload_after_yield_is_requeued_without_admission(live):
    db, engine, manager, normal = live
    single = priority_session(manager)
    manager.add_live_page(normal, "first", image_bytes("red"))
    engine.during_page = lambda _name: manager.add_live_page(single, "selected", image_bytes("blue"))

    def after_exit_scan():
        assert manager._live_closing
        manager.add_live_page(normal, "late", image_bytes("green"))
        assert not (manager._live[0] / "0002.png").exists()

    engine.after_last_scan = after_exit_scan
    manager._run(db.get_job(normal))
    assert pages(db, normal)[1]["output"]
    assert pages(db, normal)[2]["output"] is None and pages(db, normal)[2]["error"] is None
    assert db.get_job(normal)["status"] == "queued"
    assert db.next_queued_job()["id"] == single


def test_success_stage_progress_does_not_admit_next_page_early(live):
    db, engine, manager, normal = live
    manager.add_live_page(normal, "first", image_bytes("red"))
    manager.add_live_page(normal, "second", image_bytes("green"))

    def report_stage(name):
        if name != "0001.png":
            return
        watched, run = manager._live
        Engine._handle_event(run, json.dumps({"progress": 1, "message": "OCR 완료"}))
        assert not (watched / "0002.png").exists()
        assert manager._live_page == name

    engine.during_page = report_stage
    manager._run(db.get_job(normal))
    assert all(page["output"] for page in pages(db, normal).values())


def test_priority_retry_releases_idle_live_watcher(live):
    db, engine, manager, normal = live
    single = priority_session(manager)
    manager.add_live_page(single, "bad", image_bytes("blue"))
    engine.fail.add("0001")
    manager._run(db.get_job(single))
    engine.fail.clear()
    manager.add_live_page(normal, "normal", image_bytes("red"))
    markers = []

    def retry_while_idle():
        manager.retry(single, failed_only=True)
        markers.append((manager._live[0] / ".closed").exists())

    engine.after_last_scan = retry_while_idle
    manager._run(db.get_job(normal))
    assert markers == [True]
    assert db.next_queued_job()["id"] == single
    assert pages(db, single)[1]["error"] is None


def test_upload_after_finish_before_current_clears_is_queued(live, monkeypatch):
    db, _engine, manager, normal = live
    manager.add_live_page(normal, "first", image_bytes("red"))
    finish = manager._finish_live

    def finish_then_upload(job_id, run):
        finish(job_id, run)
        manager.add_live_page(job_id, "after finish", image_bytes("green"))

    monkeypatch.setattr(manager, "_finish_live", finish_then_upload)
    manager._run(db.get_job(normal))
    assert db.get_job(normal)["status"] == "queued"
    assert pages(db, normal)[2]["error"] is None
