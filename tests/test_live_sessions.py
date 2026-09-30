import io

import pytest
from PIL import Image

from app.db import Database
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
            run.started_files.add(source.name)
            if self.during_page:
                self.during_page(source.name)
            if source.stem in self.fail:
                run.failed_files[source.name] = "boom"
                continue
            (run.output_dir / f"{source.stem}.png").write_bytes(source.read_bytes())


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


def test_cached_translation_is_not_reused_across_model_choices(live):
    db, _engine, manager, job_id = live
    manager.add_live_page(job_id, "a", image_bytes("red"))
    manager._run(db.get_job(job_id))
    other = manager.open_live(engine_id="watch", provider_id="p", options={}, instructions="", title="x", llm={"reasoning": "off"})
    assert manager.add_live_page(other, "a", image_bytes("red"))["translated"] is False
    assert manager.add_live_page(job_id, "a", image_bytes("red"))["translated"] is True


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
    manager._run(db.get_job(job_id))
    assert markers == [False, True]
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
