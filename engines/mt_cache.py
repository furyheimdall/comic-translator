"""Private checkpoints for the backend's MangaTranslator two-step/local-OCR path.

The upstream renderer still runs normally. Only preprocessing function results are
replayed, including masks, colors, original OSB pixels, geometry and OCR strings.
Snapshots are taken immediately, before later pipeline stages mutate metadata.
One-step/LLM OCR and overlapping inpaint are deliberately not intercepted: they
have no independent local-OCR boundary. The backend does not enable those modes.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from contextvars import ContextVar
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile


_SCHEMA = 1
_TRANSLATION_INPUTS = (
    "input_language", "reading_direction", "translation_mode", "ocr_method",
    "upscale_method", "whiteout_conjoined_bubbles", "bubble_min_side_pixels",
    "osb_min_side_pixels", "send_full_page_context", "context_image_max_side_pixels",
)


def _sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _config_value(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _config_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in {"request_coordinator", "huggingface_token"}
        }
    if isinstance(value, dict):
        return {key: _config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_value(item) for item in value]
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    return str(value)


def _fingerprint(config, source_hash, image_path, output_path):
    # Include all non-translation settings, conservatively including rendering.
    # API credentials, provider/model and translation prompts cannot affect these
    # local stages. The source hash invalidates changed vendor implementations.
    settings = {
        field.name: _config_value(getattr(config, field.name))
        for field in dataclasses.fields(config)
        if field.name not in {"translation", "request_coordinator", "verbose"}
    }
    settings["translation"] = {
        key: getattr(config.translation, key) for key in _TRANSLATION_INPUTS
    }
    settings["source"] = source_hash
    settings["output_suffix"] = Path(output_path or image_path).suffix.lower()
    return hashlib.sha256(_json(settings).encode()).hexdigest()


class _Payload:
    """JSON plus typed binary members; never deserialize executable pickle."""

    def __init__(self, archive):
        self.archive = archive
        self.counter = 0

    def encode(self, value):
        import numpy as np
        from PIL import Image

        if value is None or isinstance(value, (bool, str, int, float)):
            return value
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, (np.ndarray, Image.Image)):
            member = f"data/{self.counter}"
            self.counter += 1
            buffer = io.BytesIO()
            if isinstance(value, np.ndarray):
                np.save(buffer, value, allow_pickle=False)
                kind = "array"
            else:
                value.save(buffer, format="PNG")
                kind = "image"
            self.archive.writestr(member, buffer.getvalue())
            return {"type": kind, "member": member}
        if isinstance(value, (list, tuple)):
            return {"type": "tuple" if isinstance(value, tuple) else "list",
                    "items": [self.encode(item) for item in value]}
        if isinstance(value, dict):
            return {"type": "dict", "items": [
                [self.encode(key), self.encode(item)] for key, item in value.items()
            ]}
        raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")

    def decode(self, value):
        import numpy as np
        from PIL import Image

        if not isinstance(value, dict):
            if value is None or isinstance(value, (bool, str, int, float)):
                return value
            raise ValueError("Invalid checkpoint value")
        kind = value["type"]
        if kind in {"array", "image"}:
            buffer = io.BytesIO(self.archive.read(value["member"]))
            if kind == "array":
                return np.load(buffer, allow_pickle=False)
            with Image.open(buffer) as image:
                image.load()
                return image.copy()
        if kind in {"list", "tuple"}:
            items = [self.decode(item) for item in value["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "dict":
            return {self.decode(key): self.decode(item) for key, item in value["items"]}
        raise ValueError("Invalid checkpoint type")


class _Page:
    def __init__(self, root, image_path, fingerprint):
        self.entry = root / Path(image_path).stem
        self.manifest = {
            "schema": _SCHEMA,
            "engine": "mangatranslator",
            "input_sha256": _sha256(image_path),
            "fingerprint": fingerprint,
        }
        self.records = {}
        self.counts = {}
        self.loaded = None
        self.temp = None
        self.archive = None
        self.disabled = False
        self.published = False
        self.osb_detection_empty = None
        try:
            manifest = json.loads((self.entry / "manifest.json").read_text())
            if any(manifest.get(key) != value for key, value in self.manifest.items()):
                raise ValueError("Checkpoint inputs changed")
            payload = self.entry / "payload.zip"
            if _sha256(payload) != manifest["payload_sha256"]:
                raise ValueError("Checkpoint payload changed")
            with ZipFile(payload) as archive:
                codec = _Payload(archive)
                records = json.loads(archive.read("index.json"))
                self.loaded = {key: codec.decode(value) for key, value in records.items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile):
            self.loaded = None
        if self.loaded is None:
            # A copied stale/corrupt checkpoint is not a published entry for
            # these inputs. Discard it so this cold run can repair the cache.
            if self.entry.exists():
                shutil.rmtree(self.entry)
            root.mkdir(parents=True, exist_ok=True)
            self.temp = Path(tempfile.mkdtemp(prefix=".mt-", dir=root))
            self.archive = ZipFile(self.temp / "payload.zip", "w", compression=ZIP_DEFLATED)
            self.codec = _Payload(self.archive)

    def stage(self, name, function, args, kwargs):
        count = self.counts.get(name, 0)
        self.counts[name] = count + 1
        key = f"{name}:{count}"
        if self.loaded is not None:
            # The complete payload is validated/deserialized before any replay.
            # Missing stages signal an incompatible snapshot, not a partial hit.
            if key not in self.loaded:
                raise _InvalidCheckpoint(key)
            return self.loaded[key]
        try:
            result = function(*args, **kwargs)
        except Exception:
            # Upstream may deliberately recover a stage failure. Such a page
            # must not publish an incomplete successful-looking checkpoint.
            self.disabled = True
            raise
        if name in {"_perform_manga_ocr", "_perform_paddle_ocr_vl"} and (
            not isinstance(result, (list, tuple))
            or len(result) != len(args[0])
            or any(text == "[OCR FAILED]" for text in result)
        ):
            # The upstream OCR deliberately converts model errors to a normal
            # return value. Never freeze those errors across retries.
            self.disabled = True
        if not self.disabled and not self.published:
            try:
                self.records[key] = self.codec.encode(result)
            except (OSError, ValueError, TypeError):
                self.disabled = True
        return result

    def publish(self):
        if self.loaded is not None or self.disabled or self.published:
            return
        try:
            self.archive.writestr("index.json", _json(self.records))
            self.archive.close()
            self.manifest["payload_sha256"] = _sha256(self.temp / "payload.zip")
            (self.temp / "manifest.json").write_text(_json(self.manifest))
            # A published page is immutable. Never replace another complete entry.
            if not self.entry.exists():
                os.rename(self.temp, self.entry)
                self.temp = None
            self.published = True
        except OSError:
            self.disabled = True

    def close(self):
        if self.archive is not None:
            self.archive.close()
        if self.temp is not None:
            shutil.rmtree(self.temp, ignore_errors=True)


class _InvalidCheckpoint(BaseException):
    # Upstream catches Exception around detection, cleaning and translation.
    # This private control signal must reach translate_page for a full restart.
    pass


def install(pipeline, translation, cache_dir, vendor_dir):
    """Install job-local interception; no effects unless explicitly requested."""
    root = Path(cache_dir).resolve()
    source = hashlib.sha256()
    for directory in (Path(vendor_dir) / "core", Path(vendor_dir) / "utils"):
        for path in sorted(directory.rglob("*.py")):
            source.update(str(path.relative_to(vendor_dir)).encode())
            source.update(bytes.fromhex(_sha256(path)))
    source.update(bytes.fromhex(_sha256(__file__)))
    source_hash = source.hexdigest()
    current = ContextVar("mt_private_checkpoint", default=None)
    import core.outside_text_processor as outside_text

    def flag_on_error(module, name):
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except Exception:
                page = current.get()
                if page is not None:
                    page.disabled = True
                raise

        setattr(module, name, wrapper)

    # Upstream recovers these failures as ordinary fallback images. Mark the
    # page before those catch blocks consume the exception.
    flag_on_error(pipeline, "clean_speech_bubbles")
    flag_on_error(outside_text, "FluxKleinInpainter")

    # Both bubble verification and outside-text detection recover failed YOLO
    # loads/inference into ordinary boxes. Those fallbacks must not become
    # permanent preprocessing checkpoints.
    from core.ml.model_manager import ModelManager

    original_load_osb = ModelManager.load_yolo_osbtext

    @functools.wraps(original_load_osb)
    def observe_osb_model(*args, **kwargs):
        page = current.get()
        try:
            model = original_load_osb(*args, **kwargs)
        except Exception:
            if page is not None:
                page.disabled = True
            raise
        if page is None:
            return model

        @functools.wraps(model)
        def infer(*model_args, **model_kwargs):
            try:
                return model(*model_args, **model_kwargs)
            except Exception:
                page.disabled = True
                raise

        return infer

    ModelManager.load_yolo_osbtext = observe_osb_model

    original_detect = outside_text.OutsideTextDetector.detect_outside_text

    @functools.wraps(original_detect)
    def observe_detection(*args, **kwargs):
        page = current.get()
        try:
            result = original_detect(*args, **kwargs)
        except Exception:
            if page is not None:
                page.disabled = True
            raise
        if page is not None:
            page.osb_detection_empty = not bool(result)
        return result

    outside_text.OutsideTextDetector.detect_outside_text = observe_detection
    original_prepare = outside_text.prepare_outside_text_work

    @functools.wraps(original_prepare)
    def observe_prepare(*args, **kwargs):
        page = current.get()
        if page is not None:
            page.osb_detection_empty = None
        result = original_prepare(*args, **kwargs)
        config = args[1] if len(args) > 1 else kwargs["config"]
        if page is not None and config.outside_text.enabled and result is None and page.osb_detection_empty is not True:
            # None also means no detections. Only accept that case when the
            # detector completed successfully and found nothing.
            page.disabled = True
        return result

    outside_text.prepare_outside_text_work = observe_prepare
    original_finish = outside_text.finish_outside_text_work

    @functools.wraps(original_finish)
    def observe_finish(*args, **kwargs):
        result = original_finish(*args, **kwargs)
        page = current.get()
        work = args[0] if args else kwargs["work"]
        if page is not None and result[0] is work.pil_image:
            # The recovered error path returns the original object; a
            # legitimate no-op is conservatively uncached as well.
            page.disabled = True
        return result

    outside_text.finish_outside_text_work = observe_finish

    def intercept(module, name, *, publish=False):
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            page = current.get()
            if page is None:
                return original(*args, **kwargs)
            result = page.stage(name, original, args, kwargs)
            if publish:
                # Local OCR has finished and the upstream function has not yet
                # issued its translation request. The erased pixels and all
                # render metadata are already in the snapshot at this point.
                page.publish()
            return result

        setattr(module, name, wrapper)

    for name in (
        "_apply_pre_upscale_if_needed", "detect_speech_bubbles", "detect_panels",
        "process_outside_text", "_clean_speech_bubbles_for_page",
        "prepare_bubble_images_for_translation",
    ):
        intercept(pipeline, name)
    for name in ("_perform_manga_ocr", "_perform_paddle_ocr_vl"):
        intercept(translation, name, publish=True)

    original_page = pipeline.translate_and_render
    signature = inspect.signature(original_page)

    @functools.wraps(original_page)
    def translate_page(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        config = bound.arguments["config"]
        if (
            config.translation.translation_mode != "two-step"
            or config.translation.ocr_method not in {"manga-ocr", "paddleocr-vl-1.6"}
            or config.overlap_llm_with_inpaint
            or config.test_mode
            or config.cleaning_only
            or config.upscaling_only
        ):
            return original_page(*args, **kwargs)
        # Upstream validation can clamp preprocessing settings. Fingerprint the
        # resolved values, not the unvalidated CLI configuration.
        pipeline.validate_config(config)
        image_path = bound.arguments["image_path"]
        fingerprint = _fingerprint(
            config, source_hash, image_path, bound.arguments.get("output_path")
        )
        try:
            page = _Page(root, image_path, fingerprint)
        except (OSError, ValueError, TypeError):
            return original_page(*args, **kwargs)
        token = current.set(page)
        try:
            try:
                result = original_page(*args, **kwargs)
            except _InvalidCheckpoint:
                # This happens before translation for a structurally incompatible
                # entry. Restart all preprocessing; never mix partial snapshots.
                current.reset(token)
                token = current.set(None)
                try:
                    shutil.rmtree(page.entry)
                except OSError:
                    return original_page(*args, **kwargs)
                return translate_page(*args, **kwargs)
            # Pages without translatable regions have no OCR boundary and no
            # translation request. Their complete empty-page path is cacheable.
            if not page.counts.get("prepare_bubble_images_for_translation"):
                page.publish()
            return result
        finally:
            current.reset(token)
            page.close()

    pipeline.translate_and_render = translate_page
