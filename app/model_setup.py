"""Explicit model preparation for the files consumed by Koharu and MangaTranslator.

The Koharu catalog mirrors pinned files in koharu-ml 0.83.5; its Rust
HuggingFaceFile::path is NOT the huggingface_hub cache layout.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database
from .secretbox import SecretBox
from .settings import ROOT, Settings


@dataclass(frozen=True)
class Artifact:
    repo: str
    revision: str
    filename: str
    # MangaTranslator stores individual weights under ./models, not HF_HOME.
    mt_target: str = ""


@dataclass(frozen=True)
class Profile:
    id: str
    engine: str
    label: str
    description: str
    files: tuple[Artifact, ...]
    default: bool = False


def files(repo: str, revision: str, *names: str) -> tuple[Artifact, ...]:
    return tuple(Artifact(repo, revision, name) for name in names)


DETECTION = (
    *files("mayocream/koharu-layout-rfdetr-seg-2xl-1152", "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2", "inference_config.json", "model.safetensors"),
    *files("mayocream/comic-text-detector", "15ade029f4dabd502bc97af6051c8b9f2bec24d5", "yolo-v5.safetensors", "unet.safetensors", "dbnet.safetensors"),
)
PADDLE = files("PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "511b09642bb324401f15f97cc23bc67e8f0a291d", "PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf")
LAMA = files("mayocream/lama-manga", "f91c85b26913b3e83f9877867b4c336da3675238", "lama-manga.safetensors")
BASE = DETECTION + PADDLE + LAMA


PROFILES = (
    Profile("koharu-default", "koharu", "Koharu 기본 · 레이아웃 + PaddleOCR-VL 1.6 + LaMa", "기본 OCR/지우기 조합에 필요한 고정 모델 파일입니다. GPU 실행 가능 여부는 별도로 확인해야 합니다.", BASE, True),
    Profile("koharu-ocr-manga-ocr", "koharu", "Koharu 추가 OCR · manga-ocr", "기본 구성과 함께 사용하는 추가 OCR입니다.", BASE + files("mayocream/manga-ocr", "4380edba990b959c508752350955350c1c80c31c", "config.json", "model.safetensors", "preprocessor_config.json", "vocab.txt")),
    Profile("koharu-ocr-baberu-ocr", "koharu", "Koharu 추가 OCR · Baberu", "기본 구성과 함께 사용하는 추가 OCR입니다.", BASE + (*files("genshiai-daichi/baberu-ocr", "d9cc13153e9a1cd8fdfa3b7b1cc329da2020aeae", "config.json", "generation_config.json", "model.safetensors", "tokenizer/vocab.json"), *files("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415", "config.json", "preprocessor_config.json"))),
    Profile("koharu-ocr-hayai-ocr", "koharu", "Koharu 추가 OCR · Hayai", "기본 구성과 함께 사용하는 추가 OCR입니다.", BASE + files("JustANormalTinkerer/hayai-ocr-v2", "4a4ce477c9a8841f208b94e1d9ed5c0938965e05", "config.json", "model.safetensors", "tokenizer.json")),
    Profile("koharu-inpainting-aot-inpainting", "koharu", "Koharu 추가 지우기 · AOT", "기본 구성과 함께 사용하는 추가 지우기 모델입니다.", BASE + files("mayocream/aot-inpainting", "cffe2346ac2b5ebe1f2d61335d602d12cc144c6f", "model.safetensors")),
    Profile("koharu-inpainting-flux2-klein", "koharu", "Koharu 추가 지우기 · FLUX.2 Klein", "용량과 VRAM 사용량이 큰 선택 모델입니다.", BASE + (*files("unsloth/FLUX.2-klein-4B-GGUF", "0084d1df98e2e2137fe776d55170bc4792ec1d66", "flux-2-klein-4b-Q4_K_M.gguf"), *files("black-forest-labs/FLUX.2-small-decoder", "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a", "full_encoder_small_decoder.safetensors"), *files("unsloth/Qwen3-4B-GGUF", "22c9fc8a8c7700b76a1789366280a6a5a1ad1120", "Qwen3-4B-Q4_K_M.gguf"))),
    Profile("koharu-inpainting-rorem-mixed", "koharu", "Koharu 추가 지우기 · RORem", "라이선스/접근 제한이 있는 저장소는 Hugging Face에서 사용 조건에 동의하고 토큰을 등록해야 합니다.", BASE + (*files("mayocream/RORem-mixed-GGUF", "62c75b3e6f078a19e2698b0f677e8a4aa4c9ea56", "rorem-mixed-unet-q4_K.gguf", "sdxl-version-marker.safetensors"), *files("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "115134f363124c53c7d878647567d04daf26e41e", "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors"))),
    Profile("mt-bubble-detector", "mangatranslator", "MangaTranslator · YOLO 말풍선 감지", "알려진 단일 파일만 준비합니다. MT 전체 모델 준비 또는 오프라인 실행을 뜻하지 않습니다.", (Artifact("kitsumed/yolov8m_seg-speech-bubble", "main", "model.pt", "yolo/yolov8m_seg-speech-bubble.pt"),)),
    Profile("mt-osb-detector", "mangatranslator", "MangaTranslator · 말풍선 밖 텍스트 감지", "OSB 사용 시 필요한 알려진 파일입니다. 다른 OCR/FLUX 모델은 실행 시 별도로 다운로드될 수 있습니다.", (Artifact("deepghs/AnimeText_yolo", "main", "yolo12x_animetext/model.pt", "yolo/animetext_yolov12x.pt"),)),
)
PROFILE_BY_ID = {profile.id: profile for profile in PROFILES}


def default_cache() -> Path:
    """Keep the existing Rust dirs::cache_dir()/koharu/packages location."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    return (Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".cache").resolve()


def artifact_path(artifact: Artifact, cache: Path, settings: Settings) -> Path:
    if artifact.mt_target:
        return settings.mt_dir / "models" / artifact.mt_target
    return cache / "koharu" / "packages" / "hugging-face" / "models" / artifact.repo.replace("/", "--") / "snapshots" / artifact.revision / artifact.filename


def _safe_directory(path: Path, settings: Settings) -> None:
    # Never let a user-selected cache point inside production data or the checkout.
    if path == Path("/") or path.is_relative_to(settings.data_dir.resolve()) or path.is_relative_to(ROOT):
        raise ValueError("캐시 위치는 앱 소스 및 작업 데이터와 분리된 디렉터리여야 합니다.")
    # Existing symlinked cache descendants could redirect downloads into private data.
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError("심볼릭 링크 캐시 경로는 사용할 수 없습니다.")


def _ready(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        marker = path.with_name(path.name + ".ct-complete")
        # Files downloaded by the existing engine predate our manifest; Rust
        # considers those usable if present. Our own files have an exact size.
        if marker.is_file():
            return path.stat().st_size == json.loads(marker.read_text())["size"]
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("보안 연결이 아닌 다운로드 주소가 반환되었습니다.")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


_OPENER = urllib.request.build_opener(_SafeRedirect())


class ModelSetup:
    def __init__(self, db: Database, box: SecretBox, settings: Settings, engines: dict[str, Any]) -> None:
        self.db, self.box, self.settings, self.engines = db, box, settings, engines
        self._lock = threading.RLock()
        self._status: dict[str, Any] = {"state": "idle", "message": "다운로드 준비"}
        self._thread: threading.Thread | None = None

    def _row(self) -> dict[str, Any]:
        return self.db.get_model_setup() or {"cache_dir": None, "token": None}

    def cache_dir(self) -> Path:
        return Path(self._row()["cache_dir"] or default_cache())

    def token(self) -> str | None:
        sealed = self._row()["token"]
        if sealed is not None:
            return self.box.open(sealed)["token"]
        if os.environ.get("HF_TOKEN"):
            return os.environ["HF_TOKEN"]
        path = self.settings.hf_token_file
        return path.read_text().strip() if path.is_file() else None

    def environment(self, engine: str) -> dict[str, str]:
        cache = self.cache_dir()
        # Until a path is explicitly saved, preserve a legacy HF_HOME used by
        # an existing MangaTranslator installation. Rust uses XDG_CACHE_HOME.
        legacy_hf_home = os.environ.get("HF_HOME") if self._row()["cache_dir"] is None else None
        env = {"XDG_CACHE_HOME": str(cache), "HF_HOME": legacy_hf_home or str(cache / "huggingface")}
        token = self.token()
        if token:
            env["HF_TOKEN"] = token
        else:
            env["HF_TOKEN"] = ""
            if self._row()["token"] is not None:
                # Explicit clear also masks a token cached by huggingface_hub
                # or inherited through its older environment variable.
                env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_TOKEN_PATH="/dev/null",
                           HUGGING_FACE_HUB_TOKEN="")
        return env

    def public(self) -> dict[str, Any]:
        with self._lock:
            row = self._row()
            cache = self.cache_dir()
            effective_token = self.token()
            source = ("saved" if row["token"] is not None else
                      "env" if os.environ.get("HF_TOKEN") else "file") if effective_token else None
            readiness: dict[Path, bool] = {}
            installed = {engine: instance.availability()[0] for engine, instance in self.engines.items()}
            profiles = []
            for profile in PROFILES:
                models: dict[tuple[str, str], dict[str, Any]] = {}
                missing = []
                for artifact in profile.files:
                    key = artifact.repo, artifact.revision
                    models.setdefault(key, {"repo": artifact.repo, "revision": artifact.revision, "files": [], "url": "https://huggingface.co/" + artifact.repo})["files"].append(artifact.filename)
                    path = artifact_path(artifact, cache, self.settings)
                    if path not in readiness:
                        readiness[path] = _ready(path)
                    if not readiness[path]:
                        missing.append(f"{artifact.repo}/{artifact.filename}")
                profiles.append({"id": profile.id, "engine": profile.engine, "label": profile.label, "description": profile.description, "default": profile.default, "supplemental": profile.engine == "mangatranslator", "engine_installed": installed[profile.engine], "models": list(models.values()), "prepared": not missing, "missing": missing})
            legacy_hf_home = os.environ.get("HF_HOME") if row["cache_dir"] is None else None
            return {
                "cache_dir": str(cache),
                "koharu_cache_dir": str(cache / "koharu" / "packages"),
                "hf_cache_dir": legacy_hf_home or str(cache / "huggingface"),
                "token_configured": bool(source),
                "token_source": source,
                "status": self._status.copy(),
                "profiles": profiles,
            }

    def require_prepared_for_new_job(self, options: dict[str, Any]) -> None:
        # Legacy installations with existing Koharu jobs retain their old
        # on-demand behavior until a user explicitly saves model settings.
        if self.db.get_model_setup() is None and any(
            job["engine"] == "koharu" for job in self.db.list_jobs()
        ):
            return
        cache = self.cache_dir()
        selected = [
            PROFILE_BY_ID["koharu-default"],
            *(
                [PROFILE_BY_ID[f"koharu-ocr-{options['ocr']}"]]
                if options["ocr"] != "paddleocr-vl-1.6" else []
            ),
            *(
                [PROFILE_BY_ID[f"koharu-inpainting-{options['inpainting']}"]]
                if options["inpainting"] != "lama" else []
            ),
        ]
        for profile in selected:
            if any(not _ready(artifact_path(file, cache, self.settings)) for file in profile.files):
                raise RuntimeError(f"모델 준비에서 '{profile.label}'을(를) 먼저 다운로드하세요.")

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] == "running":
                raise RuntimeError("다운로드 진행 중에는 설정을 변경할 수 없습니다.")
            if not isinstance(payload, dict) or any(key not in {"cache_dir", "token", "clear_token"} for key in payload):
                raise ValueError("설정 항목이 올바르지 않습니다.")
            row = self._row()
            cache = row["cache_dir"]
            if "cache_dir" in payload:
                raw = payload["cache_dir"]
                if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
                    raise ValueError("캐시 경로를 입력하세요.")
                candidate = Path(raw).expanduser()
                if not candidate.is_absolute():
                    raise ValueError("캐시 경로는 절대 경로여야 합니다.")
                _safe_directory(candidate, self.settings)
                cache = str(candidate.resolve())
                _safe_directory(Path(cache), self.settings)
            token = row["token"]
            if payload.get("clear_token") is True:
                token = self.box.seal({"token": ""})
            elif payload.get("clear_token") not in (None, False):
                raise ValueError("토큰 삭제 값이 올바르지 않습니다.")
            if "token" in payload:
                raw_token = payload["token"]
                if not isinstance(raw_token, str):
                    raise ValueError("토큰 값이 올바르지 않습니다.")
                if raw_token:
                    if raw_token != raw_token.strip() or any(c.isspace() or ord(c) < 33 for c in raw_token):
                        raise ValueError("토큰에 공백이나 제어 문자를 넣을 수 없습니다.")
                    token = self.box.seal({"token": raw_token})
            self.db.save_model_setup(cache, token)
            return self.public()

    def prepare(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] == "running":
                raise RuntimeError("이미 다른 모델 다운로드가 진행 중입니다.")
            if profile_id not in PROFILE_BY_ID:
                raise ValueError("모델 준비 항목을 찾을 수 없습니다.")
            profile = PROFILE_BY_ID[profile_id]
            cache, token = self.cache_dir(), self.token()
            _safe_directory(cache, self.settings)
            self._status = {"state": "running", "message": "다운로드 준비", "profile_id": profile.id, "current": "", "done": 0, "total": len(profile.files), "started_at": time.time()}
            self._thread = threading.Thread(target=self._worker, args=(profile, cache, token), daemon=True, name="model-setup")
            self._thread.start()
            return self._status.copy()

    def _worker(self, profile: Profile, cache: Path, token: str | None) -> None:
        try:
            for index, artifact in enumerate(profile.files):
                target = artifact_path(artifact, cache, self.settings)
                with self._lock:
                    self._status.update(current=f"{artifact.repo}/{artifact.filename}", message="다운로드 진행", done=index)
                if not _ready(target):
                    self._download(artifact, target, token)
                with self._lock:
                    self._status["done"] = index + 1
            with self._lock:
                self._status.update(state="done", message="다운로드 완료", current="", finished_at=time.time())
        except (Exception) as exc:
            # Never send raw HTTP exception strings/URLs/headers (or tokens) to the UI/log.
            if isinstance(exc, urllib.error.HTTPError):
                reason = f"HTTP {exc.code} (접근 제한 저장소라면 Hugging Face에서 사용 조건에 동의하고 토큰을 등록하세요)"
            elif isinstance(exc, urllib.error.URLError):
                reason = "네트워크 연결을 확인하세요."
            elif isinstance(exc, OSError):
                reason = "캐시 저장 경로 및 여유 공간을 확인하세요."
            elif isinstance(exc, ValueError):
                reason = "다운로드 응답이 올바르지 않습니다."
            else:
                reason = "모델 다운로드에 실패했습니다."
            with self._lock:
                self._status.update(state="failed", message=f"다운로드 실패: {reason}", finished_at=time.time())

    def _download(self, artifact: Artifact, target: Path, token: str | None) -> None:
        # No token in the URL or the log. Redirects discard Authorization.
        url = "https://huggingface.co/" + artifact.repo + "/resolve/" + urllib.parse.quote(artifact.revision, safe="/") + "/" + urllib.parse.quote(artifact.filename, safe="/")
        headers = {"User-Agent": "comic-translator/model-setup"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(url, headers=headers)
        # Ensure existing descendants cannot redirect writes into other trees.
        parent = target.parent
        while parent != parent.parent:
            if parent.is_symlink():
                raise ValueError("캐시 경로에 심볼릭 링크가 있습니다.")
            parent = parent.parent
        if target.is_symlink():
            raise ValueError("모델 파일 경로에 심볼릭 링크가 있습니다.")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = None
        marker_tmp = None
        try:
            with _OPENER.open(request, timeout=60) as response:
                if response.headers.get("Content-Type", "").lower().startswith("text/html"):
                    raise ValueError("HTML 대신 모델 파일이 필요합니다.")
                expected = response.headers.get("Content-Length")
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ct-download-", delete=False) as stage:
                    tmp = Path(stage.name)
                    os.fchmod(stage.fileno(), stat.S_IRUSR | stat.S_IWUSR)
                    length = 0
                    while chunk := response.read(1024 * 1024):
                        stage.write(chunk)
                        length += len(chunk)
                        with self._lock:
                            self._status["message"] = f"다운로드 진행 · {length / (1024 * 1024):.1f} MiB"
                    stage.flush()
                    os.fsync(stage.fileno())
                if not length or (expected is not None and length != int(expected)):
                    raise ValueError("모델 파일 전송이 완료되지 않았습니다.")
            os.replace(tmp, target)
            tmp = None
            # The stable target survives restart; our size marker distinguishes
            # complete app downloads from truncated transfers.
            marker = target.with_name(target.name + ".ct-complete")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, prefix=".ct-marker-", delete=False) as stage:
                marker_tmp = Path(stage.name)
                os.fchmod(stage.fileno(), 0o600)
                json.dump({"size": length}, stage)
            os.replace(marker_tmp, marker)
            marker_tmp = None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            if marker_tmp is not None:
                marker_tmp.unlink(missing_ok=True)
