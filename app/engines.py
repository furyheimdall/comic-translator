"""Image-processing engines (detection → OCR → inpainting → Korean typesetting).

Each engine runs as a child process so its GPU memory is released as soon as
the job ends — the DGX Spark shares RAM between the LLM server and vision
models. Engines reach the LLM only through the internal OpenAI-compatible
proxy (`/llm/v1`), so every configured provider works with both engines.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .settings import Settings

PROGRESS_PREFIX = "@@CT@@"


@dataclass
class EngineRun:
    job_id: str
    input_dir: Path
    output_dir: Path
    log_path: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    instructions: str
    options: dict[str, Any]
    report: Callable[[float, str], None]
    page_errors: list[str] = field(default_factory=list)
    failed_files: dict[str, str] = field(default_factory=dict)
    started_files: set[str] = field(default_factory=set)
    # Live sessions keep the engine resident and feed pages into input_dir.
    watch_idle_seconds: int | None = None


@dataclass(frozen=True)
class EngineOption:
    key: str
    label: str
    type: str  # select | bool | int
    default: Any
    choices: tuple[tuple[str, str], ...] = ()
    help: str = ""
    minimum: int | None = None
    maximum: int | None = None

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "choices": [{"value": value, "label": label} for value, label in self.choices],
            "help": self.help,
            "min": self.minimum,
            "max": self.maximum,
        }


class EngineCancelled(RuntimeError):
    pass


class Engine:
    id: str
    name: str
    description: str
    options: tuple[EngineOption, ...]
    supports_live = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_environment: Callable[[str], dict[str, str]] = lambda _engine: {}
        self._process: subprocess.Popen | None = None
        self._cancelled = threading.Event()

    def availability(self) -> tuple[bool, str]:
        raise NotImplementedError

    def command(self, run: EngineRun) -> tuple[list[str], dict[str, str], Path]:
        raise NotImplementedError

    def resolve_options(self, supplied: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for option in self.options:
            value = supplied.get(option.key, option.default)
            if option.type == "bool":
                value = bool(value)
            elif option.type == "int":
                value = int(value)
                if (option.minimum is not None and value < option.minimum) or (
                    option.maximum is not None and value > option.maximum
                ):
                    raise ValueError(f"{self.name}: '{option.label}'은(는) {option.minimum}–{option.maximum} 범위여야 합니다.")
            elif option.type == "select" and value not in {choice for choice, _ in option.choices}:
                raise ValueError(f"{self.name}: '{option.label}' 값이 올바르지 않습니다: {value}")
            resolved[option.key] = value
        return resolved

    def public(self) -> dict[str, Any]:
        available, reason = self.availability()
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "available": available,
            "reason": reason,
            "live": self.supports_live,
            "options": [option.public() for option in self.options],
        }

    def cancel(self) -> None:
        self._cancelled.set()
        process = self._process
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    def run(self, run: EngineRun) -> None:
        self._cancelled.clear()
        argv, env, cwd = self.command(run)
        run.output_dir.mkdir(parents=True, exist_ok=True)
        with open(run.log_path, "a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(_redact(argv))}\n")
            log.flush()
            self._process = subprocess.Popen(
                argv,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if line.startswith(PROGRESS_PREFIX):
                    self._handle_event(run, line[len(PROGRESS_PREFIX) :])
                else:
                    log.write(line)
                    log.flush()
            code = self._process.wait()
        self._process = None
        if self._cancelled.is_set():
            raise EngineCancelled("작업이 취소되었습니다.")
        if code != 0:
            raise RuntimeError(f"{self.name} 엔진이 코드 {code}로 종료되었습니다. 작업 로그를 확인하세요.")

    @staticmethod
    def _handle_event(run: EngineRun, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        if "page_error" in event:
            name = str(event["page_error"])
            run.page_errors.append(f"{name}: {event.get('message', '')}")
            run.failed_files[name] = str(event.get("message", ""))
            return
        if "page" in event:
            run.started_files.add(str(event["page"]))
        run.report(float(event.get("progress", 0.0)), str(event.get("message", "")))


def _redact(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for arg in argv:
        redacted.append("***" if hide_next else arg)
        hide_next = arg in ("--openai-compatible-api-key", "--api-key", "--osb-hf-token")
    return redacted


# --------------------------------------------------------------------------- MangaTranslator


class MangaTranslatorEngine(Engine):
    id = "mangatranslator"
    name = "MangaTranslator"
    description = "Python · YOLO 말풍선 감지 + manga-ocr/PaddleOCR-VL + OpenCV/FLUX.2 Klein 지우기 + Skia 식자"
    options = (
        EngineOption(
            "ocr",
            "OCR",
            "select",
            "manga-ocr",
            (("manga-ocr", "manga-ocr (일본어 만화 전용)"), ("paddleocr-vl-1.6", "PaddleOCR-VL 1.6")),
        ),
        EngineOption(
            "osb",
            "말풍선 밖 텍스트 처리",
            "bool",
            True,
            help="말풍선 밖 텍스트(그림 위 대사·설명·효과음)를 감지해 지우고 번역합니다. Hugging Face 토큰과 deepghs/AnimeText_yolo 접근 권한이 필요합니다.",
        ),
        EngineOption(
            "inpainting",
            "말풍선 밖 지우기",
            "select",
            "flux_klein_4b",
            (("flux_klein_4b", "FLUX.2 Klein 4B (고품질)"), ("flux_klein_9b", "FLUX.2 Klein 9B (최고품질, 메모리 큼)"), ("opencv", "OpenCV (빠름)")),
        ),
        EngineOption(
            "context_pages", "이전 페이지 문맥 수", "int", 5, help="번역 시 함께 보내는 이전 페이지 대사 수", minimum=0, maximum=50
        ),
        EngineOption(
            "max_font_size",
            "최대 글자 크기(px)",
            "int",
            32,
            help="원고 해상도에 맞춰 조정하세요. 한국어는 가로쓰기라 일본어보다 크게 잡아야 읽기 좋습니다.",
            minimum=8,
            maximum=120,
        ),
    )

    def availability(self) -> tuple[bool, str]:
        if not self.settings.mt_python.exists():
            return False, "scripts/setup_mangatranslator.sh 를 먼저 실행하세요."
        if not self._font_pack():
            return False, f"한국어 폰트 팩이 없습니다: {self.settings.font_dir}"
        return True, ""

    def _font_pack(self) -> Path | None:
        if not self.settings.font_dir.is_dir():
            return None
        for pack in sorted(self.settings.font_dir.iterdir()):
            if pack.is_dir() and any(p.suffix.lower() in (".ttf", ".otf") for p in pack.iterdir()):
                return pack
        return None

    def command(self, run: EngineRun) -> tuple[list[str], dict[str, str], Path]:
        opts = run.options
        font_pack = self._font_pack()
        assert font_pack is not None
        args = [
            "--batch",
            "--input", str(run.input_dir),
            "--output", str(run.output_dir),
            "--input-language", "Japanese",
            "--output-language", "Korean",
            "--reading-direction", "rtl",
            "--font-dir", str(font_pack),
            "--provider", "OpenAI-Compatible",
            "--openai-compatible-url", run.llm_base_url,
            "--openai-compatible-api-key", run.llm_api_key,
            "--model-name", run.llm_model,
            "--translation-mode", "two-step",
            "--ocr-method", opts["ocr"],
            "--no-full-page-context",
            "--batch-previous-context-texts", str(opts["context_pages"]),
            "--max-font-size", str(opts["max_font_size"]),
            "--reasoning-effort", "none",
            "--max-tokens", "16384",
            "--no-auto-vertical-text",
            "--output-format", "png",
        ]
        env: dict[str, str] = self.model_environment(self.id)
        if opts["osb"]:
            args += [
                "--osb-enable",
                "--osb-font-dir", str(font_pack),
                "--osb-no-auto-vertical-text",
                "--osb-inpainting-method", opts["inpainting"],
                "--osb-flux-backend", "sdnq",
            ]
        argv = [str(self.settings.mt_python), str(Path(__file__).resolve().parent.parent / "engines" / "mt_runner.py"), str(self.settings.mt_dir), "--", *args]
        return argv, env, self.settings.mt_dir


# --------------------------------------------------------------------------- Koharu


class KoharuEngine(Engine):
    id = "koharu"
    name = "Koharu"
    supports_live = True
    description = "Rust · RF-DETR 레이아웃 + Comic Text Detector 글자 검출 + PaddleOCR-VL/manga-ocr + LaMa/FLUX.2 Klein + Vello 식자"
    options = (
        EngineOption(
            "ocr",
            "OCR",
            "select",
            "paddleocr-vl-1.6",
            (("paddleocr-vl-1.6", "PaddleOCR-VL 1.6"), ("manga-ocr", "manga-ocr"), ("baberu-ocr", "Baberu OCR"), ("hayai-ocr", "Hayai OCR")),
        ),
        EngineOption(
            "inpainting",
            "지우기 모델",
            "select",
            "lama",
            (("lama", "LaMa (만화 전용, 빠름)"), ("flux2-klein", "FLUX.2 Klein (고품질)"), ("rorem-mixed", "RORem"), ("aot-inpainting", "AOT-GAN")),
        ),
        EngineOption("vision", "페이지 이미지를 LLM에 전달", "bool", False, help="비전 지원 모델에서만 켜세요."),
    )

    def availability(self) -> tuple[bool, str]:
        if not self.settings.koharu_bin.exists():
            return False, "scripts/setup_koharu.sh 를 먼저 실행하세요."
        return True, ""

    def command(self, run: EngineRun) -> tuple[list[str], dict[str, str], Path]:
        opts = run.options
        argv = [
            str(self.settings.koharu_bin),
            "--input-dir", str(run.input_dir),
            "--output-dir", str(run.output_dir),
            "--base-url", run.llm_base_url,
            "--model", run.llm_model,
            "--ocr", opts["ocr"],
            "--inpainting", opts["inpainting"],
            "--target-language", "ko-KR",
            "--font-family", self.settings.koharu_font_family,
        ]
        if run.instructions.strip():
            argv += ["--instructions", run.instructions]
        if opts["vision"]:
            argv.append("--vision")
        if run.watch_idle_seconds is not None:
            argv += ["--watch", "--idle-timeout", str(run.watch_idle_seconds)]
        return argv, {"CT_LLM_API_KEY": run.llm_api_key, **self.model_environment(self.id)}, run.output_dir


def build_engines(settings: Settings) -> dict[str, Engine]:
    return {engine.id: engine for engine in (KoharuEngine(settings), MangaTranslatorEngine(settings))}
