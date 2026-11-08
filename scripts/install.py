#!/usr/bin/env python3
"""Install this source checkout as an opt-in user service (or manual launcher)."""

from __future__ import annotations

import argparse
import csv
import getpass
import ipaddress
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
from pathlib import Path


# Minimum viability for the default pipelines, not a guarantee for alternate
# models or simultaneous local LLM inference. MT's default FLUX.2 Klein 4B
# alone needs ~8 GiB bf16 weights plus encoders, OCR and inference activations.
# Koharu defaults to RF-DETR, PaddleOCR-VL 1.6 and LaMa. A completed GB10
# job coincided with ~10 GiB of available-RAM swing; leave room to attempt it.
DEDICATED_MIN_GIB = {"koharu": 8, "mangatranslator": 16, "both": 16}
UNIFIED_MIN_GIB = {"koharu": 12, "mangatranslator": 24, "both": 24}
# Mac policy is deliberately separate: total unified memory is a planning
# budget, not an empirically established requirement or a free-memory cutoff.
MAC_KOHARU_MIN_GIB = 16
UNIFIED_GPU_RE = re.compile(r"\b(?:GB10|GH200)\b", re.IGNORECASE)
GIB = 1024 ** 3

ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z", re.ASCII)
KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)


class InstallError(Exception):
    pass


def safe_path(value: str | Path) -> Path:
    raw = str(value)
    if not raw or any(ord(c) < 32 or ord(c) == 127 for c in raw):
        raise InstallError("paths must not contain control characters")
    path = Path(raw).expanduser().resolve()
    if any(ord(c) < 32 or ord(c) == 127 for c in str(path)):
        raise InstallError("resolved paths must not contain control characters")
    return path


def host_type(value: str) -> str:
    if value not in ("127.0.0.1", "0.0.0.0"):
        raise argparse.ArgumentTypeError("use 127.0.0.1 locally or 0.0.0.0 for LAN; engines require IPv4 loopback for their proxy")
    return value


def port_type(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be 1..65535") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be 1..65535")
    return port


def service_type(value: str) -> str:
    if not NAME_RE.fullmatch(value) or value in {".", ".."} or ".." in value:
        raise argparse.ArgumentTypeError("service name must contain only letters, numbers, dots, hyphens and underscores")
    return value


def validate_python(request: str) -> None:
    version = re.fullmatch(r"3\.(\d+)(?:\.\d+)?", request)
    if version:
        if int(version.group(1)) < 12:
            raise InstallError("web Python must be >=3.12")
        return
    executable = safe_path(request) if "/" in request else shutil.which(request)
    if not executable or not Path(executable).is_file():
        raise InstallError("--python must be a 3.12+ version or an existing interpreter")
    version_info = subprocess.run(
        [str(executable), "-c", "import sys; print(*sys.version_info[:2])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        major, minor = map(int, version_info.split())
    except ValueError as exc:
        raise InstallError(f"invalid Python interpreter: {request}") from exc
    if (major, minor) < (3, 12):
        raise InstallError("web Python must be >=3.12")

def env_quote(value: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise InstallError("environment values cannot contain NUL or newlines")
    # systemd.exec(5) EnvironmentFile double quotes: only these four escapes
    # are interpreted. This is not shell syntax and must never be shell-sourced.
    return '"' + ''.join('\\' + ch if ch in '\\"$`' else ch for ch in value) + '"'


def render_env(values: dict[str, str]) -> str:
    for key in values:
        if not KEY_RE.fullmatch(key):
            raise InstallError(f"invalid environment key: {key}")
    return ''.join(f"{key}={env_quote(value)}\n" for key, value in values.items())


def read_env(path: Path) -> dict[str, str]:
    """Parse only our single-line quoted EnvironmentFile format; never execute it."""
    result = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    for line in lines:
        if not line.strip() or line.startswith(("#", ";")):
            continue
        key, sep, quoted = line.partition("=")
        if not sep or not KEY_RE.fullmatch(key) or len(quoted) < 2 or quoted[0] != '"' or quoted[-1] != '"':
            raise InstallError(f"invalid environment assignment in {path}")
        inner = quoted[1:-1]
        value = []
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == '\\' and i + 1 < len(inner) and inner[i + 1] in '\\"$`':
                i += 1
                ch = inner[i]
            elif ch == '"':
                raise InstallError(f"invalid environment quoting in {path}")
            value.append(ch)
            i += 1
        if key in result:
            raise InstallError(f"duplicate environment key in {path}: {key}")
        result[key] = ''.join(value)
    if not {"CT_DATA_DIR", "CT_HOST", "CT_PORT"}.issubset(result):
        raise InstallError(f"installer EnvironmentFile is incomplete: {path}")
    return result


def unit_arg(path: Path) -> str:
    # ExecStart= is a command line, but systemd rejects a quoted executable
    # containing literal quotes. /usr/bin/env execs our source interpreter as
    # its first argument, supporting arbitrary printable checkout paths.
    text = str(path).replace("%", "%%").replace("$", "$$")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_unit(root: Path, env_file: Path) -> str:
    return ("[Unit]\nDescription=Comic Translator\nAfter=network-online.target\n"
            "Wants=network-online.target\n\n[Service]\nType=simple\n"
            # Use the same launcher/parser as manual execution; systemd path
            # directives do not share ExecStart's quoting rules.
            f"ExecStart=/usr/bin/env {unit_arg(root / '.venv/bin/python')} {unit_arg(env_file.parent / 'run.py')}\n"
            "Restart=on-failure\n\n[Install]\nWantedBy=default.target\n")


def read_secret(path: Path, label: str) -> str:
    if not path.is_file():
        raise InstallError(f"{label} file is missing or not a regular file: {path}")
    if path.stat().st_size > 4096:
        raise InstallError(f"{label} file is too large: {path}")
    value = path.read_text(encoding="utf-8").removesuffix("\n")
    if not value or any(ch in value for ch in "\n\r\x00"):
        raise InstallError(f"{label} file must contain one nonempty line")
    return value


def require_executable(name: str) -> None:
    if not shutil.which(name):
        raise InstallError(f"missing required command: {name}")


def _memory_mib(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdecimal() else None


def _meminfo(path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InstallError(f"cannot measure unified GPU memory from {path}: {exc}") from exc
    for line in lines:
        match = re.fullmatch(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
        if match:
            values[match[1]] = int(match[2]) * 1024
    if not all(values.get(key, 0) > 0 for key in ("MemTotal", "MemAvailable")):
        raise InstallError("cannot measure unified GPU memory: /proc/meminfo lacks MemTotal or MemAvailable")
    if values["MemAvailable"] > values["MemTotal"]:
        raise InstallError("cannot measure unified GPU memory: MemAvailable exceeds MemTotal")
    return values["MemAvailable"], values["MemTotal"]


def check_platform(engine: str) -> None:
    if sys.platform != "darwin" or engine == "none":
        return
    if engine != "koharu":
        raise InstallError("macOS supports Koharu only; use --engine koharu or --engine none. "
                           "MangaTranslator requires the Linux/NVIDIA installation.")
    if platform.machine() != "arm64":
        raise InstallError("macOS Koharu requires native Apple Silicon (arm64). "
                           "Use a native Terminal/Python/Rust toolchain, not Rosetta; "
                           "use --engine none for the web app only.")


def _mac_memory() -> tuple[int, int]:
    total_result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
    pages_result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    total_text = total_result.stdout.strip()
    page_size = re.search(r"page size of (\d+) bytes", pages_result.stdout)
    counts = dict(re.findall(r"^(Pages free|Pages inactive):\s+(\d+)\.\s*$",
                             pages_result.stdout, re.MULTILINE))
    if not total_text.isdecimal() or not page_size or not all(
        key in counts for key in ("Pages free", "Pages inactive")
    ):
        raise InstallError("cannot measure Mac unified memory: invalid sysctl hw.memsize or vm_stat output")
    total, size = int(total_text), int(page_size[1])
    available = (int(counts["Pages free"]) + int(counts["Pages inactive"])) * size
    if total <= 0 or size <= 0 or available > total:
        raise InstallError("cannot measure Mac unified memory: inconsistent sysctl/vm_stat values")
    return available, total


def preflight_mac_memory() -> None:
    available, total = _mac_memory()
    print(f"Mac unified memory: total {total / GIB:.1f} GiB; conservative free + inactive "
          f"estimate {available / GIB:.1f} GiB (sysctl hw.memsize, vm_stat page size/counts; "
          "compressed memory, swap and other reclaimable pages are not counted).")
    print(f"Koharu Mac planning policy: at least {MAC_KOHARU_MIN_GIB} GiB total unified memory, "
          "not an empirically established Mac requirement. Available memory is advisory only; "
          "this does not guarantee inference or concurrent LLM capacity.")
    if total < MAC_KOHARU_MIN_GIB * GIB:
        raise InstallError(f"Koharu Mac planning policy requires at least {MAC_KOHARU_MIN_GIB} GiB total "
                           "unified memory; use --engine none for the web app only.")
    print("Memory advisory: close competing applications before inference; model loading and "
          "runtime memory pressure still determine whether a job can complete.")


def preflight_gpu_memory(engine: str) -> None:
    check_platform(engine)
    if engine == "none":
        print("GPU 사전 검사 생략: --engine none (웹 앱만 설치, GPU 엔진 미설치).")
        return
    if sys.platform == "darwin":
        preflight_mac_memory()
        return
    require_executable("nvidia-smi")
    gpu_list = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if gpu_list.returncode or not gpu_list.stdout.strip():
        raise InstallError("GPU 사전 검사 실패: nvidia-smi -L에서 NVIDIA GPU를 확인할 수 없습니다")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total,name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode or not result.stdout.strip():
        raise InstallError("GPU 사전 검사 실패: nvidia-smi에서 GPU 메모리를 측정할 수 없습니다")
    rows = list(csv.reader(result.stdout.splitlines()))
    if not rows or any(len(row) != 3 for row in rows):
        raise InstallError("GPU 사전 검사 실패: nvidia-smi GPU 메모리 출력 형식을 알 수 없습니다")
    viable = False
    for row in rows:
        free_text, total_text, name_text = (field.strip() for field in row)
        name = name_text or "이름 없음"
        free_mib, total_mib = _memory_mib(free_text), _memory_mib(total_text)
        if free_mib is not None and total_mib is not None and 0 < total_mib and 0 <= free_mib <= total_mib:
            free, total = free_mib * 1024 ** 2, total_mib * 1024 ** 2
            required = DEDICATED_MIN_GIB[engine]
            print(f"GPU 사전 검사 ({name}): 전용 VRAM 여유 {free / GIB:.1f} GiB / 전체 {total / GIB:.1f} GiB; "
                  f"설치 전 최소 여유 기준 {required} GiB (nvidia-smi memory.free/total, GPU별 검사).")
            viable |= free >= required * GIB and total >= required * GIB
        elif UNIFIED_GPU_RE.search(name) and free_mib is None and total_mib is None:
            required = UNIFIED_MIN_GIB[engine]
            try:
                free, total = _meminfo()
            except InstallError as exc:
                print(f"GPU 사전 검사 ({name}): 설치 전 최소 여유 기준 {required} GiB, "
                      f"통합 메모리 측정 실패 ({exc}).")
                continue
            print(f"GPU 사전 검사 ({name}): 통합 메모리 여유 {free / GIB:.1f} GiB / 전체 {total / GIB:.1f} GiB; "
                  f"설치 전 최소 여유 기준 {required} GiB (nvidia-smi VRAM N/A, /proc/meminfo "
                  "MemAvailable/MemTotal; CPU와 GPU가 공유하며 현재 다른 작업 사용량 반영).")
            viable |= free >= required * GIB and total >= required * GIB
        else:
            print(f"GPU 사전 검사 ({name}): VRAM 여유/전체를 신뢰성 있게 측정할 수 없음; "
                  f"전용 VRAM 설치 전 최소 여유 기준 {DEDICATED_MIN_GIB[engine]} GiB "
                  "(nvidia-smi memory.free/total; 알려진 통합 GPU가 아니거나 메모리 수치가 유효하지 않음).")
    print("위 기준은 기본 모델 파이프라인 설치 전 최소 여유 확인이며, 다른 모델 옵션이나 "
          "동시 실행 LLM의 추론 가능성을 보장하지 않습니다.")
    if not viable:
        raise InstallError("GPU 사전 검사 실패: 한 GPU에서도 선택한 엔진의 최소 여유 메모리가 확인되지 않습니다; "
                           "다른 GPU 작업을 종료해 여유를 확보하거나 --engine none으로 웹 앱만 설치하세요")


def check_prerequisites(engine: str, use_systemd: bool) -> None:
    require_executable("uv")
    if use_systemd:
        require_executable("systemctl")
        subprocess.run(["systemctl", "--user", "show-environment"], check=True, stdout=subprocess.DEVNULL)
    if engine in {"koharu", "mangatranslator", "both"}:
        for command in ("bash", "git"):
            require_executable(command)
        if engine in {"koharu", "both"} and not (ROOT / "engines/koharu/quality.patch").is_file():
            raise InstallError("Koharu source patch is missing")
    if engine in {"koharu", "both"}:
        require_executable("rustup")
        require_executable("python3")
        if sys.platform == "darwin":
            for command in ("xcrun", "pkg-config"):
                require_executable(command)
            try:
                clang = Path(subprocess.run(
                    ["xcrun", "--find", "clang"], capture_output=True, text=True, check=True,
                ).stdout.strip())
                sdk = Path(subprocess.run(
                    ["xcrun", "--sdk", "macosx", "--show-sdk-path"], capture_output=True, text=True, check=True,
                ).stdout.strip())
            except subprocess.CalledProcessError as exc:
                raise InstallError("Koharu requires selected Xcode Command Line Tools "
                                   "(xcode-select --install; check xcode-select -p)") from exc
            if not clang.is_file() or not sdk.is_dir():
                raise InstallError("Koharu requires Xcode Command Line Tools (xcode-select --install)")
            clang_dir = Path(os.environ.get("LIBCLANG_PATH") or clang.parent.parent / "lib")
            if not clang_dir.is_dir() or not list(clang_dir.glob("libclang*.dylib")):
                raise InstallError("Koharu requires Xcode libclang.dylib (or LIBCLANG_PATH)")
        else:
            for command in ("gcc", "ldconfig"):
                require_executable(command)
            clang_dir = Path(os.environ.get("LIBCLANG_PATH") or max(
                (str(folder) for folder in Path("/usr/lib").glob("llvm-*/lib") if list(folder.glob("libclang.so*"))),
                default="",
            ))
            if not clang_dir.is_dir() or not list(clang_dir.glob("libclang.so*")):
                raise InstallError("Koharu requires libclang (or LIBCLANG_PATH)")
            fontconfig = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True)
            if "libfontconfig.so.1 " not in fontconfig.stdout:
                raise InstallError("Koharu requires libfontconfig.so.1")
            gcc_include = subprocess.run(["gcc", "-print-file-name=include/stddef.h"], capture_output=True, text=True, check=True).stdout.strip()
            if not Path(gcc_include).is_file():
                raise InstallError("Koharu requires GCC's stddef.h")
        if not (ROOT / "engines/koharu/ct_batch.rs").is_file():
            raise InstallError("Koharu batch driver is missing")
    if engine in {"mangatranslator", "both"}:
        require_executable("curl")
        if not (ROOT / "scripts/setup_mangatranslator.sh").is_file():
            raise InstallError("MangaTranslator setup script is missing")


def ensure_private_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise InstallError(f"not a directory: {path}")
        return
    path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)


def create_new(path: Path, content: str, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def launcher_source(env_file: Path) -> str:
    return ("#!/usr/bin/env python3\n"
            "# Generated by scripts/install.py; do not shell-source the EnvironmentFile.\n"
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from scripts.install import launch\n"
            f"launch({str(env_file)!r})\n")


def launch(env_file: str) -> None:
    values = read_env(safe_path(env_file))
    os.chdir(ROOT)
    os.execve(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), "-m", "app.main"], {**os.environ, **values})


def access_urls(host: str, port: int) -> list[str]:
    if host == "127.0.0.1":
        return [f"http://127.0.0.1:{port}"]
    addresses = {"127.0.0.1"}
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
            if family == socket.AF_INET:
                ip = ipaddress.ip_address(sockaddr[0])
                if not ip.is_loopback and not ip.is_link_local:
                    addresses.add(str(ip))
    except OSError:
        pass
    return [f"http://{address}:{port}" for address in sorted(addresses)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=host_type, default="127.0.0.1")
    parser.add_argument("--port", type=port_type, default=18710)
    parser.add_argument("--data-dir", default="~/.local/share/comic-translator")
    parser.add_argument("--config-dir", default="~/.config/comic-translator")
    parser.add_argument("--service-name", type=service_type, default="comic-translator")
    parser.add_argument("--engine", choices=("koharu", "mangatranslator", "both", "none"), default="koharu")
    parser.add_argument("--python", default="3.13", help="web Python interpreter or uv Python version (>=3.12)")
    parser.add_argument("--password-file", type=Path, help="read web login password from a file, never from argv")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--no-systemd", action="store_true", help="create only a manual launcher (default on macOS)")
    parser.add_argument("--enable", action="store_true", help="explicitly enable the user service")
    parser.add_argument("--start", action="store_true", help="explicitly start the user service")
    return parser.parse_args(argv)


def install(args: argparse.Namespace) -> None:
    check_platform(args.engine)
    if sys.platform == "darwin" and (args.enable or args.start):
        raise InstallError("--enable and --start are unavailable on macOS; use the generated manual launcher (no systemd/launchd service).")
    use_systemd = sys.platform != "darwin" and not args.no_systemd
    if not use_systemd and (args.enable or args.start):
        raise InstallError("--enable and --start require systemd (omit --no-systemd)")
    data = safe_path(args.data_dir)
    config = safe_path(args.config_dir)
    env_file = config / "env"
    launcher = config / "run.py"
    xdg_config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    unit = safe_path(Path(xdg_config) / "systemd/user" / f"{args.service_name}.service")
    targets = [env_file, launcher] + ([unit] if use_systemd else [])
    for path in targets:
        if path.exists() or path.is_symlink():
            raise InstallError(f"refusing to replace existing file: {path}")
    for directory in (data, config):
        if directory.exists() and not directory.is_dir():
            raise InstallError(f"not a directory: {directory}")
    if data.exists() and any(data.iterdir()):
        raise InstallError(f"refusing nonempty data directory: {data}; use a fresh path and migrate only after stopping any existing service")
    if use_systemd and unit.parent.exists() and not unit.parent.is_dir():
        raise InstallError(f"not a directory: {unit.parent}")
    if (ROOT / ".venv").exists() or (ROOT / ".venv").is_symlink():
        raise InstallError(f"refusing to modify an existing source environment: {ROOT / '.venv'}")
    for item in ("pyproject.toml", "uv.lock", "app/main.py", "web/index.html"):
        if not (ROOT / item).is_file():
            raise InstallError(f"source checkout incomplete: {item}")
    validate_python(args.python)
    preflight_gpu_memory(args.engine)
    check_prerequisites(args.engine, use_systemd)
    password = read_secret(safe_path(args.password_file), "password") if args.password_file else None
    is_local = args.host == "127.0.0.1"
    generated_password = False
    if not is_local and not password:
        if args.non_interactive:
            password = secrets.token_urlsafe(32)
            generated_password = True
        else:
            if not sys.stdin.isatty():
                raise InstallError("external bind without a TTY requires --non-interactive or --password-file")
            password = getpass.getpass("Web login password (blank generates a secure random password): ")
            if not password:
                password = secrets.token_urlsafe(32)
                generated_password = True
    # Model cache and token are configured in the web UI after installation.
    values = {
        "CT_DATA_DIR": str(data), "CT_HOST": args.host, "CT_PORT": str(args.port),
        "CT_MT_DIR": str(ROOT / "engines/mangatranslator"),
        "CT_MT_PYTHON": str(ROOT / "engines/mangatranslator/.venv/bin/python"),
        "CT_KOHARU_BIN": str(ROOT / "engines/koharu-target/release/ct_batch"),
        "CT_FONT_DIR": str(ROOT / "fonts/korean"),
    }
    if password:
        values["CT_PASSWORD"] = password
    environment = render_env(values)
    service = render_unit(ROOT, env_file)
    # Do not touch the current service, existing data, or config until all
    # foreseeable validation has passed. Dependencies must install successfully
    # before new service configuration becomes visible.
    uv_env = os.environ.copy()
    uv_env.pop("UV_PROJECT_ENVIRONMENT", None)
    uv_env.pop("VIRTUAL_ENV", None)
    subprocess.run(["uv", "sync", "--locked", "--no-dev", "--python", args.python], cwd=ROOT, env=uv_env, check=True)
    for engine, script in (("koharu", "setup_koharu.sh"), ("mangatranslator", "setup_mangatranslator.sh")):
        if args.engine in (engine, "both"):
            subprocess.run(["bash", str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    for directory in (data, config):
        ensure_private_dir(directory)
    if use_systemd:
        unit.parent.mkdir(parents=True, exist_ok=True)
    create_new(env_file, environment, 0o600)
    create_new(launcher, launcher_source(env_file), 0o700)
    if use_systemd:
        create_new(unit, service, 0o644)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        if args.enable:
            subprocess.run(["systemctl", "--user", "enable", f"{args.service_name}.service"], check=True)
        if args.start:
            subprocess.run(["systemctl", "--user", "start", f"{args.service_name}.service"], check=True)
    print(f"Installed source checkout: {ROOT}")
    print(f"EnvironmentFile (private): {env_file}")
    if generated_password:
        print("A login password was generated and stored only in the private EnvironmentFile; retrieve it locally before logging in.")
    if not use_systemd:
        print(f"Manual start (not started): {launcher}")
    else:
        print(f"User service: {args.service_name}.service (enable={args.enable}, start={args.start})")
        print(f"Manual start (not started): {launcher}")
    print("Access URL(s), once started:")
    for url in access_urls(args.host, args.port):
        print(f"  {url}")
    if args.host == "0.0.0.0":
        print("For additional LAN/Tailscale interfaces, substitute their actual address for the loopback address.")
    print("Configure API providers, the model cache path, and Hugging Face token in the web UI; prepare models separately.")


def main(argv: list[str] | None = None) -> int:
    try:
        install(parse_args(argv))
    except (InstallError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
