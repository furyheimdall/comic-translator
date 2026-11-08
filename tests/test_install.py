"""Installer safety invariants that protect existing installs and arbitrary paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import install


def test_environment_roundtrip_handles_shell_and_systemd_metacharacters(tmp_path: Path) -> None:
    expected = {"CT_PASSWORD": ' spaced "quote" \\ $dollar `ticks` %percent #hash ;semicolon ',
                "CT_DATA_DIR": '/tmp/source tree/with % and "quotes"',
                "CT_HOST": "0.0.0.0", "CT_PORT": "18710"}
    env_file = tmp_path / "env"
    env_file.write_text("# maintained locally\n;\n  \n" + install.render_env(expected), encoding="utf-8")
    assert install.read_env(env_file) == expected


@pytest.mark.parametrize("bad", ["../../etc/passwd", "../prod", "a/b", "bad\nname", "#comment", "-flag"])
def test_rejects_unsafe_service_names(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        install.service_type(bad)


@pytest.mark.parametrize("bad", ["bad host", "example.com", "0.0.0.999", "::zz", "::1", "localhost", "192.168.1.10"])
def test_rejects_nonliteral_remote_hosts(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        install.host_type(bad)


@pytest.mark.parametrize("interpreter", ["3.10", "3.11.9", "not-an-interpreter"])
def test_rejects_unsupported_python_before_install(interpreter: str) -> None:
    with pytest.raises(install.InstallError):
        install.validate_python(interpreter)


def test_existing_config_is_refused_before_dependency_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config"
    config.mkdir()
    existing = config / "env"
    existing.write_text("DO_NOT_CHANGE=1\n")
    monkeypatch.setattr(install, "ROOT", tmp_path / "source")
    monkeypatch.setattr(install, "check_prerequisites", lambda *args: pytest.fail("checked prerequisites after collision"))
    args = SimpleNamespace(no_systemd=True, enable=False, start=False,
                           data_dir=tmp_path / "data", config_dir=config,
                           service_name="test", password_file=None, host="127.0.0.1",
                           port=18710, engine="none", python="3.13", non_interactive=True)
    with pytest.raises(install.InstallError, match="refusing to replace existing file"):
        install.install(args)
    assert existing.read_text() == "DO_NOT_CHANGE=1\n"


def test_existing_data_is_refused_before_dependency_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    existing = data / "app.db"
    existing.write_bytes(b"live data")
    monkeypatch.setattr(install, "ROOT", tmp_path / "source")
    monkeypatch.setattr(install, "check_prerequisites", lambda *args: pytest.fail("checked prerequisites after collision"))
    args = install.parse_args([
        "--no-systemd", "--engine", "none", "--non-interactive",
        "--data-dir", str(data), "--config-dir", str(tmp_path / "config"),
    ])
    with pytest.raises(install.InstallError, match="refusing nonempty data directory"):
        install.install(args)
    assert existing.read_bytes() == b"live data"


def test_external_noninteractive_bind_generates_private_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "source"
    for name in ("pyproject.toml", "uv.lock", "app/main.py", "web/index.html"):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    monkeypatch.setattr(install, "ROOT", root)
    monkeypatch.setattr(install, "check_prerequisites", lambda *args: None)
    monkeypatch.setattr(install.subprocess, "run", lambda *args, **kwargs: None)
    config = tmp_path / "config with spaces"
    args = install.parse_args([
        "--no-systemd", "--engine", "none", "--non-interactive", "--host", "0.0.0.0",
        "--data-dir", str(tmp_path / "data"), "--config-dir", str(config),
    ])
    install.install(args)
    secret = install.read_env(config / "env")["CT_PASSWORD"]
    assert "HF_HOME" not in install.read_env(config / "env")
    assert "CT_HF_TOKEN_FILE" not in install.read_env(config / "env")
    assert len(secret) >= 40
    assert secret not in capsys.readouterr().out
    assert (config / "env").stat().st_mode & 0o777 == 0o600
    assert config.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    ("report", "engine", "passes"),
    [
        ("8191, 24576, NVIDIA A100", "koharu", False),
        ("8192, 24576, NVIDIA A100", "koharu", True),
        ("15000, 24576, NVIDIA A100", "mangatranslator", False),
        ("8192, 8192, NVIDIA A100\n16384, 24576, NVIDIA H100", "both", True),
        ("8192, 8192, NVIDIA A100\n8192, 8192, NVIDIA A100", "both", False),
        ("[N/A], [N/A], NVIDIA A100", "koharu", False),
        ("[N/A], 24576, NVIDIA GB10", "koharu", False),
    ],
)
def test_dedicated_vram_uses_one_gpu_free_memory_not_aggregate(
    report: str, engine: str, passes: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install, "require_executable", lambda _: None)

    def nvidia(command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA GPU\n" if command[-1] == "-L" else report)

    monkeypatch.setattr(install.subprocess, "run", nvidia)
    if passes:
        install.preflight_gpu_memory(engine)
    else:
        with pytest.raises(install.InstallError):
            install.preflight_gpu_memory(engine)


@pytest.mark.parametrize(("available_kib", "engine", "passes"), [
    (18164640, "koharu", True),  # GB10 with a concurrently loaded LLM (~17.3 GiB free).
    (15 * 1024 * 1024 - 1, "koharu", True),  # Covers the observed ~14.9 GiB install rejection.
    (12 * 1024 * 1024, "koharu", True),
    (12 * 1024 * 1024 - 1, "koharu", False),
    (18164640, "mangatranslator", False),
    (24 * 1024 * 1024 - 1, "mangatranslator", False),
    (24 * 1024 * 1024, "mangatranslator", True),
])
def test_gb10_unified_memory_uses_available_ram_not_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    available_kib: int, engine: str, passes: bool,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal:       127535316 kB\nMemAvailable: {available_kib} kB\n")
    read_meminfo = install._meminfo
    monkeypatch.setattr(install, "_meminfo", lambda: read_meminfo(meminfo))
    monkeypatch.setattr(install, "require_executable", lambda _: None)

    def nvidia(command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA GB10\n" if command[-1] == "-L" else "[N/A], [N/A], NVIDIA GB10\n")

    monkeypatch.setattr(install.subprocess, "run", nvidia)
    if passes:
        install.preflight_gpu_memory(engine)
    else:
        with pytest.raises(install.InstallError):
            install.preflight_gpu_memory(engine)


def test_unified_gpu_without_memavailable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 127535316 kB\n")
    read_meminfo = install._meminfo
    monkeypatch.setattr(install, "_meminfo", lambda: read_meminfo(meminfo))
    monkeypatch.setattr(install, "require_executable", lambda _: None)

    def nvidia(command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA GB10\n" if command[-1] == "-L" else "[N/A], [N/A], NVIDIA GB10\n")

    monkeypatch.setattr(install.subprocess, "run", nvidia)
    with pytest.raises(install.InstallError):
        install.preflight_gpu_memory("koharu")


def test_gpu_preflight_rejects_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install, "require_executable", lambda _: None)
    monkeypatch.setattr(
        install.subprocess, "run",
        lambda command, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    with pytest.raises(install.InstallError, match="NVIDIA GPU"):
        install.preflight_gpu_memory("koharu")


@pytest.mark.parametrize("gpu_report", ["4096, 24576, NVIDIA A100", "[N/A], [N/A], NVIDIA A100"])
def test_insufficient_or_unknown_gpu_memory_aborts_before_uv_or_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu_report: str,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "linux")
    root = tmp_path / "source"
    for name in ("pyproject.toml", "uv.lock", "app/main.py", "web/index.html"):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    monkeypatch.setattr(install, "ROOT", root)
    monkeypatch.setattr(install, "require_executable", lambda _: None)

    def nvidia(command: list[str], **_kwargs: object) -> SimpleNamespace:
        assert command[0] == "nvidia-smi", "uv or engine setup ran before GPU preflight"
        return SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA A100\n" if command[-1] == "-L" else gpu_report)

    monkeypatch.setattr(install.subprocess, "run", nvidia)
    data, config = tmp_path / "data", tmp_path / "config"
    args = install.parse_args(["--no-systemd", "--engine", "koharu", "--data-dir", str(data), "--config-dir", str(config)])
    with pytest.raises(install.InstallError, match="GPU 사전 검사 실패"):
        install.install(args)
    assert not data.exists() and not config.exists() and not (root / ".venv").exists()


@pytest.mark.parametrize(("engine", "machine", "reason"), [
    ("mangatranslator", "arm64", "macOS supports Koharu only"),
    ("both", "arm64", "macOS supports Koharu only"),
    ("koharu", "x86_64", "native Apple Silicon"),
])
def test_mac_unsupported_engine_or_rosetta_fails_before_install_actions(
    engine: str, machine: str, reason: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    monkeypatch.setattr(install, "safe_path", lambda *_: pytest.fail("inspected installation paths"))
    monkeypatch.setattr(install.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("ran installer command"))
    with pytest.raises(install.InstallError, match=reason):
        install.install(install.parse_args(["--engine", engine]))


@pytest.mark.parametrize("flag", ["--enable", "--start"])
@pytest.mark.parametrize("no_systemd", [False, True])
def test_mac_rejects_service_flags_before_install_actions(
    flag: str, no_systemd: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install, "safe_path", lambda *_: pytest.fail("inspected installation paths"))
    monkeypatch.setattr(install.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("ran installer command"))
    argv = ["--engine", "none", flag] + (["--no-systemd"] if no_systemd else [])
    with pytest.raises(install.InstallError, match="unavailable on macOS"):
        install.install(install.parse_args(argv))


@pytest.mark.parametrize(("total", "passes"), [
    (8 * 1024 ** 3, False),
    (16 * 1024 ** 3 - 1, False),
    (16 * 1024 ** 3, True),
    (32 * 1024 ** 3, True),
])
def test_mac_memory_gates_total_not_available_or_compressed_memory(
    total: int, passes: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(install, "require_executable", lambda _: pytest.fail("required NVIDIA tooling"))

    def mac_memory(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command == ["sysctl", "-n", "hw.memsize"]:
            return SimpleNamespace(stdout=str(total))
        assert command == ["vm_stat"]
        return SimpleNamespace(stdout=(
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 32768.\nPages inactive: 32768.\n"
            "Pages speculative: 999999.\nPages purgeable: 999999.\n"
            "Pages occupied by compressor: 999999.\nSwapouts: 999999.\n"
        ))

    monkeypatch.setattr(install.subprocess, "run", mac_memory)
    assert install._mac_memory() == (1024 ** 3, total)
    if passes:
        install.preflight_gpu_memory("koharu")
    else:
        with pytest.raises(install.InstallError, match="16 GiB total"):
            install.preflight_gpu_memory("koharu")


@pytest.mark.parametrize("machine", ["arm64", "x86_64"])
def test_mac_web_only_skips_hardware_preflight(machine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    monkeypatch.setattr(install, "require_executable", lambda _: pytest.fail("required GPU tooling"))
    monkeypatch.setattr(install.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("measured GPU memory"))
    install.preflight_gpu_memory("none")


@pytest.mark.parametrize(("system", "engine", "service_flags"), [
    ("darwin", "koharu", []),
    ("darwin", "none", []),
    ("linux", "none", []),
    ("linux", "none", ["--enable", "--start"]),
])
def test_install_uses_manual_launcher_on_mac_and_user_service_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    system: str, engine: str, service_flags: list[str],
) -> None:
    root = tmp_path / "source"
    for name in ("pyproject.toml", "uv.lock", "app/main.py", "web/index.html"):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    monkeypatch.setattr(install, "ROOT", root)
    monkeypatch.setattr(install.sys, "platform", system)
    monkeypatch.setattr(install.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(install, "preflight_gpu_memory", lambda _: None)
    monkeypatch.setattr(install, "check_prerequisites", lambda *_args: None)
    monkeypatch.setattr(install.subprocess, "run", lambda *_args, **_kwargs: None)
    config = tmp_path / "config with spaces"
    install.install(install.parse_args([
        "--engine", engine, "--non-interactive",
        "--data-dir", str(tmp_path / "data"), "--config-dir", str(config),
        *service_flags,
    ]))
    use_systemd = system == "linux"
    assert (config / "run.py").is_file()
    assert (config / "run.py").stat().st_mode & 0o777 == 0o700
    unit = tmp_path / "xdg/systemd/user/comic-translator.service"
    assert unit.exists() == use_systemd
