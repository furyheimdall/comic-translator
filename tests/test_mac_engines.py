"""Platform support must not advertise CUDA-only installs as usable on macOS."""

from dataclasses import replace

from app import engines
from app.settings import load_settings


def test_mac_disables_cuda_engine_even_when_its_files_exist(tmp_path, monkeypatch):
    interpreter = tmp_path / "python"
    interpreter.touch()
    fonts = tmp_path / "fonts" / "Korean"
    fonts.mkdir(parents=True)
    (fonts / "font.otf").touch()
    settings = replace(load_settings(), mt_python=interpreter, font_dir=fonts.parent)
    monkeypatch.setattr(engines.platform, "system", lambda: "Linux")
    assert engines.MangaTranslatorEngine(settings).availability()[0]
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    available, reason = engines.MangaTranslatorEngine(settings).availability()
    assert not available and "Koharu" in reason


def test_mac_requires_native_arm64_for_koharu(tmp_path, monkeypatch):
    binary = tmp_path / "ct_batch"
    binary.touch()
    settings = replace(load_settings(), koharu_bin=binary)
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engines.platform, "machine", lambda: "x86_64")
    assert not engines.KoharuEngine(settings).availability()[0]
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    assert engines.KoharuEngine(settings).availability() == (True, "")
