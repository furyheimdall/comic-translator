#!/usr/bin/env python3
"""Build an allowlisted source archive; never package service state or engine clones."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parent.parent
REQUIRED = frozenset({
    ".env.example", ".gitignore", "LICENSE", "README.md", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "uv.lock",
    "app/__init__.py", "app/main.py",
    "web/index.html", "web/app.js", "web/style.css",
    "scripts/install.py", "scripts/package_release.py",
    "scripts/setup_koharu.sh", "scripts/setup_mangatranslator.sh",
    "engines/mt_runner.py", "engines/koharu/ct_batch.rs", "engines/koharu/quality.patch",
    "tests/fixtures/make_sample_pages.py",
})
EXACT = REQUIRED
SOURCE_DIRS = {
    "app": frozenset({".py"}),
    "web": frozenset({".html", ".js", ".css"}),
    "scripts": frozenset({".py", ".sh"}),
    "tests": frozenset({".py"}),
    "tests/fixtures": frozenset({".py"}),
    ".github/workflows": frozenset({".yml", ".yaml"}),
}
# Restrict source directories to one file level. No recursive traversal into data,
# downloaded engines, fonts, caches, build artifacts or virtual environments.
SENSITIVE_NAME = re.compile(
    r"(?:^|[._-])(?:secret|secrets|token|tokens|credential|credentials|password|passwd|api[_-]?keys?|private[_-]?keys?)(?:$|[._-])"
    r"|^id_(?:rsa|ed25519|ecdsa)$|(?:^|[._-])(?:env|pem|p12|pfx)(?:$|[._-])"
    r"|(?:\.(?:key|db|sqlite|sqlite3|log|token))$",
    re.IGNORECASE,
)
PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9 ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----")
CREDENTIAL = re.compile(
    rb"(?:AKIA|ASIA)[A-Z0-9]{16}\b|AIza[0-9A-Za-z_-]{35}\b"
    rb"|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
    rb"|sk-(?:proj-|ant-(?:api\d+-)?|live_)?[A-Za-z0-9_-]{24,}"
    rb"|hf_[A-Za-z0-9]{34,}|xai-[A-Za-z0-9_-]{24,}"
    rb"|glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{16,})\b"
)
# A source annotation or variable/function reference is not a literal secret.
# Only quoted direct assignments in source, or bare assignments in shell/env
# files, are candidates. Independently check recognizable token formats above.
CREDENTIAL_NAME = (
    r"(?:CT_PASSWORD|HF_TOKEN|GITHUB_TOKEN|"
    r"(?:[A-Za-z][A-Za-z0-9_]*_)?"
    r"(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET_KEY|CLIENT_SECRET|PASSWORD))"
)
ASSIGNMENT = re.compile(
    r"(?im)^[ \t]*(?:export[ \t]+)?[\"']?" + CREDENTIAL_NAME
    + r"[\"']?[ \t]*[:=][ \t]*(?P<quote>[\"'])(?P<value>[^\"'\r\n]*)(?P=quote)"
)
ENV_ASSIGNMENT = re.compile(
    r"(?im)^[ \t]*(?:export[ \t]+)?" + CREDENTIAL_NAME
    + r"[ \t]*=[ \t]*(?![\"'])(?P<value>[^\s#;,]+)"
)
PLACEHOLDERS = frozenset({
    "test", "testing", "password", "secret", "dummy", "example", "placeholder",
    "changeme", "change-me", "your-password", "your-api-key", "your-secret",
    "replace-me", "unused", "fake", "none", "null", "<password>", "<secret>",
})


class PackageError(ValueError):
    """Source tree or staged Git index is unsafe for publication."""


def allowed_path(path: str) -> bool:
    """Return whether a relative POSIX path is permitted by the release manifest."""
    parts = PurePosixPath(path).parts
    if (
        not parts or path != PurePosixPath(path).as_posix()
        or any(p in {".", ".."} for p in parts)
        or re.search(r"[\\:\x00-\x1f\x7f]", path)
    ):
        return False
    if path in EXACT:
        return True
    parent = PurePosixPath(path).parent.as_posix()
    suffix = PurePosixPath(path).suffix
    return parent in SOURCE_DIRS and suffix in SOURCE_DIRS[parent] and not path.startswith("/")


def _safe_name(path: str) -> None:
    if not allowed_path(path):
        raise PackageError(f"Not in source release allowlist: {path}")
    if any(SENSITIVE_NAME.search(part) for part in PurePosixPath(path).parts if part != ".env.example"):
        raise PackageError(f"Sensitive filename: {path}")


def _inspect_content(path: str, content: bytes) -> None:
    if b"\0" in content:
        raise PackageError(f"Binary content in source file: {path}")
    if PRIVATE_KEY.search(content) or CREDENTIAL.search(content):
        raise PackageError(f"Credential/private key found in: {path}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackageError(f"Non-UTF-8 source file: {path}") from exc
    assignments = list(ASSIGNMENT.finditer(text))
    if path.endswith((".sh", ".env.example")):
        assignments.extend(ENV_ASSIGNMENT.finditer(text))
    for match in assignments:
        value = match.group("value").strip().lower()
        if value and value not in PLACEHOLDERS and not value.startswith(("${", "$(", "<", "your_", "example_", "test_", "fake_")):
            raise PackageError(f"Literal credential assignment in: {path}")


def _read_regular_file(root: Path, rel: str) -> tuple[bytes, int]:
    path = root
    for part in PurePosixPath(rel).parts:
        path = path / part
        if path.is_symlink():
            raise PackageError(f"Symlink in release path: {rel}")
    if not path.is_file():
        raise PackageError(f"Missing regular source file: {rel}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise PackageError(f"Not a regular source file: {rel}")
        data = handle.read()
        mode = os.fstat(handle.fileno()).st_mode
    _inspect_content(rel, data)
    return data, 0o755 if mode & 0o111 else 0o644


def _source_paths(root: Path) -> set[str]:
    paths = {rel for rel in EXACT if (root / rel).exists() or (root / rel).is_symlink()}
    for directory, suffixes in SOURCE_DIRS.items():
        folder = root
        for part in PurePosixPath(directory).parts:
            folder = folder / part
            if folder.is_symlink():
                raise PackageError(f"Symlink in source directory: {directory}")
        if not folder.exists():
            continue
        if not folder.is_dir():
            raise PackageError(f"Not a source directory: {directory}")
        for file in folder.iterdir():
            if file.suffix in suffixes:
                paths.add(f"{directory}/{file.name}")
    return paths


def _tracked_paths(root: Path) -> set[str] | None:
    # Extracted release archives have no .git; an actual checkout must only
    # package paths recorded in its index, not untracked local source files.
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return {entry.decode("utf-8") for entry in result.stdout.split(b"\0") if entry}
    except UnicodeDecodeError as exc:
        raise PackageError("Non-UTF-8 Git index path") from exc


def allowed_files(root: Path) -> dict[str, tuple[bytes, int]]:
    """Inspect allowlisted source bytes; in Git checkouts include tracked files only."""
    root = Path(root)
    paths = _source_paths(root)
    tracked = _tracked_paths(root)
    if tracked is not None:
        paths &= tracked
    missing = REQUIRED - paths
    if missing:
        raise PackageError(f"Missing required release files: {', '.join(sorted(missing))}")
    files = {}
    for rel in sorted(paths):
        _safe_name(rel)
        files[rel] = _read_regular_file(root, rel)
    if not any(rel.startswith("app/") for rel in paths) or not any(rel.startswith("web/") for rel in paths):
        raise PackageError("Missing application or web source files")
    return files


def check_staged(root: Path) -> list[str]:
    """Check every Git index entry and its blob, never the working tree copy."""
    if not (root / ".git").exists():
        raise PackageError("--check-staged requires the root of a Git checkout")
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    staged: list[str] = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        try:
            meta, raw_path = entry.split(b"\t", 1)
            mode, oid, stage = meta.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise PackageError("Invalid Git index entry") from exc
        _safe_name(path)
        if stage != "0" or mode not in {"100644", "100755"}:
            raise PackageError(f"Nonregular or unmerged staged source: {path}")
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", oid],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
        _inspect_content(path, blob)
        staged.append(path)
    missing = REQUIRED - set(staged)
    if missing:
        raise PackageError(f"Missing required staged files: {', '.join(sorted(missing))}")
    return staged


def release_version(files: dict[str, tuple[bytes, int]]) -> str:
    project = tomllib.loads(files["pyproject.toml"][0].decode("utf-8"))["project"]
    version = project["version"]
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*(?:[a-zA-Z0-9.+-]+)?", version):
        raise PackageError("Invalid release version")
    return version


def build(root: Path, output_dir: Path, files: dict[str, tuple[bytes, int]]) -> Path:
    name = f"comic-translator-{release_version(files)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{name}.tar.gz"
    # A unique, exclusively created temp file prevents a preexisting symlink
    # from redirecting archive writes to user state; replace is atomic.
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_dir, prefix=f".{name}.", suffix=".tmp", delete=False,
    ) as temporary:
        partial = Path(temporary.name)
        try:
            with tarfile.open(fileobj=temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
                for rel, (data, mode) in sorted(files.items()):
                    info = tarfile.TarInfo(f"{name}/{rel}")
                    info.size = len(data)
                    info.mode = mode
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    try:
        partial.chmod(0o644)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check-only", action="store_true", help="inspect release source without creating an archive")
    operation.add_argument("--check-staged", action="store_true", help="inspect all Git-index paths and blob bytes")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist", help="output directory for the tar.gz")
    args = parser.parse_args(argv)
    try:
        if args.check_staged:
            staged = check_staged(ROOT)
            print(f"Staged source safe: {len(staged)} files")
        else:
            files = allowed_files(ROOT)
            if args.check_only:
                print(f"Release source safe: {len(files)} files (version {release_version(files)})")
            else:
                print(build(ROOT, args.output_dir, files))
    except (PackageError, OSError, subprocess.CalledProcessError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"Release packaging rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
