"""Source release tests use disposable fake trees and never inspect live service data."""

import importlib.util
from pathlib import Path
import subprocess
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "package_release", Path(__file__).resolve().parents[1] / "scripts/package_release.py"
)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def source_tree(tmp_path):
    for rel in package.REQUIRED:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "comic-translator"\nversion = "2.3.4"\n')
    for rel in ("app/__init__.py", "app/main.py", "web/index.html", "web/app.js", "scripts/install.sh", "tests/test_core.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test source\n")
    return tmp_path


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_archive_contains_only_allowlisted_source_with_portable_metadata(tmp_path):
    root = source_tree(tmp_path / "source")
    excluded = (
        "data/app.db", "data/secret.key", "data/jobs/private.png", "engines/koharu-src/src/main.rs",
        "engines/mangatranslator/model.py", "engines/koharu-target/release/bin", "engines/sysshim/lib/key",
        "fonts/korean/font.ttf", "app/__pycache__/main.pyc", ".venv/lib/private.py",
        "web/customer.png", "dist/old.tar.gz",
    )
    for rel in excluded:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private runtime bytes")
    files = package.allowed_files(root)
    archive = package.build(root, tmp_path / "output", files)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        assert {m.name for m in members} == {f"comic-translator-2.3.4/{p}" for p in files}
        assert all(m.isfile() and not m.name.startswith("/") and m.mtime == 0 and m.uid == m.gid == 0 for m in members)
        assert handle.extractfile("comic-translator-2.3.4/tests/fixtures/make_sample_pages.py").read() == b"fixture source\n"
        assert not any(b"private runtime bytes" in handle.extractfile(m).read() for m in members)
    assert not set(excluded) & files.keys()


def test_existing_temp_name_symlink_does_not_redirect_archive_writes(tmp_path):
    root = source_tree(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    private = tmp_path / "existing-state"
    private.write_bytes(b"preserve this file")
    collision = output / ".comic-translator-2.3.4.tar.gz.tmp"
    collision.symlink_to(private)
    archive = package.build(root, output, package.allowed_files(root))
    assert private.read_bytes() == b"preserve this file"
    assert collision.is_symlink()
    with tarfile.open(archive, "r:gz") as handle:
        assert handle.getmember("comic-translator-2.3.4/app/main.py").isfile()


def test_symlink_and_sensitive_filename_rejected_even_inside_source_folders(tmp_path):
    root = source_tree(tmp_path / "source")
    private = tmp_path / "private.py"
    private.write_text("never publish this\n")
    (root / "app" / "escape.py").symlink_to(private)
    with pytest.raises(package.PackageError, match="Symlink"):
        package.allowed_files(root)
    (root / "app" / "escape.py").unlink()
    (root / "app" / "credentials.py").write_text("safe looking content\n")
    with pytest.raises(package.PackageError, match="Sensitive filename"):
        package.allowed_files(root)


def test_git_checkout_ignores_untracked_sources_and_accepts_fixture_passwords(tmp_path):
    root = source_tree(tmp_path / "source")
    (root / "tests" / "test_core.py").write_text('CT_PASSWORD="password"\n')
    git(root, "init", "-q")
    git(root, "add", "-A")
    (root / "app" / "unpublished.py").write_text("private scratch code\n")
    assert "app/unpublished.py" not in package.allowed_files(root)
    assert "tests/test_core.py" in package.check_staged(root)


def test_index_bytes_are_inspected_instead_of_clean_worktree(tmp_path):
    root = source_tree(tmp_path / "source")
    git(root, "init", "-q")
    target = root / "app" / "main.py"
    target.write_text('API_KEY="plausible-non-placeholder-credential"\n')
    git(root, "add", "-A")
    target.write_text('API_KEY="placeholder"\n')
    with pytest.raises(package.PackageError, match="credential"):
        package.check_staged(root)
    git(root, "add", "app/main.py")
    assert "app/main.py" in package.check_staged(root)
    # A path excluded from the archive is unsafe in the Git index as well.
    (root / "data").mkdir()
    (root / "data" / "app.db").write_text("not publishable\n")
    git(root, "add", "-f", "data/app.db")
    with pytest.raises(package.PackageError, match="allowlist"):
        package.check_staged(root)


def test_staged_symlinks_and_private_keys_are_rejected(tmp_path):
    root = source_tree(tmp_path / "source")
    git(root, "init", "-q")
    (root / "app" / "main.py").write_text("-----BEGIN " + "PRIVATE KEY-----\nnot a fixture credential\n")
    git(root, "add", "-A")
    with pytest.raises(package.PackageError, match="private key"):
        package.check_staged(root)
    (root / "app" / "main.py").unlink()
    (root / "app" / "main.py").symlink_to("../../private.py")
    git(root, "add", "app/main.py")
    with pytest.raises(package.PackageError, match="Nonregular"):
        package.check_staged(root)


def test_known_credential_format_in_staged_blob_is_rejected(tmp_path):
    root = source_tree(tmp_path / "source")
    git(root, "init", "-q")
    (root / "app" / "main.py").write_text("# ghp_" + "A" * 40 + "\n")
    git(root, "add", "-A")
    (root / "app" / "main.py").write_text("# placeholder credential, not staged\n")
    with pytest.raises(package.PackageError, match="Credential/private key"):
        package.check_staged(root)


def test_code_annotations_and_references_are_not_literal_credentials(tmp_path):
    root = source_tree(tmp_path / "source")
    (root / "app" / "main.py").write_text(
        "password: str | None\npassword = password_from_file(path)\n"
        "token = secrets.token_urlsafe(32)\n"
    )
    (root / "scripts" / "install.py").write_text("password = read_secret(path)\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    assert "scripts/install.py" in package.check_staged(root)
    assert package.allowed_files(root)["app/main.py"][0].startswith(b"password: str")


def test_staged_quoted_password_literal_rejected_even_if_worktree_clean(tmp_path):
    root = source_tree(tmp_path / "source")
    git(root, "init", "-q")
    target = root / "scripts" / "install.py"
    target.write_text('password = "distinct-concrete-literal-value"\n')
    git(root, "add", "-A")
    target.write_text("password = password_from_file(path)\n")
    with pytest.raises(package.PackageError, match="Literal credential assignment"):
        package.check_staged(root)
