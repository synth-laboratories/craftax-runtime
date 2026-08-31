"""Persistent Craftax Rust engine REPL (stdin/stdout JSONL) for fast policy sweeps."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[2]
RUST_DIR = TASK_ROOT / "gold_rust"
REPL_BINARY = RUST_DIR / "target" / "release" / "craftax_repl"
PREBUILT_SCHEMA = "gamebench.prebuilt_binary.v1"
PREBUILT_CANONICAL_REPOSITORY = "https://github.com/JoshuaPurtell/gamebench.git"
PREBUILT_CANONICAL_COMMIT = "80c630db6ab35e7c9ae2b79eda51ac2bfc16ad6b"
PREBUILT_BUILDER_IMAGE = (
    "rust@sha256:8fa55b2f3ddf97471ab6a767bfa3f37e6bad0986ba823e75fea57e2a2a5c3073"
)
PREBUILT_WORKING_DIRECTORY = "/workspace/tasks/craftax-singleplayer/gold_rust"
PREBUILT_BUILD_COMMAND = (
    "cargo",
    "build",
    "--release",
    "--locked",
    "--bin",
    "craftax_repl",
)
PREBUILT_RUSTC_IDENTITY = "rustc version 1.97.0 (2d8144b78 2026-07-07)"
PREBUILT_DIR = TASK_ROOT / "fixtures" / "bin" / "linux-aarch64"
PREBUILT_BINARY = PREBUILT_DIR / "craftax_repl"
PREBUILT_MANIFEST = PREBUILT_DIR / "craftax_repl.manifest.json"
PREBUILT_SOURCE_CLOSURE = frozenset(
    {
        "Cargo.lock",
        "Cargo.toml",
        "src/bin/craftax_repl.rs",
        "src/lib.rs",
        "src/native.rs",
        "src/render.rs",
        "src/sprites.rs",
    }
)
_BUILD_LOCK = threading.Lock()
_REQUEST_ID = 0
_REQUEST_LOCK = threading.Lock()
_READOUT_MODES = frozenset({"full", "policy"})


@dataclass(frozen=True)
class PrebuiltRustReplManifest:
    """Strict identity contract for the tracked Linux/AArch64 fixture."""

    binary_sha256: str
    binary_git_blob_sha1: str
    binary_size_bytes: int
    binary_mode: int
    gnu_build_id_sha1: str
    dynamic_loader: str
    minimum_glibc: tuple[int, int]
    source_files: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> PrebuiltRustReplManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid Craftax prebuilt manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Craftax prebuilt manifest must be an object: {path}"
            )
        _require_exact_keys(
            payload, {"schema", "artifact", "target", "source", "build"}, path
        )
        schema = _required_string(payload, "schema", path)
        if schema != PREBUILT_SCHEMA:
            raise RuntimeError(
                f"unsupported Craftax prebuilt manifest schema in {path}: {schema!r}"
            )

        artifact = _required_object(payload, "artifact", path)
        target = _required_object(payload, "target", path)
        source = _required_object(payload, "source", path)
        build = _required_object(payload, "build", path)
        reproducibility = _required_object(build, "reproducibility", path)
        raw_source_files = _required_object(source, "files", path)

        _require_exact_keys(
            artifact,
            {
                "name",
                "path",
                "sha256",
                "git_blob_sha1",
                "size_bytes",
                "mode",
                "gnu_build_id_sha1",
                "executable",
            },
            path,
        )
        _require_exact_keys(
            target,
            {
                "os",
                "architecture",
                "triple",
                "format",
                "dynamic_loader",
                "minimum_glibc",
            },
            path,
        )
        _require_exact_keys(
            source, {"repository", "canonical_commit", "root", "files"}, path
        )
        _require_exact_keys(
            build,
            {
                "container_image",
                "working_directory",
                "command",
                "rustc",
                "reproducibility",
            },
            path,
        )
        _require_exact_keys(
            reproducibility,
            {
                "verified_at",
                "canonical_rebuild_sha256",
                "independent_rebuild_sha256",
                "byte_identical",
            },
            path,
        )

        name = _required_string(artifact, "name", path)
        artifact_path = _required_string(artifact, "path", path)
        binary_sha256 = _required_string(artifact, "sha256", path)
        binary_git_blob_sha1 = _required_string(artifact, "git_blob_sha1", path)
        binary_size_bytes = _required_int(artifact, "size_bytes", path)
        binary_mode_raw = _required_string(artifact, "mode", path)
        gnu_build_id_sha1 = _required_string(
            artifact, "gnu_build_id_sha1", path
        )
        executable = _required_bool(artifact, "executable", path)
        if name != "craftax_repl" or artifact_path != "craftax_repl" or not executable:
            raise RuntimeError(f"invalid Craftax artifact identity in {path}")
        if not _is_sha256(binary_sha256):
            raise RuntimeError(f"invalid binary sha256 in {path}: {binary_sha256!r}")
        if not _is_sha1(binary_git_blob_sha1):
            raise RuntimeError(
                f"invalid binary git_blob_sha1 in {path}: {binary_git_blob_sha1!r}"
            )
        if not _is_sha1(gnu_build_id_sha1):
            raise RuntimeError(
                f"invalid GNU build ID in {path}: {gnu_build_id_sha1!r}"
            )
        if binary_size_bytes <= 0:
            raise RuntimeError(f"invalid binary size_bytes in {path}: {binary_size_bytes!r}")
        if binary_mode_raw != "0755":
            raise RuntimeError(f"invalid binary mode in {path}: {binary_mode_raw!r}")

        target_os = _required_string(target, "os", path)
        target_architecture = _required_string(target, "architecture", path)
        target_triple = _required_string(target, "triple", path)
        target_format = _required_string(target, "format", path)
        dynamic_loader = _required_string(target, "dynamic_loader", path)
        minimum_glibc_raw = _required_string(target, "minimum_glibc", path)
        if (
            target_os != "linux"
            or target_architecture != "aarch64"
            or target_triple != "aarch64-unknown-linux-gnu"
            or target_format != "elf64-littleaarch64-pie"
            or dynamic_loader != "/lib/ld-linux-aarch64.so.1"
        ):
            raise RuntimeError(
                f"invalid Craftax fixture target identity in {path}"
            )
        minimum_glibc = _parse_version(minimum_glibc_raw, "minimum_glibc", path)

        repository = _required_string(source, "repository", path)
        canonical_commit = _required_string(source, "canonical_commit", path)
        source_root = _required_string(source, "root", path)
        expected_source_root = "tasks/craftax-singleplayer/gold_rust"
        if (
            repository != PREBUILT_CANONICAL_REPOSITORY
            or canonical_commit != PREBUILT_CANONICAL_COMMIT
            or not _is_sha1(canonical_commit)
            or source_root != expected_source_root
        ):
            raise RuntimeError(
                f"invalid canonical Craftax source identity in {path}"
            )

        source_files: dict[str, str] = {}
        for relative_path, digest in raw_source_files.items():
            if not isinstance(relative_path, str) or not _safe_relative_path(relative_path):
                raise RuntimeError(
                    f"invalid source path in Craftax prebuilt manifest {path}: {relative_path!r}"
                )
            if not _is_sha256(digest):
                raise RuntimeError(
                    f"invalid source sha256 in Craftax prebuilt manifest {path}: "
                    f"{relative_path}={digest!r}"
                )
            source_files[relative_path] = digest
        if not source_files:
            raise RuntimeError(f"Craftax prebuilt manifest has no source closure: {path}")
        if set(source_files) != PREBUILT_SOURCE_CLOSURE:
            raise RuntimeError(
                f"Craftax prebuilt manifest source closure changed in {path}: "
                f"expected={sorted(PREBUILT_SOURCE_CLOSURE)}, "
                f"actual={sorted(source_files)}"
            )

        container_image = _required_string(build, "container_image", path)
        working_directory = _required_string(build, "working_directory", path)
        command = _required_string_list(build, "command", path)
        rustc_identity = _required_string(build, "rustc", path)
        if (
            container_image != PREBUILT_BUILDER_IMAGE
            or working_directory != PREBUILT_WORKING_DIRECTORY
            or tuple(command) != PREBUILT_BUILD_COMMAND
            or rustc_identity != PREBUILT_RUSTC_IDENTITY
        ):
            raise RuntimeError(f"invalid Craftax fixture build identity in {path}")

        verified_at = _required_string(reproducibility, "verified_at", path)
        canonical_rebuild_sha256 = _required_string(
            reproducibility, "canonical_rebuild_sha256", path
        )
        independent_rebuild_sha256 = _required_string(
            reproducibility, "independent_rebuild_sha256", path
        )
        byte_identical = _required_bool(reproducibility, "byte_identical", path)
        try:
            datetime.strptime(verified_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise RuntimeError(
                f"invalid reproducibility timestamp in {path}: {verified_at!r}"
            ) from exc
        if (
            canonical_rebuild_sha256 != binary_sha256
            or independent_rebuild_sha256 != binary_sha256
            or not byte_identical
        ):
            raise RuntimeError(f"invalid Craftax fixture reproducibility proof in {path}")

        return cls(
            binary_sha256=binary_sha256,
            binary_git_blob_sha1=binary_git_blob_sha1,
            binary_size_bytes=binary_size_bytes,
            binary_mode=0o755,
            gnu_build_id_sha1=gnu_build_id_sha1,
            dynamic_loader=dynamic_loader,
            minimum_glibc=minimum_glibc,
            source_files=source_files,
        )

    def verify_source_closure(self) -> None:
        mismatches: list[str] = []
        for relative_path, expected_sha256 in sorted(self.source_files.items()):
            source_path = RUST_DIR / relative_path
            if source_path.is_symlink() or not source_path.is_file():
                raise RuntimeError(
                    f"Craftax prebuilt source is missing or not a regular file: {source_path}"
                )
            if _sha256_file(source_path) != expected_sha256:
                mismatches.append(relative_path)
        if mismatches:
            raise RuntimeError(
                "Craftax prebuilt source digest mismatch; publish a new fixture for: "
                + ", ".join(mismatches)
            )

    def verify_fixture_binary(self, path: Path) -> None:
        metadata = self._verify_binary_content(path)
        actual_mode = stat.S_IMODE(metadata.st_mode)
        if not actual_mode & stat.S_IXUSR:
            raise RuntimeError(
                f"Craftax prebuilt fixture is not executable at {path}: "
                f"actual={actual_mode:#05o}"
            )

    def verify_materialized_binary(self, path: Path) -> None:
        metadata = self._verify_binary_content(path)
        actual_mode = stat.S_IMODE(metadata.st_mode)
        if actual_mode != self.binary_mode:
            raise RuntimeError(
                f"Craftax materialized binary mode mismatch at {path}: "
                f"expected={self.binary_mode:#05o}, actual={actual_mode:#05o}"
            )

    def _verify_binary_content(self, path: Path) -> os.stat_result:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Craftax prebuilt binary is missing: {path}")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Craftax prebuilt binary is not a regular file: {path}")
        if metadata.st_size != self.binary_size_bytes:
            raise RuntimeError(
                f"Craftax prebuilt binary size mismatch at {path}: "
                f"expected={self.binary_size_bytes}, actual={metadata.st_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != self.binary_sha256:
            raise RuntimeError(
                f"Craftax prebuilt binary digest mismatch at {path}: "
                f"expected={self.binary_sha256}, actual={actual_sha256}"
            )
        actual_git_blob_sha1 = _git_blob_sha1_file(path)
        if actual_git_blob_sha1 != self.binary_git_blob_sha1:
            raise RuntimeError(
                f"Craftax prebuilt Git blob mismatch at {path}: "
                f"expected={self.binary_git_blob_sha1}, actual={actual_git_blob_sha1}"
            )
        _verify_elf_identity(path, self)
        return metadata

    def verify_host_compatibility(self) -> None:
        loader_path = Path(self.dynamic_loader)
        if not loader_path.is_file() or not os.access(loader_path, os.X_OK):
            raise RuntimeError(
                f"Craftax prebuilt dynamic loader is unavailable: {loader_path}"
            )
        actual_glibc = _host_glibc_version()
        if actual_glibc < self.minimum_glibc:
            raise RuntimeError(
                "Craftax prebuilt binary requires newer glibc: "
                f"required={_format_version(self.minimum_glibc)}, "
                f"host={_format_version(actual_glibc)}"
            )


def ensure_rust_repl_binary() -> Path:
    if _host_uses_linux_aarch64_fixture():
        with _BUILD_LOCK:
            manifest = PrebuiltRustReplManifest.load(PREBUILT_MANIFEST)
            manifest.verify_source_closure()
            manifest.verify_fixture_binary(PREBUILT_BINARY)
            manifest.verify_host_compatibility()
            if not _binary_matches_manifest(REPL_BINARY, manifest):
                _materialize_prebuilt_binary(manifest)
            manifest.verify_materialized_binary(REPL_BINARY)
            return REPL_BINARY

    host_target_dir, host_repl_binary = _host_build_paths()
    if _binary_is_current(host_repl_binary):
        return host_repl_binary
    with _BUILD_LOCK:
        if _binary_is_current(host_repl_binary):
            return host_repl_binary
        command = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--quiet",
            "--manifest-path",
            str(RUST_DIR / "Cargo.toml"),
            "--bin",
            "craftax_repl",
        ]
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(host_target_dir)
        subprocess.run(command, check=True, env=environment)
        if not host_repl_binary.is_file():
            raise RuntimeError(
                f"Rust REPL binary missing after host build: {host_repl_binary}"
            )
        return host_repl_binary


def _required_object(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Craftax prebuilt manifest {path} requires object field {key!r}")
    return value


def _required_string(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"Craftax prebuilt manifest {path} requires non-empty string field {key!r}"
        )
    return value


def _required_string_list(
    payload: dict[str, Any], key: str, path: Path
) -> list[str]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RuntimeError(
            f"Craftax prebuilt manifest {path} requires string-list field {key!r}"
        )
    return value


def _required_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise RuntimeError(
            f"Craftax prebuilt manifest {path} requires integer field {key!r}"
        )
    return value


def _required_bool(payload: dict[str, Any], key: str, path: Path) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise RuntimeError(
            f"Craftax prebuilt manifest {path} requires boolean field {key!r}"
        )
    return value


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], path: Path
) -> None:
    actual = set(payload)
    if actual != expected:
        raise RuntimeError(
            f"Craftax prebuilt manifest {path} has ambiguous fields: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_sha1(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _parse_version(value: str, field: str, path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if match is None:
        raise RuntimeError(
            f"invalid {field} version in Craftax prebuilt manifest {path}: {value!r}"
        )
    return int(match.group(1)), int(match.group(2))


def _format_version(value: tuple[int, int]) -> str:
    return f"{value[0]}.{value[1]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1_file(path: Path) -> str:
    metadata = path.stat()
    digest = hashlib.sha1()
    digest.update(f"blob {metadata.st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_platform_identity() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if system not in {"Darwin", "Linux"}:
        raise RuntimeError(f"unsupported Craftax REPL host operating system: {system!r}")
    machine_aliases = {
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }
    architecture = machine_aliases.get(machine)
    if architecture is None:
        raise RuntimeError(f"unsupported Craftax REPL host architecture: {machine!r}")
    return system.lower(), architecture


def _host_uses_linux_aarch64_fixture() -> bool:
    return _host_platform_identity() == ("linux", "aarch64")


def _host_build_paths() -> tuple[Path, Path]:
    system, architecture = _host_platform_identity()
    target_dir = (
        RUST_DIR / "target" / "gamebench-host" / f"{system}-{architecture}"
    )
    return target_dir, target_dir / "release" / "craftax_repl"


def _binary_matches_manifest(
    binary: Path, manifest: PrebuiltRustReplManifest
) -> bool:
    try:
        manifest.verify_materialized_binary(binary)
    except RuntimeError:
        return False
    return True


def _materialize_prebuilt_binary(manifest: PrebuiltRustReplManifest) -> None:
    REPL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".craftax_repl.", dir=REPL_BINARY.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(PREBUILT_BINARY, temporary_path)
        temporary_path.chmod(0o755)
        manifest.verify_materialized_binary(temporary_path)
        os.replace(temporary_path, REPL_BINARY)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_elf_identity(path: Path, manifest: PrebuiltRustReplManifest) -> None:
    data = path.read_bytes()
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise RuntimeError(f"Craftax prebuilt binary is not ELF64 little-endian: {path}")

    elf_type = struct.unpack_from("<H", data, 16)[0]
    machine = struct.unpack_from("<H", data, 18)[0]
    elf_version = struct.unpack_from("<I", data, 20)[0]
    program_header_offset = struct.unpack_from("<Q", data, 32)[0]
    elf_header_size = struct.unpack_from("<H", data, 52)[0]
    program_header_size = struct.unpack_from("<H", data, 54)[0]
    program_header_count = struct.unpack_from("<H", data, 56)[0]
    if (
        elf_type != 3
        or machine != 183
        or elf_version != 1
        or elf_header_size != 64
        or program_header_size != 56
        or program_header_count <= 0
    ):
        raise RuntimeError(f"invalid Linux/AArch64 PIE ELF header in {path}")
    headers_end = program_header_offset + program_header_size * program_header_count
    if program_header_offset < elf_header_size or headers_end > len(data):
        raise RuntimeError(f"invalid ELF program-header bounds in {path}")

    interpreters: list[str] = []
    build_ids: list[str] = []
    for index in range(program_header_count):
        offset = program_header_offset + index * program_header_size
        program_type = struct.unpack_from("<I", data, offset)[0]
        file_offset = struct.unpack_from("<Q", data, offset + 8)[0]
        file_size = struct.unpack_from("<Q", data, offset + 32)[0]
        segment_end = file_offset + file_size
        if segment_end > len(data):
            raise RuntimeError(f"invalid ELF segment bounds in {path}")
        if program_type == 3:
            raw_interpreter = data[file_offset:segment_end]
            if not raw_interpreter.endswith(b"\0"):
                raise RuntimeError(f"unterminated ELF interpreter in {path}")
            try:
                interpreters.append(raw_interpreter[:-1].decode("ascii"))
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"non-ASCII ELF interpreter in {path}") from exc
        elif program_type == 4:
            build_ids.extend(_gnu_build_ids_from_notes(data, file_offset, segment_end))

    if len(interpreters) != 1 or interpreters[0] != manifest.dynamic_loader:
        raise RuntimeError(
            f"Craftax ELF loader mismatch at {path}: "
            f"expected={manifest.dynamic_loader!r}, actual={interpreters!r}"
        )
    if len(build_ids) != 1 or build_ids[0] != manifest.gnu_build_id_sha1:
        raise RuntimeError(
            f"Craftax ELF Build ID mismatch at {path}: "
            f"expected={manifest.gnu_build_id_sha1}, actual={build_ids!r}"
        )
    if PREBUILT_RUSTC_IDENTITY.encode("ascii") not in data:
        raise RuntimeError(f"Craftax ELF rustc identity mismatch at {path}")

    glibc_versions = {
        (int(major), int(minor))
        for major, minor in re.findall(rb"GLIBC_(\d+)\.(\d+)", data)
    }
    if not glibc_versions or max(glibc_versions) != manifest.minimum_glibc:
        raise RuntimeError(
            f"Craftax ELF glibc requirement mismatch at {path}: "
            f"manifest={_format_version(manifest.minimum_glibc)}, "
            f"binary={_format_version(max(glibc_versions)) if glibc_versions else 'missing'}"
        )


def _gnu_build_ids_from_notes(data: bytes, start: int, end: int) -> list[str]:
    build_ids: list[str] = []
    cursor = start
    while cursor + 12 <= end:
        name_size, descriptor_size, note_type = struct.unpack_from("<III", data, cursor)
        name_start = cursor + 12
        name_end = name_start + name_size
        descriptor_start = (name_end + 3) & ~3
        descriptor_end = descriptor_start + descriptor_size
        if name_end > end or descriptor_end > end:
            raise RuntimeError("invalid GNU ELF note bounds")
        name = data[name_start:name_end].rstrip(b"\0")
        if note_type == 3 and name == b"GNU":
            build_ids.append(data[descriptor_start:descriptor_end].hex())
        cursor = (descriptor_end + 3) & ~3
    return build_ids


def _host_glibc_version() -> tuple[int, int]:
    try:
        raw_version = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError("unable to determine host glibc version") from exc
    if raw_version is None:
        raise RuntimeError("host did not report a glibc version")
    match = re.fullmatch(r"glibc (\d+)\.(\d+)", raw_version)
    if match is None:
        raise RuntimeError(f"ambiguous host glibc version: {raw_version!r}")
    return int(match.group(1)), int(match.group(2))


def _binary_is_current(binary: Path) -> bool:
    if not binary.is_file():
        return False
    source_paths = [
        RUST_DIR / "Cargo.lock",
        RUST_DIR / "Cargo.toml",
        *list((RUST_DIR / "src").rglob("*.rs")),
    ]
    source_mtime = max(path.stat().st_mtime for path in source_paths)
    return binary.stat().st_mtime >= source_mtime


def _next_request_id() -> int:
    global _REQUEST_ID
    with _REQUEST_LOCK:
        _REQUEST_ID += 1
        return _REQUEST_ID


class RustReplSession:
    """One long-lived craftax_repl subprocess per worker thread/process."""

    def __init__(self, *, binary_path: Path | None = None) -> None:
        binary = binary_path or ensure_rust_repl_binary()
        self._proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("failed to open craftax_repl pipes")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._io_lock = threading.Lock()
        ping = self._request({"op": "ping"})
        if not ping.get("ok"):
            raise RuntimeError(f"craftax_repl ping failed: {ping}")

    def close(self) -> None:
        with self._io_lock:
            if self._proc.poll() is not None:
                return
            try:
                self._write_unlocked({"id": _next_request_id(), "op": "close"})
                _ = self._read_unlocked()
            except OSError:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._proc.kill()
                except OSError:
                    pass

    def reset(
        self,
        *,
        task: dict[str, Any],
        seed: int,
        readout_mode: str = "full",
        replay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._request(
            {
                "op": "reset",
                "task": task,
                "seed": seed,
                "readout_mode": self._readout_mode(readout_mode),
                "replay": replay,
            }
        )
        if not response.get("ok"):
            raise RuntimeError(
                f"craftax_repl reset failed: {response.get('error', response)}"
            )
        return response

    def save_replay(self, path: Path) -> dict[str, Any]:
        response = self._request({"op": "save_replay", "path": str(path)})
        if not response.get("ok"):
            raise RuntimeError(
                f"craftax_repl save_replay failed: {response.get('error', response)}"
            )
        return response

    def render_png(self) -> dict[str, Any]:
        """Return a renderer-authentic PNG for the current world state."""

        response = self._request({"op": "render_png"})
        if not response.get("ok"):
            raise RuntimeError(
                f"craftax_repl render_png failed: {response.get('error', response)}"
            )
        return response

    def step(self, action: str, *, readout_mode: str = "full") -> dict[str, Any]:
        response = self._request(
            {
                "op": "step",
                "action": action,
                "readout_mode": self._readout_mode(readout_mode),
            }
        )
        if not response.get("ok"):
            raise RuntimeError(
                f"craftax_repl step failed: {response.get('error', response)}"
            )
        return response

    def steps(
        self, actions: list[str], *, readout_mode: str = "full"
    ) -> dict[str, Any]:
        response = self._request(
            {
                "op": "steps",
                "actions": actions,
                "readout_mode": self._readout_mode(readout_mode),
            }
        )
        if not response.get("ok"):
            raise RuntimeError(
                f"craftax_repl steps failed: {response.get('error', response)}"
            )
        return response

    def readout(self, *, readout_mode: str = "full") -> dict[str, Any]:
        response = self._request(
            {"op": "readout", "readout_mode": self._readout_mode(readout_mode)}
        )
        if not response.get("ok"):
            raise RuntimeError(
                f"craftax_repl readout failed: {response.get('error', response)}"
            )
        return response

    def _readout_mode(self, value: str) -> str:
        if value not in _READOUT_MODES:
            raise ValueError(f"unsupported craftax_repl readout_mode: {value}")
        return value

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._io_lock:
            request = {"id": _next_request_id(), **payload}
            self._write_unlocked(request)
            return self._read_unlocked()

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self._stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stdin.flush()

    def _read_unlocked(self) -> dict[str, Any]:
        while True:
            line = self._stdout.readline()
            if not line:
                code = self._proc.poll()
                raise RuntimeError(f"craftax_repl exited unexpectedly (code={code})")
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError(f"craftax_repl returned non-object: {parsed!r}")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
