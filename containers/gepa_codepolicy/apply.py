"""Apply helpers: whole_file.v1 and a restricted unified_diff.v1."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_whole_file(current: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    content = str(payload.get("content") or "")
    expected = str(payload.get("content_hash") or "").strip()
    if expected and expected != sha256_text(content):
        return current, {
            "schema_id": "apply_report.v1",
            "patch_ok": False,
            "compile_ok": False,
            "restart_ok": True,
            "reject_reason": "content_hash_mismatch",
            "base_hash": sha256_text(current),
        }
    return content, {
        "schema_id": "apply_report.v1",
        "patch_ok": True,
        "compile_ok": True,
        "restart_ok": True,
        "path": payload.get("path"),
        "base_hash": sha256_text(current),
        "content_hash": sha256_text(content),
    }


def apply_unified_diff(original: str, diff_text: str) -> str:
    orig_lines = original.splitlines()
    out: list[str] = []
    src_i = 0
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    lines = diff_text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("---") or line.startswith("+++"):
            idx += 1
            continue
        match = hunk_re.match(line)
        if not match:
            idx += 1
            continue
        old_start = int(match.group(1)) - 1
        if old_start < 0:
            old_start = 0
        while src_i < old_start and src_i < len(orig_lines):
            out.append(orig_lines[src_i])
            src_i += 1
        idx += 1
        while idx < len(lines) and not lines[idx].startswith("@@"):
            hunk = lines[idx]
            if hunk.startswith("\\"):
                idx += 1
                continue
            if hunk.startswith(" "):
                out.append(hunk[1:])
                src_i += 1
            elif hunk.startswith("-"):
                src_i += 1
            elif hunk.startswith("+"):
                out.append(hunk[1:])
            elif hunk.startswith("---") or hunk.startswith("+++"):
                break
            else:
                break
            idx += 1
    out.extend(orig_lines[src_i:])
    text = "\n".join(out)
    if original.endswith("\n"):
        text += "\n"
    return text


def apply_source(current: str, value: Any) -> tuple[str, dict[str, Any], str]:
    if value is None:
        return current, {
            "schema_id": "apply_report.v1",
            "patch_ok": True,
            "compile_ok": True,
            "restart_ok": True,
            "protocol_id": "identity",
            "content_hash": sha256_text(current),
            "base_hash": sha256_text(current),
        }, "identity"
    if isinstance(value, str):
        report = {
            "schema_id": "apply_report.v1",
            "patch_ok": True,
            "compile_ok": True,
            "restart_ok": True,
            "protocol_id": "whole_file.v1",
            "content_hash": sha256_text(value),
            "base_hash": sha256_text(current),
        }
        return value, report, "whole_file.v1"
    if not isinstance(value, dict):
        return current, {
            "schema_id": "apply_report.v1",
            "patch_ok": False,
            "compile_ok": False,
            "reject_reason": "invalid_apply_payload",
        }, "unknown"
    protocol = str(value.get("protocol_id") or "")
    if protocol == "whole_file.v1" or (not protocol and value.get("content") is not None):
        source, report = apply_whole_file(current, value)
        report["protocol_id"] = protocol or "whole_file.v1"
        return source, report, report["protocol_id"]
    if protocol == "unified_diff.v1":
        expected_base = str(value.get("base_hash") or "").strip()
        if expected_base and expected_base != sha256_text(current):
            return current, {
                "schema_id": "apply_report.v1",
                "patch_ok": False,
                "compile_ok": False,
                "reject_reason": "base_hash_mismatch",
                "protocol_id": protocol,
                "base_hash": sha256_text(current),
            }, protocol
        try:
            source = apply_unified_diff(current, str(value.get("diff") or ""))
        except Exception as exc:  # noqa: BLE001
            return current, {
                "schema_id": "apply_report.v1",
                "patch_ok": False,
                "compile_ok": False,
                "reject_reason": f"diff_apply_failed:{exc}",
                "protocol_id": protocol,
            }, protocol
        return source, {
            "schema_id": "apply_report.v1",
            "patch_ok": True,
            "compile_ok": True,
            "restart_ok": True,
            "protocol_id": protocol,
            "path": value.get("path"),
            "base_hash": sha256_text(current),
            "content_hash": sha256_text(source),
        }, protocol
    return current, {
        "schema_id": "apply_report.v1",
        "patch_ok": False,
        "compile_ok": False,
        "reject_reason": f"unknown_protocol:{protocol or 'missing'}",
        "protocol_id": protocol or "unknown",
    }, protocol or "unknown"
