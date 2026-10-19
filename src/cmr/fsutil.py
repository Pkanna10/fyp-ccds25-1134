"""Crash-safe file writes and JSONL reads (torch-free).

Jobs may be killed at any instruction and re-run with the same arguments, so:

* whole-file outputs (``run_args.json``, ``summary.json``, ``resume.pt``, ...) go through
  :func:`write_atomic`: a kill leaves the old file or the new one, never a torn one;
* append-only JSONL logs (``predictions.jsonl``, ...) are written one ``json + "\\n"``
  record at a time; a kill can leave at most one torn last line, which
  :func:`read_jsonl` drops (and with ``repair=True`` cuts off), so a resumed run, which skips
  the records already present, redoes it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO


def write_atomic(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write ``path`` via a temporary file in the same folder, fsync, ``os.replace``.

    Args:
        write: called once with the binary temporary file; writes the whole new content.

    Example:

        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "summary.json"
        >>> write_atomic(p, lambda f: f.write(b'{"acc": 0.5}'))
        >>> p.read_text()
        '{"acc": 0.5}'
    """
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    try:  # fsync the folder so the rename survives a crash (POSIX; not supported everywhere)
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: Path, text: str) -> None:
    """:func:`write_atomic` for UTF-8 text."""
    write_atomic(path, lambda f: f.write(text.encode()))


def read_jsonl(path: Path, repair: bool = False) -> list[dict[str, Any]]:
    """Records of an append-only JSONL file; [] if it does not exist.

    An unterminated last line that parses as a JSON object is kept (only its newline was
    lost); any other is a torn record and is dropped with a warning.

    Args:
        repair: also fix the tail on disk (add the newline, or cut off the torn record);
            pass True only when about to append.

    Raises:
        ValueError: a newline-terminated line is not valid JSON (real corruption).

    Example (the last record lost only its newline, so it is kept and repaired):

        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "predictions.jsonl"
        >>> _ = p.write_text('{"q": 1}\\n{"q": 2}')
        >>> read_jsonl(p, repair=True)
        [{'q': 1}, {'q': 2}]
        >>> p.read_text()
        '{"q": 1}\\n{"q": 2}\\n'
    """
    if not path.exists():
        return []

    # 1. split off an unterminated tail: whole record (keep) or torn (drop)
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1  # every record before ``end`` is newline-terminated
    tail: dict[str, Any] | None = None
    if end < len(data) and data[end:].strip():
        tail = _json_object(data[end:])
        if tail is None:
            print(
                f"[io] {path}: dropping an unterminated last line ({len(data) - end} bytes, "
                "an interrupted write)",
                flush=True,
            )

    # 2. repair mode: make the file end on a newline so the next append starts cleanly
    if repair and end < len(data):
        with path.open("r+b") as f:
            if tail is not None:  # a whole record: terminate it
                f.seek(0, os.SEEK_END)
                f.write(b"\n")
            else:  # a torn record (or trailing blanks): cut it off
                f.truncate(end)
            f.flush()
            os.fsync(f.fileno())

    # 3. parse every terminated line; a bad one here is corruption, not a kill
    records: list[dict[str, Any]] = []
    for n, line in enumerate(data[:end].decode().split("\n"), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: corrupt record (not an interrupted append)") from e
    if tail is not None:
        records.append(tail)
    return records


def _json_object(raw: bytes) -> dict[str, Any] | None:
    """``raw`` decoded as one JSON object, else None (torn record or not an object)."""
    try:
        value = json.loads(raw.decode())
    except ValueError:  # incl. UnicodeDecodeError (a multi-byte character cut in half)
        return None
    return value if isinstance(value, dict) else None
