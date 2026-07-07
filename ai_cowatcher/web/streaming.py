"""HTTP Range streaming helpers for local video files."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def guess_video_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "video/mp4"


def parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int]:
    """Parse a Range header into inclusive (start, end) byte offsets."""
    if not range_header or file_size <= 0:
        return 0, max(file_size - 1, 0)

    match = RANGE_RE.match(range_header.strip())
    if not match:
        return 0, file_size - 1

    start_s, end_s = match.groups()
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else file_size - 1
    end = min(end, file_size - 1)
    if start > end or start < 0:
        raise ValueError(f"Invalid Range header: {range_header!r}")
    return start, end


def iter_file_range(path: Path, start: int, end: int, chunk_size: int = 256 * 1024) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


async def async_iter_file_range(
    path: Path, start: int, end: int, chunk_size: int = 256 * 1024
) -> AsyncIterator[bytes]:
    for chunk in iter_file_range(path, start, end, chunk_size=chunk_size):
        yield chunk
