"""
Progress reporting for file transfers (Drive uploads/downloads, shard copies).

Kept free of imports from the rest of the package so ``gdrive``, ``data``,
``utils`` and ``tokenizer`` can all use it without import cycles.

A *progress* argument accepted by the transfer functions is one of:
  - ``True``  -- draw a byte bar, but only for transfers of at least ``MIN_BAR_BYTES``
                 (a 2 MB tokenizer or a 6 KB manifest stays quiet);
  - ``False`` -- silent;
  - a callable ``on_bytes(n)`` -- called with the number of *new* bytes moved, so a
    caller (e.g. a folder download) can drive its own overall bar.
"""

import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from tqdm.auto import tqdm

CHUNK_BYTES = 8 * 1024 * 1024  # multiple of 256 KiB, as the Drive resumable protocol requires
MIN_BAR_BYTES = 10 * 1024 * 1024

ProgressArg = bool | Callable[[int], None]


def _noop(_n: int) -> None:
    pass


@contextmanager
def transfer_progress(progress: ProgressArg, total: int, desc: str) -> Iterator[Callable[[int], None]]:
    """Yield an ``on_bytes(n)`` callback for a transfer of *total* bytes described by *desc*."""
    if callable(progress):
        yield progress
        return
    if not progress or total < MIN_BAR_BYTES:
        yield _noop
        return
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc, leave=False) as bar:
        yield bar.update


def copy_with_progress(src: str, dst: str, on_bytes: Callable[[int], None] = _noop,
                       chunk_bytes: int = CHUNK_BYTES) -> str:
    """Copy *src* to *dst* in chunks (``shutil.copy2`` semantics: data plus timestamps/mode).

    ``on_bytes`` receives the size of each chunk as it lands. Returns *dst*.
    """
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            buf = fin.read(chunk_bytes)
            if not buf:
                break
            fout.write(buf)
            on_bytes(len(buf))
    shutil.copystat(src, dst)
    return dst


def human_size(n_bytes: int | float) -> str:
    """``1234567`` -> ``'1.2 MB'`` (decimal units, matching what Drive and Colab show)."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"  # pragma: no cover


def file_size(path: str) -> int:
    """Size in bytes, 0 if the file is missing (progress totals must never raise)."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
