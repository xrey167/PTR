"""Shared by the freeze tools: write several files all or nothing."""

import os
import tempfile
from pathlib import Path


def atomic_write_all(updates):
    """Write {path: text} so that either all land or none: stage every file next to its target, then
    replace them one by one; if a replace fails, the files already replaced are restored."""
    staged = []
    replaced = []
    try:
        for path, text in updates.items():
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".freeze-")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            staged.append((path, Path(tmp)))
        for path, tmp in staged:
            original = path.read_bytes()
            os.replace(tmp, path)
            replaced.append((path, original))
    except BaseException:
        for path, original in replaced:
            path.write_bytes(original)
        raise
    finally:
        for _path, tmp in staged:
            tmp.unlink(missing_ok=True)
