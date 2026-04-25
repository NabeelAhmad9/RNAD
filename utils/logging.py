from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)



def save_json(path: str | Path, payload: Dict) -> None:
    content = json.dumps(payload, indent=2, default=str)
    atomic_write_text(path, content + "\n")



def append_dict_row_csv(path: str | Path, row: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    exists = p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)



def save_rows_csv(path: str | Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    atomic_write_text(path, buffer.getvalue())
