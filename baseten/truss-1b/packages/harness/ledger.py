"""Append-only results ledger, stored as JSON lines.

Two design choices carry the weight here, and both come from watching the failure they
prevent.

**Read line by line, never whole-file.** A whole-file `json.loads` on a JSON-*lines* file
dies on line 2 with "Extra data". If the caller treats that as "file unreadable", the
result is an audit that reports a single failure and inspects zero rows -- silently
exempting the exact artifact it exists to police. A corrupt line here is one bad row,
counted and reported as such, and every other row is still read.

**Refuse the write, do not warn.** A row missing provenance is not a row with a caveat.
It is a number nobody can reproduce, and once it is in the file it is indistinguishable
from a good one. So `append` raises rather than writing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "runs" / "ledger.jsonl"

# All eight are required. A row missing any of them is not written.
REQUIRED_PROVENANCE = (
    "reader_model",
    "reader_revision",
    "tokenizer_fingerprint",
    "arm",
    "split",
    "seed",
    "timestamp_utc",
    "corpus_hash",
)


class ProvenanceIncomplete(ValueError):
    """Raised instead of writing a row that could not be reproduced."""


@dataclass(frozen=True)
class BadLine:
    """A line that did not parse. Counted as one bad row, not a dead file."""

    lineno: int
    raw: str
    error: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def missing_provenance(row: dict) -> list[str]:
    return [f for f in REQUIRED_PROVENANCE if row.get(f) in (None, "")]


def append(row: dict, path: Path | str = DEFAULT_PATH) -> dict:
    """Append one result row. Raises if provenance is incomplete.

    Returns the row as written, with `timestamp_utc` filled in if absent, so callers can
    log exactly what landed.
    """
    row = dict(row)
    row.setdefault("timestamp_utc", now_utc())

    missing = missing_provenance(row)
    if missing:
        raise ProvenanceIncomplete(
            f"refusing to write ledger row; missing provenance: {', '.join(missing)}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=False)
    if "\n" in line:  # defensive: a literal newline would split one row into two
        raise ValueError("serialized row contains a newline; refusing to corrupt the ledger")

    # Append with an explicit flush+fsync. A ledger that loses its last row on a crash
    # after an expensive GPU run is worse than one that writes slowly.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return row


def read(path: Path | str = DEFAULT_PATH) -> Iterator[dict | BadLine]:
    """Yield every row in order. Malformed lines are yielded as BadLine, not raised.

    A missing file yields nothing, which is different from an empty file only in that
    neither is an error. Callers that care about the distinction should check existence.
    """
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield BadLine(lineno=lineno, raw=stripped[:200], error=str(exc))
                continue
            if not isinstance(parsed, dict):
                yield BadLine(lineno=lineno, raw=stripped[:200], error="row is not an object")
                continue
            yield parsed


def rows(path: Path | str = DEFAULT_PATH) -> list[dict]:
    """Just the good rows. Use `audit` when you need to know what was skipped."""
    return [r for r in read(path) if isinstance(r, dict)]


def audit(path: Path | str = DEFAULT_PATH) -> dict:
    """Inspect every line and report what is wrong, per line.

    The point is that `bad_lines` and `rows_missing_provenance` are counts of *rows*, so a
    file that fails to parse partway through still reports how many good rows it had.
    """
    good: list[dict] = []
    bad: list[BadLine] = []
    for item in read(path):
        (bad if isinstance(item, BadLine) else good).append(item)

    incomplete = [
        {"index": i, "missing": missing_provenance(r)}
        for i, r in enumerate(good)
        if missing_provenance(r)
    ]
    return {
        "path": str(path),
        "rows_read": len(good),
        "bad_lines": [b.__dict__ for b in bad],
        "rows_missing_provenance": incomplete,
        "clean": not bad and not incomplete,
    }
