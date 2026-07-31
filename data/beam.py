"""BEAM loader: the retention axis.

BEAM is the benchmark whose corpus reliably *exceeds* a context window, which is the only
regime where a memory system is provably not substitutable by a bigger window. LongMemEval
cannot make that argument; this can.

Splits, as published on the hub (`Mohammadta/BEAM`, CC BY-SA 4.0, ~106MB, 90 conversations):

    100K   20 conversations
    500K   35 conversations
    1M     35 conversations
    10M    separate configuration with different structural requirements

Each conversation carries probing questions across ten memory abilities (abstention,
contradiction resolution, event ordering, information extraction, instruction following,
knowledge update, multi-session reasoning, preference following, summarization, temporal
reasoning), so probe count is roughly ten times conversation count.

**Split choice is a statistics decision, not a taste one.** The tail-statistic gate needs
n >= 300. At ~10 probes per conversation that is 20 conversations -> ~200 probes on the 100K
split (below the gate) and 35 -> ~350 on 500K (above it). Pick accordingly and let the gate
enforce it rather than arguing with it.

**Downloads are off by default.** Nothing here touches the network unless a caller passes
`download=True` or sets `WMB_ALLOW_DOWNLOAD=1`, and the free-space preflight runs *before*
the first network call rather than after a partial write has already filled the disk.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("100K", "500K", "1M", "10M")
DEFAULT_SPLIT = "500K"
HF_REPO = "Mohammadta/BEAM"

# The published archive is ~106MB, but `datasets` stages a download, an extract, and an
# arrow cache. Requiring 2GB free keeps a fetch from wedging the disk partway through.
REQUIRED_FREE_BYTES = 2 * 1024**3

CACHE_DIR = Path(
    os.environ.get("WMB_BEAM_CACHE", str(Path.home() / ".cache" / "weight-memory-bench" / "beam"))
)


class DownloadNotPermitted(RuntimeError):
    """Raised when data is absent and the caller did not opt in to fetching it.

    Downloads are opt-in because an import that silently pulls 100MB+ is a surprise, and
    because a benchmark run should fail loudly on missing data rather than quietly acquire
    a corpus whose version nobody recorded.
    """


class InsufficientDiskSpace(RuntimeError):
    """Raised by the preflight, before any network call."""


@dataclass(frozen=True)
class Probe:
    """One probing question against one conversation."""

    conversation_id: str
    ability: str
    question: str
    answer: str

    @property
    def is_abstention(self) -> bool:
        return self.ability.lower().strip() == "abstention"


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    category: str
    chat: tuple[str, ...]
    probes: tuple[Probe, ...]

    def transcript_text(self) -> str:
        return "\n\n".join(self.chat)

    def turn_chunks(self) -> list[str]:
        return list(self.chat)


def preflight(path: Path | str = CACHE_DIR, required: int = REQUIRED_FREE_BYTES) -> dict:
    """Check free space on the cache volume. Runs before any network call.

    Checks the nearest existing ancestor, because `shutil.disk_usage` needs a real path and
    the cache directory may not exist yet.
    """
    path = Path(path)
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    result = {
        "path": str(path),
        "checked": str(probe),
        "free_bytes": usage.free,
        "required_bytes": required,
        "ok": usage.free >= required,
    }
    if not result["ok"]:
        raise InsufficientDiskSpace(
            f"need {required / 1024**3:.1f}GB free at {probe}, have {usage.free / 1024**3:.1f}GB"
        )
    return result


def _download_allowed(explicit: bool) -> bool:
    return explicit or os.environ.get("WMB_ALLOW_DOWNLOAD") == "1"


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_probes(conversation_id: str, raw) -> tuple[Probe, ...]:
    """Parse the probing questions payload.

    The hub types this column as a string, which in practice holds JSON. It is parsed
    defensively: a payload that cannot be understood yields no probes for that conversation
    rather than a guessed one, because a fabricated probe would silently enter the
    denominator of every arm's score.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ()

    items = raw if isinstance(raw, list) else raw.get("questions", []) if isinstance(raw, dict) else []
    probes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("probing_question") or ""
        answer = item.get("answer") or item.get("gold_answer") or ""
        ability = item.get("ability") or item.get("type") or item.get("category") or "unknown"
        if not question:
            continue
        probes.append(
            Probe(
                conversation_id=conversation_id,
                ability=str(ability),
                question=str(question),
                answer=str(answer),
            )
        )
    return tuple(probes)


def load(split: str = DEFAULT_SPLIT, *, download: bool = False, limit: int | None = None):
    """Load a BEAM split. Does not touch the network unless downloading is permitted."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "BEAM needs the `datasets` package: uv pip install -e '.[beam]'"
        ) from exc

    cached = CACHE_DIR.exists() and any(CACHE_DIR.iterdir())
    if not cached:
        if not _download_allowed(download):
            raise DownloadNotPermitted(
                f"BEAM {split} is not cached at {CACHE_DIR}. Pass download=True or set "
                "WMB_ALLOW_DOWNLOAD=1 to fetch it (~106MB, needs 2GB free)."
            )
        preflight()  # before the first network call, not after

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_REPO, split, cache_dir=str(CACHE_DIR))
    rows = ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds

    out = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        cid = str(row.get("conversation_id") or f"conv_{i}")
        seed = row.get("conversation_seed") or {}
        category = str(seed.get("category", "")) if isinstance(seed, dict) else ""
        chat = row.get("chat") or []
        out.append(
            Conversation(
                conversation_id=cid,
                category=category,
                chat=tuple(_as_text(c) for c in chat),
                probes=_parse_probes(cid, row.get("probing_questions")),
            )
        )
    return out


def all_probes(conversations) -> list[Probe]:
    return [p for c in conversations for p in c.probes]


def stats(conversations) -> dict:
    probes = all_probes(conversations)
    by_ability: dict[str, int] = {}
    for p in probes:
        by_ability[p.ability] = by_ability.get(p.ability, 0) + 1
    return {
        "conversations": len(conversations),
        "probes": len(probes),
        "abstention_probes": sum(1 for p in probes if p.is_abstention),
        "by_ability": dict(sorted(by_ability.items())),
    }
