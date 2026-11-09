"""Read Markdown policies and keep source identities through chunking."""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from docqa.text import sentences

DEFAULT_CORPUS = Path("data/corpus")


@dataclass(frozen=True)
class Chunk:
    """An indexed passage with a stable citation for this chunk configuration."""

    doc_id: str
    section: str
    chunk_id: str
    text: str

    @property
    def citation(self) -> str:
        """Return the identifier used inside square brackets."""
        return f"{self.doc_id}#{self.chunk_id}"


def _sections(markdown: str) -> Iterator[tuple[str, str]]:
    # Preserve the hidden benchmark text as untrusted evidence.
    markdown = re.sub(r"<!--(.*?)-->", r" \1 ", markdown, flags=re.DOTALL)
    parts = re.split(r"(?m)^ {0,3}#{1,6}[ \t]+([^\n]+)[ \t]*$", markdown)
    for heading, body in zip(["Overview", *parts[1::2]], parts[::2], strict=True):
        body = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", body)
        body = re.sub(r"(?m)^ {0,3}(?:[-+*>][ \t]+|[0-9]+\.[ \t]+|(?:~~~|```).*$)", "", body)
        body = re.sub(r"[*`]|_{1,2}(?=\w)|(?<=\w)_{1,2}", "", body)
        cleaned = " ".join(body.split())
        if cleaned:
            yield heading.strip().rstrip("#").strip(), cleaned


def _windows(text: str, size: int, overlap: int) -> Iterator[str]:
    words = text.split()
    boundaries = [0, *accumulate(len(sentence.split()) for sentence in sentences(text))]
    start = 0
    while start < len(words):
        limit = min(start + size, len(words))
        end = boundaries[bisect_right(boundaries, limit) - 1]
        if end <= start:
            end = limit  # A single sentence can exceed the configured window.
        yield " ".join(words[start:end])
        if end == len(words):
            break
        target = max(start + 1, end - overlap)
        aligned = boundaries[bisect_left(boundaries, target)]
        start = aligned if aligned <= end else target


def load_chunks(root: Path, size: int, overlap: int) -> list[Chunk]:
    """Load Markdown under root, packing sentences with up to overlap shared words."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size.")
    chunks = []
    for path in sorted(root.rglob("*.md")):
        doc_id = path.relative_to(root).with_suffix("").as_posix()
        if re.fullmatch(r"[\w./-]+", doc_id) is None:
            raise ValueError(f"Use letters, digits, dots, underscores or dashes in paths: {path}")
        index = 0
        for section, body in _sections(path.read_text(encoding="utf-8-sig")):
            for passage in _windows(body, size, overlap):
                index += 1
                chunks.append(Chunk(doc_id, section, f"c{index:02d}", passage))
    if not chunks:
        raise ValueError(f"No Markdown text found in {root}.")
    return chunks
