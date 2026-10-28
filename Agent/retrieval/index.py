"""Small BM25 index over markdown source documents.

Lexical and deterministic: no embedding model is required, so retrieval results are reproducible
and unit-testable. Each ``## `` section of a document is one chunk with a stable id such as
``MC-03``. ``corpus_version`` is a hash of all chunk ids, revisions and text, so any edit to a
source document changes the version recorded in every tool result and trace.
"""
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "in", "is",
    "it", "its", "of", "on", "or", "so", "that", "the", "their", "this", "to", "was", "what", "when", "which",
    "with", "you", "your", "me", "my", "about", "tell", "please", "explain", "describe",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    revision: str
    text: str


def _norm(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    out: List[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        parts = [raw] + (raw.split("_") if "_" in raw else [])
        for p in parts:
            if p and p not in STOP:
                out.append(_norm(p))
    return out


def parse_document(path: Path) -> List[Chunk]:
    """Split one markdown file (with a small ``---`` header) into one chunk per ``## `` section."""
    raw = path.read_text(encoding="utf-8")
    meta: Dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        _, head, body = raw.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
    for key in ("doc_id", "title", "revision"):
        if key not in meta:
            raise ValueError(f"{path.name}: missing '{key}' in header")
    chunks: List[Chunk] = []
    current, lines = None, []

    def flush():
        if current is not None and "".join(lines).strip():
            n = len(chunks) + 1
            chunks.append(Chunk(f"{meta['doc_id']}-{n:02d}", meta["doc_id"], meta["title"], current,
                                meta["revision"], " ".join(l.strip() for l in lines if l.strip())))

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            current, lines = line[3:].strip(), []
        elif current is not None:
            lines.append(line)
    flush()
    return chunks


class SearchIndex:
    K1, B = 1.5, 0.75

    def __init__(self, chunks: Sequence[Chunk]):
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk ids (two documents share a doc_id?)")
        self.chunks = list(chunks)
        self._tf = [Counter(tokenize(f"{c.section} {c.section} {c.title} {c.text}")) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        h = hashlib.sha256()
        for c in self.chunks:
            h.update(f"{c.chunk_id}|{c.revision}|{c.text}\n".encode("utf-8"))
        self.corpus_version = f"corpus-{h.hexdigest()[:10]}"

    @classmethod
    def from_dirs(cls, dirs: Iterable[Union[str, Path]]) -> "SearchIndex":
        chunks: List[Chunk] = []
        for d in dirs:
            for p in sorted(Path(d).glob("*.md")):
                chunks.extend(parse_document(p))
        if not chunks:
            raise ValueError("no source documents found")
        return cls(chunks)

    def get(self, chunk_id: str) -> Chunk:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        raise KeyError(chunk_id)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        q = tokenize(query)
        scored: List[Tuple[Chunk, float]] = []
        for c, tf, ln in zip(self.chunks, self._tf, self._len):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * ln / (self._avg or 1)))
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))
        return scored[:top_k]
