"""Turn retrieved chunks into a numbered context block + human-readable citations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Citation:
    index: int
    doc_title: str
    section: str
    source_file: str
    version: str
    score: float

    def label(self) -> str:
        ver = f", {self.version}" if self.version else ""
        return f"[{self.index}] {self.doc_title} > {self.section} ({self.source_file}{ver})"


def build_citations(chunks: list) -> list[Citation]:
    cites: list[Citation] = []
    for i, ch in enumerate(chunks, start=1):
        m = ch.metadata
        cites.append(Citation(
            index=i,
            doc_title=m.get("doc_title", m.get("source_file", "")),
            section=m.get("section", ""),
            source_file=m.get("source_file", ""),
            version=m.get("version", ""),
            score=round(getattr(ch, "score", 0.0), 3),
        ))
    return cites


def format_context(chunks: list) -> str:
    """Numbered context passed to the LLM; the model must cite these [n] markers."""
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        m = ch.metadata
        header = f"[{i}] {m.get('breadcrumb', m.get('doc_title', ''))} (source: {m.get('source_file','')})"
        blocks.append(f"{header}\n{ch.text}")
    return "\n\n".join(blocks)


def format_sources(chunks: list) -> list[str]:
    return [c.label() for c in build_citations(chunks)]
