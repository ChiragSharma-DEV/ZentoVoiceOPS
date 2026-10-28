"""Knowledge Base parser, chunker, and precedence manager for VoiceOps 2026."""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Precedence mapping as defined in KB-00.1
# Lower tier number = higher priority
PRECEDENCE_TIERS = {
    "policy": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
    "meta": 6,
    "catalog": 7,
}


@dataclass
class Chunk:
    """A searchable knowledge base chunk."""
    section_id: str
    doc_id: str
    doc_type: str
    title: str               # Document title
    section_title: str       # Heading title
    version: str
    effective_from: str
    status: str              # 'current' or 'superseded'
    supersedes: str
    superseded_by: str
    piece_no: int            # 0-indexed piece of this section
    total_pieces: int
    text: str                # Full text of the chunk, including section_id
    file_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "section_title": self.section_title,
            "version": self.version,
            "effective_from": self.effective_from,
            "status": self.status,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "piece_no": self.piece_no,
            "total_pieces": self.total_pieces,
            "file_name": self.file_name,
            "text": self.text,
        }


def parse_header_block(content: str) -> Tuple[Dict[str, str], str]:
    """Parse YAML-style frontmatter header block and return (metadata, body)."""
    header = {
        "doc_id": "",
        "title": "",
        "doc_type": "",
        "audience": "",
        "language": "en",
        "version": "",
        "effective_from": "",
        "status": "current",
        "supersedes": "",
        "superseded_by": "",
    }
    header_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    m = header_pattern.match(content)
    if not m:
        return header, content

    raw_header = m.group(1)
    body = content[m.end():]
    for line in raw_header.splitlines():
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()

    return header, body


def split_section_into_pieces(
    section_text: str,
    max_chars: int = 1200
) -> List[str]:
    """Split very long sections into pieces on paragraph boundaries."""
    if len(section_text) <= max_chars:
        return [section_text]

    paragraphs = section_text.split("\n\n")
    pieces = []
    current_piece: List[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current_piece and (current_len + p_len + 2 > max_chars):
            pieces.append("\n\n".join(current_piece).strip())
            current_piece = [p]
            current_len = p_len
        else:
            current_piece.append(p)
            current_len += p_len + 2

    if current_piece:
        pieces.append("\n\n".join(current_piece).strip())

    return pieces if pieces else [section_text]


def load_and_chunk_kb(kb_dir: Optional[Path] = None, max_section_chars: int = 1200) -> List[Chunk]:
    """Parse every kb/*.md file into chunks with metadata and section IDs."""
    if kb_dir is None:
        dataset_dir = os.getenv("DATASET_DIR")
        if not dataset_dir:
            raise RuntimeError(
                "DATASET_DIR is not set. Set it in .env to the extracted "
                "Capstone_Project_Dataset_FTTT_2026 directory."
            )
        kb_dir = Path(dataset_dir) / "kb"

    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found at: {kb_dir}")

    section_pattern = re.compile(r"^##\s+\[([^\]]+)\]\s*(.*)$", re.MULTILINE)
    chunks: List[Chunk] = []

    for file_path in sorted(kb_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        header, body = parse_header_block(content)

        matches = list(section_pattern.finditer(body))
        if not matches:
            # Document has no ## [ID] section headings (e.g. meta or small note)
            continue

        for i, match in enumerate(matches):
            sec_id = match.group(1).strip()
            sec_title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_full = body[start:end].strip()

            pieces = split_section_into_pieces(section_full, max_chars=max_section_chars)
            total_pieces = len(pieces)

            for p_idx, p_text in enumerate(pieces):
                # Ensure section_id appears explicitly as text in the chunk for citation
                chunk_header = (
                    f"[{sec_id}] {header['title']} | Section: {sec_title}\n"
                    f"Document: {header['doc_id']} ({header['doc_type']}, status: {header['status']}, "
                    f"effective: {header['effective_from']})\n"
                )
                if total_pieces > 1:
                    chunk_header += f"(Part {p_idx + 1} of {total_pieces})\n"

                chunk_text = f"{chunk_header}\n{p_text}"

                chunk = Chunk(
                    section_id=sec_id,
                    doc_id=header["doc_id"],
                    doc_type=header["doc_type"],
                    title=header["title"],
                    section_title=sec_title,
                    version=header["version"],
                    effective_from=header["effective_from"],
                    status=header["status"],
                    supersedes=header["supersedes"],
                    superseded_by=header["superseded_by"],
                    piece_no=p_idx,
                    total_pieces=total_pieces,
                    text=chunk_text,
                    file_name=file_path.name,
                )
                chunks.append(chunk)

    return chunks


def report_chunking(chunks: List[Chunk]) -> Dict[str, Any]:
    """Compute and report chunking statistics: docs, sections, chunks, duplicates."""
    docs = set()
    sections = set()
    section_to_files: Dict[str, set] = {}

    for c in chunks:
        docs.add(c.file_name)
        sections.add(c.section_id)
        section_to_files.setdefault(c.section_id, set()).add(c.file_name)

    duplicates = {s_id: sorted(list(files)) for s_id, files in section_to_files.items() if len(files) > 1}

    return {
        "num_docs": len(docs),
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "duplicate_section_ids": duplicates,
    }


def add_business_days(start_date: str, days: int, holidays: Optional[set] = None) -> str:
    """Calculate the end date after adding a given number of business days.
    
    Monday to Saturday, excluding Sundays (weekday == 6) and holidays.
    """
    if holidays is None:
        try:
            from src.data import HOLIDAY_DATES
            holidays = HOLIDAY_DATES
        except Exception:
            holidays = {"2026-08-15", "2026-10-02", "2026-10-19", "2026-11-09"}

    cur = date.fromisoformat(start_date.strip()[:10])
    added = 0
    while added < days:
        cur += timedelta(days=1)
        if cur.weekday() != 6 and cur.isoformat() not in holidays:
            added += 1
    return cur.isoformat()
