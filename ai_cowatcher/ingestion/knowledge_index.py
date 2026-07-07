"""Load and index curated knowledge files into Qdrant."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ai_cowatcher.config import Settings
from ai_cowatcher.domain import KnowledgeChunkRecord
from ai_cowatcher.interfaces import TextEmbedder
from ai_cowatcher.storage.qdrant_knowledge_store import QdrantKnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeIndexResult:
    title_id: str
    chunk_count: int
    source_path: str | None = None


def knowledge_file_for_title(knowledge_dir: str | Path, title_id: str) -> Path | None:
    base = Path(knowledge_dir)
    for name in (f"{title_id}.json", f"{title_id}.jsonl"):
        path = base / name
        if path.is_file():
            return path
    return None


def load_chunks_from_file(path: Path, title_id: str) -> list[KnowledgeChunkRecord]:
    if path.suffix == ".jsonl":
        return _load_jsonl(path, title_id)
    return _load_json(path, title_id)


def _load_json(path: Path, title_id: str) -> list[KnowledgeChunkRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "chunks" in data:
        raw_chunks = data["chunks"]
        file_title = str(data.get("title_id", title_id))
    elif isinstance(data, list):
        raw_chunks = data
        file_title = title_id
    else:
        raise ValueError(f"Unsupported knowledge file format: {path}")

    return [_parse_chunk(item, file_title, index) for index, item in enumerate(raw_chunks)]


def _load_jsonl(path: Path, title_id: str) -> list[KnowledgeChunkRecord]:
    chunks: list[KnowledgeChunkRecord] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        chunks.append(_parse_chunk(json.loads(line), title_id, index))
    return chunks


def _parse_chunk(item: dict, title_id: str, index: int) -> KnowledgeChunkRecord:
    chunk_id = str(item.get("chunk_id") or f"chunk_{index:04d}")
    text = str(item.get("text", "")).strip()
    if not text:
        raise ValueError(f"Knowledge chunk {chunk_id} has empty text")
    return KnowledgeChunkRecord(
        chunk_id=chunk_id,
        title_id=str(item.get("title_id", title_id)),
        category=str(item.get("category", "general")),
        text=text,
        source=str(item.get("source", "curated")),
    )


def index_title_knowledge(
    title_id: str,
    *,
    settings: Settings,
    embedder: TextEmbedder,
    knowledge_store: QdrantKnowledgeStore,
    knowledge_path: Path | None = None,
    replace: bool = True,
) -> KnowledgeIndexResult:
    """Embed and upsert curated knowledge chunks for one title."""
    path = knowledge_path or knowledge_file_for_title(settings.knowledge_dir, title_id)
    if path is None:
        logger.info("No knowledge file for title %s in %s", title_id, settings.knowledge_dir)
        return KnowledgeIndexResult(title_id=title_id, chunk_count=0)

    chunks = load_chunks_from_file(path, title_id)
    if not chunks:
        return KnowledgeIndexResult(title_id=title_id, chunk_count=0, source_path=str(path))

    if replace:
        knowledge_store.delete_title(title_id)

    knowledge_store.ensure_collection(embedder.vector_size)
    vectors = embedder.embed_texts([chunk.embedding_text for chunk in chunks])
    knowledge_store.upsert_chunks(chunks, vectors)
    logger.info("Indexed %d knowledge chunks for title %s from %s", len(chunks), title_id, path)
    return KnowledgeIndexResult(
        title_id=title_id, chunk_count=len(chunks), source_path=str(path)
    )
