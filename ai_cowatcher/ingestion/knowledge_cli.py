"""CLI for indexing curated title knowledge into Qdrant."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ai_cowatcher.config import get_settings
from ai_cowatcher.ingestion.knowledge_index import index_title_knowledge, knowledge_file_for_title
from ai_cowatcher.providers.factory import build_ingestion_providers
from ai_cowatcher.storage.qdrant_knowledge_store import QdrantKnowledgeStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index curated knowledge files into the co-watcher knowledge base"
    )
    parser.add_argument("--title-id", required=True, help="Title identifier")
    parser.add_argument(
        "--file",
        help="Path to knowledge JSON/JSONL (default: KNOWLEDGE_DIR/{title_id}.json)",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not delete existing chunks for this title before upserting",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    knowledge_path = Path(args.file) if args.file else knowledge_file_for_title(
        settings.knowledge_dir, args.title_id
    )
    if knowledge_path is None or not knowledge_path.is_file():
        print(f"No knowledge file found for title {args.title_id}", file=sys.stderr)
        return 1

    providers = build_ingestion_providers(settings)
    store = QdrantKnowledgeStore(settings)
    try:
        result = index_title_knowledge(
            args.title_id,
            settings=settings,
            embedder=providers.embedder,
            knowledge_store=store,
            knowledge_path=knowledge_path,
            replace=not args.no_replace,
        )
    except Exception:
        logging.exception("Knowledge indexing failed")
        return 1

    print(f"Indexed {result.chunk_count} knowledge chunk(s) for {result.title_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
