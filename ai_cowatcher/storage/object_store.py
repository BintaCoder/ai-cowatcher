"""Object storage for scene audio clips (MinIO or local filesystem)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ai_cowatcher.config import Settings

logger = logging.getLogger(__name__)


class ObjectStore(Protocol):
    def put_bytes(self, key: str, data: bytes, *, content_type: str = "audio/wav") -> str:
        """Store object; return the canonical key."""
        ...

    def get_bytes(self, key: str) -> bytes | None:
        ...

    def delete_prefix(self, prefix: str) -> int:
        """Delete all keys with the given prefix. Return count."""
        ...


class LocalFilesystemObjectStore:
    """Dev/test/mock store under a directory (no MinIO required)."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "audio/wav") -> str:
        del content_type
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get_bytes(self, key: str) -> bytes | None:
        path = self._key_path(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def delete_prefix(self, prefix: str) -> int:
        base = self._root / prefix.strip("/")
        if not base.exists():
            # prefix may be a folder stem
            count = 0
            for path in self._root.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(self._root).as_posix()
                    if rel.startswith(prefix.strip("/")):
                        path.unlink(missing_ok=True)
                        count += 1
            return count
        count = 0
        if base.is_file():
            base.unlink(missing_ok=True)
            return 1
        for path in base.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                count += 1
        return count

    def _key_path(self, key: str) -> Path:
        clean = key.lstrip("/").replace("..", "")
        return self._root / clean


class MinioObjectStore:
    """MinIO / S3-compatible object store."""

    def __init__(self, settings: Settings):
        from minio import Minio

        self._bucket = settings.minio_bucket
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("Created MinIO bucket %s", self._bucket)
        except Exception:  # noqa: BLE001
            logger.exception("MinIO bucket ensure failed for %s", self._bucket)
            raise

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "audio/wav") -> str:
        from io import BytesIO

        stream = BytesIO(data)
        self._client.put_object(
            self._bucket,
            key,
            stream,
            length=len(data),
            content_type=content_type,
        )
        return key

    def get_bytes(self, key: str) -> bytes | None:
        try:
            response = self._client.get_object(self._bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception:  # noqa: BLE001
            logger.debug("MinIO get failed key=%s", key, exc_info=True)
            return None

    def delete_prefix(self, prefix: str) -> int:
        from minio.deleteobjects import DeleteObject

        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        to_delete = [DeleteObject(obj.object_name) for obj in objects if obj.object_name]
        if not to_delete:
            return 0
        errors = list(self._client.remove_objects(self._bucket, to_delete))
        if errors:
            logger.warning("MinIO delete_prefix errors=%s", errors[:3])
        return len(to_delete)


def scene_audio_object_key(title_id: str, scene_id: str) -> str:
    safe_title = title_id.replace("/", "_")
    safe_scene = scene_id.replace("/", "_")
    return f"scenes/{safe_title}/{safe_scene}.wav"


def build_object_store(settings: Settings) -> ObjectStore:
    """Prefer MinIO when real mode + healthy; else local filesystem (tests/dev)."""
    if settings.mock_mode or settings.object_store_backend == "local":
        root = settings.object_store_local_dir or ".cowatcher-objects"
        logger.info("Object store: local filesystem root=%s", root)
        return LocalFilesystemObjectStore(root)

    if settings.object_store_backend == "minio":
        try:
            store = MinioObjectStore(settings)
            logger.info(
                "Object store: MinIO endpoint=%s bucket=%s",
                settings.minio_endpoint,
                settings.minio_bucket,
            )
            return store
        except Exception:  # noqa: BLE001
            logger.exception(
                "MinIO unavailable — falling back to local object store at %s",
                settings.object_store_local_dir,
            )
            return LocalFilesystemObjectStore(
                settings.object_store_local_dir or ".cowatcher-objects"
            )

    return LocalFilesystemObjectStore(settings.object_store_local_dir or ".cowatcher-objects")
