# ai-cowatcher

Pay-TV co-watcher pilot — an AI companion that answers viewer questions about what they're watching, without spoilers.

## Architecture

Full diagrams and latency notes: **[docs/E2E_ARCHITECTURE.md](docs/E2E_ARCHITECTURE.md)** (v0.3, August 2026).

In short:

- **Offline ingestion** (once per title): scenes, Whisper, optional diarization, faces, captions, scene WAVs, BGE vectors → Postgres + Qdrant; cast cache (TMDB); optional Neo4j/knowledge.
- **Real-time Q&A (pilot path)**: typed or “**Hey** …” voice → gate → **scene prefetch (playhead-local when possible) + one streamed Gemini answer** (`PILOT_LOW_LATENCY` / `merged`). Full multi-tool agent is optional and slower.
- **Streaming**: `POST /ask/stream` (SSE); JSON on `POST /ask`.
- **Navigation**: `POST /navigate`.
- **Latency hygiene**: warm BGE, no multi-hop tools by default, multimodal audio **off** by default, QA cache, `make api` without `--reload`.
- **Observability**: `/metrics-lite`, Prometheus + Grafana.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
cp .env.example .env
# Set GEMINI_API_KEY (or GOOGLE_API_KEY) when MOCK_MODE=false
docker compose up -d
make api   # single process (avoid --reload with BGE warm)
# QA cache off by default; for cache benches/demos:
# make api QA_CACHE_ENABLED=true
```

With `MOCK_MODE=true`, AI providers use local mocks — no cloud keys required for tests.

Re-ingest titles after enabling scene audio so clips + `audio_object_key` are populated.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Dependency health + config summary |
| `GET /watch` | Watch UI + ambient voice |
| `POST /catalog/titles` | Register a title and publish ingest |
| `POST /ingest` | Publish ingest event |
| `POST /ask` | Full JSON co-watcher Q&A |
| `POST /ask/stream` | SSE progressive Q&A |
| `POST /navigate` | Jump to timestamp / moment |
| `GET /metrics-lite` | Pilot KPIs |
| `GET /metrics` | Prometheus scrape |

## Observability

```bash
make up                    # includes Prometheus (:9090) and Grafana (:3000)
make api                   # GET /metrics on :8000
make worker                # ingest worker metrics on :9100/metrics
```

Grafana login: `admin` / `cowatcher`. Dashboard: **AI Co-watcher Pilot**.  
Alert thresholds: [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

## CLI

Always use the project venv (or `make`), so `ai_cowatcher` resolves:

```bash
source .venv/bin/activate
pip install -e ".[dev]"   # if you see: No module named 'ai_cowatcher'

# Preferred (works even if console scripts are stale):
python -m ai_cowatcher.ingestion.cli --title-id demo --video /path/to/video.mp4 --force

# Or:
make ingest TITLE=demo VIDEO=/path/to/video.mp4 FORCE=1

# After install, these also work:
cowatcher-ingest --title-id demo --video /path/to/video.mp4
cowatcher-ingest-worker
make worker
cowatcher-metrics-lite < ask.log
```

## Tests

```bash
pytest tests/ -v
```

## Stack

Python 3.11–3.13 · FastAPI · PySceneDetect · FFmpeg · faster-whisper · InsightFace · LiteLLM · BGE-M3 · Qdrant · PostgreSQL · Redis · Neo4j · object store (local or MinIO) · Gemini (default multimodal)
