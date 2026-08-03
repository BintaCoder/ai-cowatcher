# ai-cowatcher

Pay-TV co-watcher pilot — an AI companion that answers viewer questions about what they're watching, without spoilers.

## Architecture

Full diagrams and cost notes: **[docs/E2E_ARCHITECTURE.md](docs/E2E_ARCHITECTURE.md)** (updated August 2026).

In short:

- **Offline ingestion** (once per title): scenes, Whisper/diarize, faces, vision captions, **per-scene audio clips** (object store), BGE-M3 vectors → Postgres + Qdrant (+ optional Neo4j characters, knowledge).
- **Real-time Q&A**: ambient browser listen (or type) → utterance gate → tool-calling agent (`scene_lookup` with spoiler filter, character/cast/knowledge/memory) → **text and/or multimodal** answer from retrieved scene WAVs + Gemini → short reply + optional TTS.
- **Streaming**: `POST /ask/stream` (SSE); full JSON still on `POST /ask`.
- **Navigation**: `POST /navigate` for clock jumps, events, and semantic seeks.
- **Latency hygiene**: warm BGE/sessions at startup, thread offload for sync work, session STT (not continuous thrash).
- **Observability**: structured `/ask` logs, `GET /metrics-lite`, Prometheus + Grafana.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
cp .env.example .env
# Set GEMINI_API_KEY (or GOOGLE_API_KEY) when MOCK_MODE=false
docker compose up -d
make api   # single process (avoid --reload with BGE warm)
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

```bash
cowatcher-ingest --title-id demo --video /path/to/video.mp4   # direct (no broker)
cowatcher-ingest-worker                                       # consume broker events
make worker
cowatcher-metrics-lite < ask.log
```

## Tests

```bash
pytest tests/ -v
```

## Stack

Python 3.11–3.13 · FastAPI · PySceneDetect · FFmpeg · faster-whisper · InsightFace · LiteLLM · BGE-M3 · Qdrant · PostgreSQL · Redis · Neo4j · object store (local or MinIO) · Gemini (default multimodal)
