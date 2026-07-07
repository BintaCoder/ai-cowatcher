# ai-cowatcher

Pay-TV co-watcher pilot — an AI companion that answers viewer questions about what they're watching, without spoilers.

## Architecture

- **Offline ingestion** (once per title): scene detection, transcription, face clustering, vision captioning, and vector indexing into Postgres + Qdrant.
- **Real-time Q&A** (per question): a single orchestrating conversation agent with `scene_lookup` tool-calling; spoiler safety enforced via `end_ts <= current_ts` at retrieval time only.
- **Tiered LLM routing**: fast model by default, escalated model for nuanced questions (config-driven).
- **Pilot observability**: structured JSON logs per `/ask`, `GET /metrics-lite` rollups, and Prometheus + Grafana (`GET /metrics`, dashboard at `:3000`).

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
cowatcher-api   # or: make api
```

With `MOCK_MODE=true` (default), AI providers use local mocks — no API keys required for development and tests.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Dependency health + config summary |
| `POST /catalog/titles` | Register a new title and publish an ingest event |
| `POST /ingest` | Publish an ingest event for an existing or new title |
| `POST /ask` | Real-time co-watcher Q&A |
| `GET /metrics-lite` | Pilot KPIs (latency, escalation, don't-know rate) |
| `GET /metrics` | Prometheus scrape endpoint |

## Observability

```bash
make up                    # includes Prometheus (:9090) and Grafana (:3000)
make api                   # exposes GET /metrics on :8000
make worker                # ingest worker metrics on :9100/metrics
```

Grafana login: `admin` / `cowatcher`. Dashboard: **AI Co-watcher Pilot**.  
Alert thresholds (documented, not paged): see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

## CLI

```bash
cowatcher-ingest --title-id demo --video /path/to/video.mp4   # direct (no broker)
cowatcher-ingest-worker                                       # consume broker events
make worker                                                   # same as above
cowatcher-metrics-lite < ask.log   # summarize JSON ask logs
```

## Tests

```bash
pytest tests/ -v
```

## Stack

Python 3.11–3.13 · FastAPI · PySceneDetect · FFmpeg · faster-whisper · InsightFace · LiteLLM · BGE-M3 · Qdrant · PostgreSQL · Redis · MinIO
