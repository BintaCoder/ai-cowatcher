# ai-cowatcher

Pay-TV co-watcher pilot — an AI companion that answers viewer questions about what they're watching, without spoilers.

## Architecture

- **Offline ingestion** (once per title): scene detection, transcription, face clustering, vision captioning, and vector indexing into Postgres + Qdrant.
- **Real-time Q&A** (per question): a single orchestrating conversation agent with `scene_lookup` tool-calling; spoiler safety enforced via `end_ts <= current_ts` at retrieval time only.
- **Tiered LLM routing**: fast model by default, escalated model for nuanced questions (config-driven).
- **Pilot observability**: structured JSON logs per `/ask` and `GET /metrics-lite` for latency, escalation rate, and "don't know" rate.

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
| `POST /ingest` | Queue offline title ingestion |
| `POST /ask` | Real-time co-watcher Q&A |
| `GET /metrics-lite` | Pilot KPIs (latency, escalation, don't-know rate) |

## CLI

```bash
cowatcher-ingest --title-id demo --video /path/to/video.mp4
cowatcher-metrics-lite < ask.log   # summarize JSON ask logs
```

## Tests

```bash
pytest tests/ -v
```

## Stack

Python 3.11–3.13 · FastAPI · PySceneDetect · FFmpeg · faster-whisper · InsightFace · LiteLLM · BGE-M3 · Qdrant · PostgreSQL · Redis · MinIO
