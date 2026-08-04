# Ask Bench

Automated `/ask` latency + QA-cache sampling for pilot titles (default title name:
**Friends Ross**, resolved to ingest id `friends_ross`).
Answers land in **Postgres** for Grafana tables; bonus JSONL under `benchmarks/results/`.

## Product rules

1. **Real Gemini** — run with `MOCK_MODE=false`. The runner refuses mock mode unless you pass `--allow-mock` (smoke only).
2. **QA cache stays ON** — do not bypass. Each row records `cache_source`: `exact` | `semantic` | `miss` | `none`.
3. **Grafana-first** — primary results table is `bench_ask_result` (not JSONL-only).

## Prerequisites

1. Ingest the title (example):

   ```bash
   make ingest TITLE=friends_ross VIDEO=./friends_ross_has_problems.mp4
   ```

2. Stack + API with real models:

   ```bash
   make up-core   # postgres redis qdrant neo4j minio prometheus grafana
   # .env: MOCK_MODE=false, GEMINI_API_KEY=..., QA_CACHE_ENABLED=true (default)
   make api
   ```

3. Grafana: http://localhost:3000 (admin / cowatcher)

## Run

```bash
# after make install + editable install
cowatcher-bench-ask
# or: cowatcher-bench-ask --title-id "Friends Ross"
# still works: cowatcher-bench-ask --title-id friends_ross

# or without entrypoint
make bench-ask
# make bench-ask TITLE="Friends Ross" SEED=42
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--title-id` | `Friends Ross` | Display name **or** ingest id (resolved via Postgres) |
| `--base-url` | `http://localhost:8000` | API |
| `--n` | `5` | Sample size (from bank of 20) |
| `--seed` | (random) | Reproducible sample |
| `--questions-file` | `benchmarks/friends_ross_questions.json` | Question bank |
| `--duration-sec` | auto | Override if ffprobe/DB unavailable |
| `--allow-mock` | off | Permit `MOCK_MODE=true` |

Each run:

1. Loads settings; exits if mock without `--allow-mock`
2. Resolves video duration (ffprobe on catalog video → max scene `end_ts` → `--duration-sec`)
3. Picks 5 random questions × random playhead in `[0, duration]`
4. `POST /ask` with live `current_ts`
5. Derives `cache_source` from response `model_name` / tier
6. Inserts Postgres + appends JSONL; prints a summary table

## Cache source derivation

Server cache hits stamp:

- `model_name = "qa_cache:exact"` or `"qa_cache:semantic"`
- `model_tier = "cache"`
- `escalation_reason = "cache:exact|semantic"`

The runner parses that into `cache_source`. Any other non-empty `model_name` (real Gemini model string) is recorded as **`miss`** (lookup ran, agent answered). Empty fields → **`none`** (errors).

## Grafana

| Item | Location |
|------|----------|
| Dashboard | **Ask Bench** (`uid`: `cowatcher-ask-bench`) under folder **AI Co-watcher** |
| JSON | `observability/grafana/dashboards/cowatcher-ask-bench.json` |
| Postgres DS | `observability/grafana/provisioning/datasources/postgres.yml` → `postgres:5432`, db/user/password `cowatcher` (matches `docker-compose.yml`) |
| Prometheus DS | existing Prometheus provision |

Panels:

- Latest hour + latest `run_id` answer tables (Question / Playhead / Latency / Cache / Answer)
- Cache mix + latency by `cache_source` from Postgres
- API-side Prom: `/ask` p95, QA cache lookup rates, stage histograms during the bench window

After changing provisioning, restart Grafana (`docker compose restart grafana`).

### Manual SQL check

```sql
SELECT question_id, current_ts, latency_ms, cache_source, LEFT(answer, 80)
FROM bench_ask_result
ORDER BY created_at DESC
LIMIT 20;
```

## JSONL bonus

`benchmarks/results/<run_id>.jsonl` — one JSON object per sample. Safe to gitignore; not required for Grafana.

## Tests

Unit tests only (no live Gemini):

```bash
.venv/bin/pytest tests/test_bench_ask.py -q
```
