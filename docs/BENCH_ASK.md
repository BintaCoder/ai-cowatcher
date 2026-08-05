# Ask Bench

Automated `/ask` latency + QA-cache sampling for pilot titles (default title name:
**Friends Ross**, resolved to ingest id `friends_ross`).
Answers land in **Postgres** for Grafana tables; bonus JSONL under `benchmarks/results/`.

## Persistence (long-term)

Rows in `bench_ask_result` have **no app TTL** — they stay until the Postgres volume
is deleted. Grafana used to hide anything older than **6 hours**; panels now default
to the **last 30 days**, plus a **Run history** table so older `run_id`s stay visible.

| Store | Durability | Notes |
|-------|------------|-------|
| Postgres `bench_ask_result` | Named volume `postgres_data` | Primary; Grafana “Ask Bench” |
| `benchmarks/results/<run_id>.jsonl` | On disk / git | Archive + reimport source |
| Prometheus `/ask` series | Named volume `prometheus_data` (30d retention) | Latency charts only — not Q&A text |

**Do not** run `docker compose down -v` unless you intend to wipe history.
`make down` keeps volumes.

### Restore yesterday’s runs into Grafana

JSONL from prior benches lives under `benchmarks/results/`. After Postgres is up:

```bash
make up-core
make bench-reimport
# or one file: make bench-reimport FILE=benchmarks/results/26b168323136.jsonl
docker compose restart grafana   # pick up dashboard JSON changes
```

Reimport is idempotent (skips `run_id`s already in Postgres). Use `FORCE=1` only if you
want duplicate inserts.

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
   # .env: MOCK_MODE=false, GEMINI_API_KEY=...
   make api                              # QA cache off by default
   # make api QA_CACHE_ENABLED=true      # for cache benches
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
| `--persona-id` | `DEFAULT_PERSONA_ID` (easygoing_friend) | Companion tone (QA-cache key includes this) |
| `--companion-gender` | `neutral` | `male` \| `female` \| `neutral` (delivery / TTS hint) |
| `--all-personas` | off | Replay the same n samples for every shipped persona |
| `--questions-file` | `benchmarks/friends_ross_questions.json` | Question bank |
| `--duration-sec` | auto | Override if ffprobe/DB unavailable |
| `--allow-mock` | off | Permit `MOCK_MODE=true` |

```bash
# Witty + female voice preference
make bench-ask PERSONA=witty_friend GENDER=female SEED=42

# Tone comparison: same 3 Qs × all personas (watch Gemini spend)
make bench-ask ALL_PERSONAS=1 N=3 SEED=7
```

Each run:

1. Loads settings; exits if mock without `--allow-mock`
2. Resolves video duration (ffprobe on catalog video → max scene `end_ts` → `--duration-sec`)
3. Picks n random questions × random playhead in `[0, duration]`
4. `POST /ask` with live `current_ts`, **`persona_id`**, **`companion_gender`**
5. Derives `cache_source` from response `model_name` / tier
6. Inserts Postgres + appends JSONL (both store persona/gender); prints a summary table

## Cache source derivation

Server cache hits stamp:

- `model_name = "qa_cache:exact"` or `"qa_cache:semantic"`
- `model_tier = "cache"`
- `escalation_reason = "cache:exact|semantic"`

The runner parses that into `cache_source`. Other stamps:

| Response | `cache_source` |
|----------|----------------|
| real Gemini model string after agent answer | **`miss`** (lookup ran, agent answered) |
| free-gate reply (`model_name=gate:free`, `model_tier=gate`) | **`free`** (no LLM / no QA store) |
| empty / error | **`none`** |

### Why first-pass exports often show ~100% miss

The ask bench **does not disable the QA cache**. On a cold cache (or a fresh
`persona_id` key space after personality was added), every content sample is a
genuine **miss**. Social/filler lines now stamp **`free`** (heuristic canned reply).

To measure exact hits:

```bash
# same seed + persona twice — second pass should show exact on content rows
make bench-ask PERSONA=witty_friend SEED=42 N=5
make bench-ask PERSONA=witty_friend SEED=42 N=5
```

Persona is part of the exact Redis key and semantic filter, so repeats under a
*different* companion will still miss. That is intentional isolation, not a bug.

## Quality regression after persona changes

```bash
# include social/filler (bank q11–q15) and all personas
make bench-ask ALL_PERSONAS=1 N=10 SEED=7
```

Expect free-path socials near-instant; content in the pilot band; no 10s+ legacy
signatures when `PILOT_LOW_LATENCY=true`.

## Grafana

| Item | Location |
|------|----------|
| Dashboard | **Ask Bench** (`uid`: `cowatcher-ask-bench`) under folder **AI Co-watcher** |
| JSON | `observability/grafana/dashboards/cowatcher-ask-bench.json` |
| Postgres DS | `observability/grafana/provisioning/datasources/postgres.yml` → `postgres:5432`, db/user/password `cowatcher` (matches `docker-compose.yml`) |
| Prometheus DS | existing Prometheus provision |

**Panels (quality-first):**

1. **Answer quality review (Persona · Gender · Q · A)** — full answer text + Persona + Gender for the last **30 days**  
2. **Latest run — compare by Persona / Gender** — all rows for the newest `run_id`  
3. **Run history (last 30d)** — every `run_id` with counts / avg latency / cache mix  
4. **Rows by Persona × Gender** — counts and avg latency  

Default dashboard time range is **now-30d**. Postgres SQL panels use a 30-day filter
independent of the picker for consistency.

After changing provisioning or the dashboard JSON, reload Grafana:

```bash
docker compose restart grafana
# hard-refresh browser (cmd+shift+R) → AI Co-watcher → Ask Bench
```

Ensure you re-run the bench **after** API restart (so `persona_id` / `companion_gender` columns exist and are filled). Older rows may show Persona as `(default)` / Gender as `(unset)`.

### Manual SQL check

```sql
SELECT question_id, persona_id, companion_gender, current_ts, latency_ms, cache_source, LEFT(answer, 80)
FROM bench_ask_result
ORDER BY created_at DESC
LIMIT 20;
```

## JSONL bonus

`benchmarks/results/<run_id>.jsonl` — one JSON object per sample. Keep these as the
offline archive (safe to commit for pilot regressions). Rehydrate Grafana with
`make bench-reimport` after a volume wipe or new machine.

## Tests

Unit tests only (no live Gemini):

```bash
.venv/bin/pytest tests/test_bench_ask.py -q
```
