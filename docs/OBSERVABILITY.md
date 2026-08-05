# AI Co-watcher pilot — Prometheus alert thresholds

Documented targets for Grafana alerts or future paging. Not wired to a pager in the pilot.

## Real-time `/ask` health

| Signal | PromQL (5m window) | Warning | Critical | Notes |
|--------|-------------------|---------|----------|-------|
| p95 latency | `histogram_quantile(0.95, sum(rate(cowatcher_ask_request_duration_seconds_bucket[5m])) by (le))` | **> 3s** | **> 8s** | Pilot target: most answers under 3s with fast tier |
| Error rate | `sum(rate(cowatcher_ask_requests_total{status="error"}[5m])) / sum(rate(cowatcher_ask_requests_total[5m]))` | **> 1%** | **> 5%** | 500s from unhandled exceptions |
| Escalated tier share | `sum(rate(cowatcher_ask_model_tier_total{tier="escalated"}[15m])) / sum(rate(cowatcher_ask_model_tier_total[15m]))` | **> 40%** | **> 60%** | High escalation may indicate cost drift or mis-tuned heuristics |
| Don't-know rate | `sum(rate(cowatcher_ask_dont_know_total[15m])) / sum(rate(cowatcher_ask_requests_total{status="success"}[15m]))` | **> 25%** | **> 40%** | May indicate retrieval regression, stale index, or bad embeddings |
| QA cache hit rate | `sum(rate(cowatcher_qa_cache_hit_total[5m])) / (sum(rate(cowatcher_qa_cache_hit_total[5m])) + sum(rate(cowatcher_qa_cache_miss_total[5m])))` | **< 15%** on demo traffic with cache on | **< 5%** when `QA_CACHE_ENABLED=true` and warm paths expected | First-class panel on **AI Co-watcher Pilot** dashboard. Split exact/semantic via `cowatcher_qa_cache_hit_total{source=…}` |
| QA cache misses | `sum(rate(cowatcher_qa_cache_miss_total[5m]))` | sustained climb after warm | — | Correlate with threshold / bucket knobs |

Confirm live hit rate only when `QA_CACHE_ENABLED=true`. With the default `false` (latency isolation), hit counters stay flat — that is expected, not a regression.

## Tool & storage latency

| Signal | PromQL | Warning | Critical |
|--------|--------|---------|----------|
| scene_lookup p95 | `histogram_quantile(0.95, sum(rate(cowatcher_tool_call_duration_seconds_bucket{tool="scene_lookup"}[5m])) by (le))` | **> 500ms** | **> 2s** |
| Qdrant search p95 | `histogram_quantile(0.95, sum(rate(cowatcher_storage_query_duration_seconds_bucket{backend="qdrant"}[5m])) by (le))` | **> 300ms** | **> 1s** |
| Neo4j lookup p95 | `histogram_quantile(0.95, sum(rate(cowatcher_storage_query_duration_seconds_bucket{backend="neo4j"}[5m])) by (le))` | **> 400ms** | **> 1.5s** |

## Offline ingestion pipeline

| Signal | PromQL | Warning | Critical | Notes |
|--------|--------|---------|----------|-------|
| Job failure rate | `sum(rate(cowatcher_ingest_jobs_total{status="failed"}[30m])) / sum(rate(cowatcher_ingest_jobs_total[30m]))` | **> 5%** | **> 15%** | Worker nacks and requeues on failure |
| Queue depth (growing) | `deriv(cowatcher_ingest_queue_depth[15m])` | **> 0.1/s sustained** | **> 0.5/s sustained** | Unbounded growth = worker can't keep up |
| Queue depth (absolute) | `cowatcher_ingest_queue_depth` | **> 10** | **> 50** | Tune per deployment size |
| Ingest duration p95 | `histogram_quantile(0.95, sum(rate(cowatcher_ingest_job_duration_seconds_bucket[1h])) by (le))` | **> 45m** (30m title) | **> 90m** | Depends on title length and vision API throttling |

## Kafka-specific (when `MESSAGE_BROKER=kafka`)

Use `cowatcher_ingest_queue_depth{broker="kafka"}` as consumer lag estimate. For production, prefer `kafka_exporter` or broker metrics alongside this gauge.

## Runbook hints

- **High don't-know rate:** check Qdrant collection health, re-ingest title, verify `MOCK_MODE=false` embedder matches index.
- **Growing queue:** scale ingest workers (`cowatcher-ingest-worker` replicas), check vision API 429s in logs.
- **High /ask latency:** inspect tool-call histograms; escalated tier dominates → review `LLM_ESCALATION_*` settings. For median/p95 chase, ensure `QA_CACHE_ENABLED=true` on the measured env when using cache, then check **QA cache hit rate** on the pilot Grafana board; re-tune `QA_CACHE_SEMANTIC_THRESHOLD` / `QA_CACHE_TS_BUCKET_SEC` (see [CACHE_AND_TOKEN_TRIM.md](./CACHE_AND_TOKEN_TRIM.md)).
- **Cache hit rate flat at 0 with cache “on”:** confirm `/health` → `latency_path.qa_cache_enabled`, persona-keyed keys deployed, and warm path (`cowatcher-warm-qa-cache`) covers demo questions.
