# Cache hit rate + token/evidence trim (prompt 6)

Pilot: latency pass targeting **~800ms average** `/ask/stream` by raising QA
cache hits and shrinking the Gemini call floor. Landed **before** GCP
migration (`prompt_3_gcp_migration.md`) so migration can use these numbers
as its “before” baseline.

## Knobs (testing)

| Knob | Default | How to flip |
|------|---------|-------------|
| `QA_CACHE_ENABLED` | `false` | `make api QA_CACHE_ENABLED=true\|false` — do **not** break this switch |
| `QA_CACHE_SEMANTIC_THRESHOLD` | `0.88` | env / `.env` |
| `QA_CACHE_TS_BUCKET_SEC` | `45` | env (was 30; widen 45–60 for longer scene beats) |
| `EVIDENCE_MAX_SCENES` | `2` | env (was 3) |
| `EVIDENCE_MAX_CHARS_PER_FIELD` | `200` | env (was 280) |
| `LLM_MAX_TOKENS` | `512` | env — keep ≥512 for Gemini 3 reasoning headroom |
| `LLM_SHORT_ANSWER_MAX_TOKENS` | `160` | env — short who/what/where lines |
| `LLM_REASONING_EFFORT` | `minimal` | env — **global** via `LiteLLMCompletionClient`, not per persona |

`/health` → `latency_path` and `llm` expose the live values for verification.

## Task checklist

### A. Cache hit rate

1. **Confirm cache live on measured traffic**  
   - Measured env must have `QA_CACHE_ENABLED=true` (`/health.latency_path.qa_cache_enabled`).  
   - Persona isolation: exact key and Qdrant semantic filter both include `persona_id`
     (`ai_cowatcher/storage/qa_cache.py`). Covered by `tests/test_qa_cache.py`.  
   - Baseline PromQL (real traffic, not eval-only if possible):

```promql
# hit rate
sum(rate(cowatcher_qa_cache_hit_total[1h]))
  / (sum(rate(cowatcher_qa_cache_hit_total[1h]))
     + sum(rate(cowatcher_qa_cache_miss_total[1h])))

# by source
sum(rate(cowatcher_qa_cache_hit_total[1h])) by (source)
sum(rate(cowatcher_qa_cache_miss_total[1h]))
```

Record the values **before** and **after** knob changes on the same traffic mix.

**Logged pre-change baseline (dev bench, not live prod):** from
`benchmarks/results/26b168323136.jsonl` with cache disabled path dominant —
21/30 asks were `cache_source=miss`, miss mean ≈ **1334ms**, overall mean ≈ **936ms**.
Re-measure on real traffic with `QA_CACHE_ENABLED=true` after warm.

2. **Semantic threshold**  
   - Default moved **0.90 → 0.88** (one 0.02 step) after labeled-pair methodology
     in `benchmarks/qa_cache_threshold_pairs.json`.  
   - Tune / re-validate:

```bash
PYTHONPATH=. python scripts/tune_qa_cache_threshold.py --mock
# Prefer real BGE before shipping a further change:
PYTHONPATH=. MOCK_MODE=false python scripts/tune_qa_cache_threshold.py
```

   Gate: near-dup hit rate ≥ 0.8 and **0 false positives** on different pairs
   at the chosen threshold.

3. **`ts_bucket` width**  
   - Fully configurable via `QA_CACHE_TS_BUCKET_SEC` (always was).  
   - Default **30 → 45s** for sitcom/pilot scene lengths commonly >30s.  
   - If a title’s scenes rarely span beyond ~30s, set `30` per env.

4. **Pre-warm common patterns**  
   - `cowatcher-warm-qa-cache` / `ai_cowatcher.qa.warm_cache` now seeds common
     real phrasing (`what's going on`, `who is that`, `why is X upset`, …) in
     addition to scripted demo lines, **per persona**.  
   - Optionally seed from quality-review questions:

```bash
QA_CACHE_ENABLED=true cowatcher-warm-qa-cache \
  --question-set benchmarks/friends_ross_questions.json \
  --title-id friends_ross
```

5. **Hit-rate monitoring**  
   - Counters: `cowatcher_qa_cache_hit_total{source}`, `cowatcher_qa_cache_miss_total`.  
   - Primary Grafana board **AI Co-watcher Pilot** panels: **QA cache hit rate**,
     **QA cache hits vs misses**. Also mirrored on Ask Bench.

### B. Token / evidence trim

6. **Token caps**  
   - Short-answer cap **256 → 160**; style_notes on all three personas insist on
     **one short sentence**.  
   - Global max tokens: **512** code default (`.env.example` was 768 → 512).  
   - Re-measure `llm_total` with:  
     `make bench-ask` (or quality-review harness) ≥30 content questions / persona.

7. **`LLM_REASONING_EFFORT=minimal`**  
   - Applied in `LiteLLMCompletionClient._common_kwargs` for **every** completion
     path; no persona override exists. Tests lock this behavior.

8. **Evidence trim**  
   - Defaults **3/280 → 2/200**. Spoiler filter (`start_ts <= current_ts`) unchanged.  
   - Candidate quant:

```bash
PYTHONPATH=. python scripts/bench_evidence_cost.py --candidates 3:280,2:200,2:160
```

9. **Model fresh-gen floor**  

```bash
MOCK_MODE=false PYTHONPATH=. python scripts/measure_llm_floor.py --runs 5
```

#### Floor statement (required expectation)

**Measured (local, `gemini/gemini-3.5-flash-lite`, minimal prompt, 2026-08-04 via
`scripts/measure_llm_floor.py`):** mean ≈ **1859ms**, p50 ≈ **1859ms** on a
2-run sample (MOCK_MODE=false). Re-run with `--runs 5+` for a stable figure on
your network.

| Observation | Implication for **~800ms average** |
|-------------|-------------------------------------|
| Minimal-prompt Gemini floor ~**1.7–2.0s** in this environment | Already **above** 800ms before retrieve/gate |
| Bench cache-miss mean ≈ **0.9–1.5s** on thicker merged prompts (varies by day) | Miss path alone typically **exceeds** 800ms |
| Cache exact/free-gate paths are ≈ **low ms–tens of ms** | They pull the **blend** down |

**Conclusion:** the **800ms target is a blended average** across cache hits
(exact/semantic), free-gate social/filler, and cache-miss Gemini calls — **not** a
per-request guarantee on every fresh answer. Infra colocation (GCP migration)
tightens tail variance but does not erase a multi-hundred-ms to multi-second
fresh-gen floor on this model class.

10. **Quality-review re-run**  

```bash
# Cache off — isolate trim / token effects
make api QA_CACHE_ENABLED=false
make bench-ask   # or cowatcher-bench-ask with persona matrix

# Cache on — hit-rate + blended latency
make api QA_CACHE_ENABLED=true
QA_CACHE_ENABLED=true cowatcher-warm-qa-cache --title-id friends_ross
make bench-ask
```

Check: (a) latency direction, (b) persona tone on `benchmarks/persona_eval.json`,
(c) spoiler-safe answers on plot rows.

## Hand-off baseline for GCP migration

| Metric | Value / location |
|--------|------------------|
| Pre-trim miss mean (bench sample) | ~1334ms (`26b168323136` misses) |
| Fresh LLM floor (minimal prompt) | ~1859ms mean (`measure_llm_floor.py`, flash-lite, local) |
| Evidence token candidates | 3×280 ≈953 tok → 2×200 ≈707 tok offline est. (`bench_evidence_cost.py`) |
| Post-trim defaults | threshold 0.88, bucket 45s, evidence 2×200, short tokens 160 |
| How to re-measure | `make bench-ask` + Grafana **QA cache hit rate** + `/ask` p50/p95 |
| Floor statement | Blended average only — see §9 above |

After GCP migrate, compare against **this** baseline rather than pre-prompt-6.
