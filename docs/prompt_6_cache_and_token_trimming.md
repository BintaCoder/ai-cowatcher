# Prompt: Improve Cache Hit Rate + Trim Tokens/Evidence — ai-cowatcher

## Context
You are working in the `ai-cowatcher` codebase (AI Co-watcher, pilot v0.3.0).
Post persona-latency-fix, real-world `/ask/stream` latency measures ~300-3000ms,
averaging ~1500ms. Target is **~800ms average**. Stage instrumentation (§10 of
`E2E_ARCHITECTURE.md`) shows `cache_lookup`, `gate`, and `scene_retrieve` are
each low-cost when working correctly — the dominant cost is the Gemini call
itself (`llm_ttft` + `llm_total`).

**Sequencing note:** this work should land and be measured **before** the GCP
migration (`prompt_3_gcp_migration.md`). Infra migration mainly tightens
variance/tail latency; it will not close a ~700ms median gap on its own.
This prompt targets the median directly. Do the GCP migration after this,
using the post-trim numbers as the new baseline to compare against.

## Objective
Reduce average `/ask/stream` latency from ~1500ms toward ~800ms primarily by
(a) raising the QA cache hit rate on real traffic, and (b) shrinking the
Gemini call's floor via token and evidence trimming — without regressing
answer quality, persona tone, or spoiler safety.

## Tasks

### A. Cache hit rate

1. **Confirm the cache is actually live on production/real traffic**
   - Verify `QA_CACHE_ENABLED=true` in the environment being measured.
   - Confirm the persona-keyed cache fix (from the persona regression prompt,
     Task 5) is deployed — both the Redis exact key and the Qdrant semantic
     filter must include `persona_id`.
   - Pull current real-traffic (not eval-harness) hit rate from
     `cowatcher_qa_cache_hit_total{source=exact|semantic}` vs
     `cowatcher_qa_cache_miss_total`. This is your baseline — record it before
     making changes.

2. **Tune the semantic similarity threshold empirically**
   - Current default is `0.90` (§12). Test whether lowering it slightly
     (e.g. in `0.02` steps) increases hit rate without pulling in
     semantically-wrong cached answers.
   - Validate using a labeled set of near-duplicate question pairs (same
     intent, different phrasing) vs. genuinely-different question pairs —
     confirm the threshold change doesn't cause false-positive hits on the
     latter set before shipping.

3. **Widen or make `ts_bucket` granularity configurable**
   - Current playhead bucket is 30s (per the QA cache design). If scenes in
     typical content run longer than 30s without changing context, a wider
     bucket (e.g. 45-60s) would increase legitimate hit rate without
     serving stale-context answers. Test against real scene-length
     distribution in ingested titles before changing the default.

4. **Pre-warm expected Q&A pairs**
   - Extend `cowatcher-warm-qa-cache` / `warm_cache.py` to cover the most
     common real question patterns (not just scripted demo lines) — e.g.
     "what's going on," "who is that," "why is X upset" — per persona, for
     actively-demoed titles.
   - Identify common question patterns from real usage logs (if available)
     rather than guessing; if no real usage data exists yet, seed from the
     existing quality-review question set as a starting point.

5. **Add hit-rate monitoring as a first-class metric**
   - Ensure `cowatcher_qa_cache_hit_total` / `_miss_total`, split
     exact/semantic, are on the primary dashboard (not just available via
     query) so hit-rate regressions are visible without pulling a CSV again.

### B. Token / evidence trimming (shrinks the LLM-call floor itself)

6. **Reduce `LLM_MAX_TOKENS` / `LLM_SHORT_ANSWER_MAX_TOKENS`**
   - Audit current values against actual answer length needed — pilot
     answers are meant to be ~1 short sentence (`verbosity: short` in
     persona trait sheets). If any persona is producing longer answers than
     necessary, tighten its `style_notes` and the global short-answer cap.
   - Measure `llm_total` before/after on a fixed sample of ≥30 real
     questions per persona.

7. **Confirm `LLM_REASONING_EFFORT=minimal` is honored for every persona**
   - After the persona work, verify this setting still applies universally
     and wasn't inadvertently overridden per-persona. Reasoning-token
     overhead eats directly into the latency budget before any answer
     tokens are generated.

8. **Tighten evidence trim further**
   - Re-evaluate `EVIDENCE_MAX_SCENES` (default 3) and
     `EVIDENCE_MAX_CHARS_PER_FIELD` (default 280) — test whether reducing
     either (e.g. 2 scenes, 200 chars/field) meaningfully lowers `llm_ttft`
     without degrading answer correctness on the quality-review set.
   - Use `scripts/bench_evidence_cost.py` to quantify the token delta for
     each candidate setting before picking a new default.

9. **Check for headroom vs. the model's floor**
   - Run a minimal-prompt, minimal-evidence synthetic test against the
     current fast-tier model (`gemini/gemini-3.5-flash-lite` or current
     equivalent) to establish the approximate lower bound of `llm_total` for
     this model/config combination.
   - If the measured floor is already close to or above ~800ms for a fresh
     (non-cached) generation, document this explicitly — it means 800ms
     average is only achievable via a blended cache-hit/cache-miss traffic
     mix, not as a per-request guarantee on every fresh answer. Set
     expectations accordingly rather than continuing to chase an
     unreachable per-request floor.

10. **Re-run the quality-review harness after each change**
    - After each meaningful change (threshold tune, bucket width, token
      caps, evidence trim), re-run the harness across all shipped personas
      and confirm: (a) latency moved in the intended direction, (b) answer
      quality/persona tone did not regress (spot-check against the eval set
      from the personality prompt).

## Constraints
- Do not sacrifice spoiler safety (§14) or grounding for latency — evidence
  trimming must stay above the threshold where the model has enough context
  to answer correctly and refuse/fallback appropriately when it doesn't.
- Do not weaken persona tone differentiation to hit the latency number —
  verify against the persona eval set at each step.
- Do not change the merged single-call architecture — this is tuning within
  the existing shape, not a redesign.
- Keep all changes compatible with `MOCK_MODE`.

## Acceptance criteria
- Documented before/after cache hit rate (real traffic, not eval-harness
  traffic) showing measurable improvement.
- Documented before/after average and p95 `/ask/stream` latency across a
  fixed sample of ≥100 real-pattern questions, spanning all shipped
  personas and both cache-hit and cache-miss cases.
- A clear, written statement of the model's realistic fresh-generation
  latency floor, so the 800ms target is understood as either achievable
  per-request or only achievable as a blended average — not left ambiguous.
- No regression in answer quality or persona tone versus the existing eval
  set.
- Final post-trim baseline numbers recorded and handed off as the "before"
  measurement for the subsequent GCP migration (`prompt_3_gcp_migration.md`).
