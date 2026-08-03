# Gemini / Vertex context caching investigation (pilot cost reduction)

## Goal

Reuse the **merged system prompt** (and any fixed schema) across many `/ask`
and `/ask/stream` calls so Gemini bills fewer **input** tokens for the static
prefix of the prompt.

## Current integration

- Conversation completions go through **LiteLLM** (`LiteLLMCompletionClient`).
- Merged path messages:
  1. System: `MERGED_SYSTEM_PROMPT` (fixed)
  2. User: playhead + question + compact scene JSON (varies every call)
- Tools are **not** in the merged call (`tools=None`); scene lookup runs
  server-side before the model sees evidence.

## LiteLLM + Gemini support (as of this pilot)

| Mechanism | Supported for our stack? | Notes |
|-----------|--------------------------|--------|
| OpenAI-style **prompt caching** (automatic prefix discount on some APIs) | **Uncertain / provider-dependent** | Gemini via LiteLLM does not expose a simple “cache this system prompt” flag that maps to OpenAI’s cache headers. Provider-specific fields are required. |
| Google **context caching** (`cachedContents` / Vertex `CachedContent`) | **Not wired** | Requires creating a cache resource with Gemini/Vertex SDK or REST, then passing `cached_content` (or Vertex equivalent) into generateContent. LiteLLM support is incomplete or model-prefix-specific and not stable across free/paid Gemini tiers we use in local pilot. |
| Explicit LiteLLM `caching=True` | **App-level response cache only** | LiteLLM can cache *responses* by request hash; that is not Google context caching and would skip Qdrant/playhead-correct answers if keyed poorly. **We must not enable it for spoiler-safe /ask.** |

## Blocking constraints for the pilot

1. **Playhead + evidence change every turn** — only the system prompt is a good
   cache candidate (a few hundred tokens). Dynamic scene JSON cannot be cached
   at the provider without becoming spoiler-unsafe.
2. **Model churn** — pilot already rotates Gemini ids (`…flash-lite`,
   retired 2.x, thought signatures). Context-cache objects are **model-bound**
   and must be recreated on every model change; operationally heavy for a
   short pilot window.
3. **Free / tier APIs** — Google’s explicit context cache APIs often require
   paid Vertex or specific Gemini API features (minimum token thresholds for the
   cached body, TTL billing). Local pilot may not meet minimum size for a
   short MERGED prompt alone.
4. **LiteLLM abstraction** — forcing `cached_content` would fork our
   completion path and break **MOCK_MODE** unless carefully mocked.

## Decision (v0.3 cost pass)

**Do not force-fit context caching in this pass.**

Instead, cost control uses:

- Enforced **merged + PILOT_LOW_LATENCY** path (1 retrieve + 1 LLM)
- **Evidence trim** (`EVIDENCE_MAX_SCENES`, `EVIDENCE_MAX_CHARS_PER_FIELD`)
- **Two-tier QA cache** (+ `cowatcher-warm-qa-cache` prewarm)
- Per-call **token/cost logs + Prometheus** counters

## When to revisit

Revisit when:

1. LiteLLM documents a stable Gemini `cached_content` / Vertex cache param for
   the exact model string we pin, and
2. Production is on a **paid** Gemini or Vertex project with cache APIs enabled, and
3. Measured sample of ≥100 `/ask/stream` calls shows system-prefix fraction of
   input tokens large enough to justify cache create/TTL ops (e.g. system prompt
   grows with tools/schemas again).

## Measurement recipe (later)

1. Log `prompt_tokens` on 100 warm cache-miss asks **without** provider cache.
2. Create a Vertex/Gemini cache containing only `MERGED_SYSTEM_PROMPT`.
3. Pass cache id via LiteLLM if support lands; re-run 100 asks.
4. Compare mean `prompt_tokens` billed and provider “cached tokens” if reported.

Until then, treat this file as the **blocker record** for task 4 of the cost
reduction prompt.
