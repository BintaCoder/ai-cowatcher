# Prompt: Fix Persona-Induced Latency Regression — ai-cowatcher

## Context
You are working in the `ai-cowatcher` codebase (AI Co-watcher, pilot v0.3.0).
Companion personas (`witty_friend`, `easygoing_friend`, and an apparently
undocumented `calm_scout`) were recently added on top of the merged
gate+agent path (`PILOT_LOW_LATENCY=true`, `UTTERANCE_GATE_STRATEGY=merged`,
§4.1-4.2 of `E2E_ARCHITECTURE.md`).

A quality-review export (`Answer_quality_review__Persona___Gender___Q___A`)
shows a clear latency regression correlated with persona:

| Persona | Avg latency | Notes |
|---|---|---|
| (default, no persona) | ~1.5s | Matches §10 target — healthy baseline |
| witty_friend | ~3.4s | Elevated |
| easygoing_friend | ~4.2s | Elevated |
| calm_scout | ~15.3s | Matches the "legacy tool path" latency signature (§10) |

Additionally, canned social/filler replies ("Thanks, that helped," "Hey, how's
it going?," "You there?") are taking 1.3–18.2s across personas, when they
should resolve near-instantly via the free heuristic gate (§4.5,
`outcome=free`) regardless of which persona is active.

## Objective
Restore persona-active latency to the same ~1.5-6s band as the no-persona
baseline, without removing or diluting the personality feature.

## Tasks

1. **Audit and fix the `[SOCIAL]` / `[FILLER]` short-circuit path**
   - Confirm whether the utterance gate (`Gate`, heuristic layer) still
     resolves filler/social utterances **before** any LLM call when a
     persona is active. Instrument or log-trace a known filler line (e.g.
     "You there?") end-to-end for at least one persona and confirm it never
     reaches the Gemini call.
   - If it's currently reaching the LLM: the likely cause is that
     `canned_social_reply` is being used as a *prompt instruction* (ask the
     model to say something like this) rather than returned directly as a
     static string lookup. Fix so the canned reply is returned as-is from
     the active persona's trait sheet with **no model call** — this was the
     original spec (persona prompt, Task 2) and must be enforced.
   - Add a regression test: for each shipped persona, assert that a known
     filler/social utterance resolves with `outcome=free` and latency under
     an explicit threshold (e.g. <200ms, excluding network to the client).

2. **Audit `calm_scout` specifically for legacy-path fallthrough**
   - Confirm `calm_scout` has a properly registered trait sheet in the same
     `personas/` config location as `witty_friend` / `easygoing_friend`,
     with all required fields populated (§ persona trait sheet schema from
     the personality prompt).
   - Trace what happens when an unrecognized, partially-configured, or
     malformed `persona_id` is received: it must **not** silently fall back
     to the full multi-tool agent loop. Preferred behavior: fail fast with a
     clear error, or fall back to a fully-specified default persona — never
     a silent fallback to the expensive path.
   - Add a startup-time or request-time validation step that rejects/flags
     any `persona_id` whose trait sheet is missing required fields, rather
     than allowing partial configs to reach the request path.
   - Add a regression test asserting that every shipped persona (including
     `calm_scout`, if it's meant to ship) resolves via the merged pilot path,
     not the legacy tool loop — assert on `cowatcher_legacy_tool_path_total`
     staying at 0 across a test sweep of all personas.

3. **Segment existing metrics by persona**
   - Add `persona_id` as a label on the existing latency/routing metrics
     rather than introducing new ones:
     - `cowatcher_ask_stage_duration_seconds{stage=..., persona_id=...}`
     - `cowatcher_utterance_gate_total{outcome=..., persona_id=...}`
     - `cowatcher_legacy_tool_path_total{persona_id=...}`
   - This makes future persona-specific regressions visible in the existing
     dashboard without needing another CSV export + manual analysis.

4. **Re-run the quality-review harness after the fix**
   - Re-run whatever produced `Answer_quality_review__Persona___Gender___Q___A`
     across all shipped personas, including the filler/social question set
     (`q11`, `q12`, `q13` or equivalents) and at least one repeat scene/
     question pair per persona.
   - Confirm: (a) filler/social latency drops to near-zero across all
     personas, (b) all personas' content-question latency lands in the
     ~1.5-6s band, (c) no persona shows a legacy-path latency signature
     (10s+).

5. **Investigate the 100% cache-miss rate seen in the export**
   - Confirm whether the quality-review harness intentionally bypasses the
     QA cache (common for eval/quality tooling, to get fresh answers rather
     than cached ones). If intentional, document this explicitly in the
     harness's README/comments so it isn't mistaken for a bug again.
   - If NOT intentional: verify the QA cache key change from the companion
     personality prompt (Task 4 — `persona_id` added to both the Redis exact
     key and the Qdrant semantic filter) was actually implemented and is
     being hit on genuine repeats. Add a test for a same-persona,
     same-question, same-ts-bucket repeat asserting a cache hit.

6. **Investigate the two fully-errored runs**
   - The export includes two runs (all `status=error`, `latency=-1`) that
     predate the working runs chronologically. Confirm these are resolved
     (e.g. tied to a deploy that's since been fixed) rather than an
     intermittent failure mode that could recur. If the root cause is
     unknown, add error logging/alerting so a full-request failure like this
     surfaces immediately rather than being discovered later via a CSV
     export.

## Constraints
- Do not remove or weaken the personality feature to fix latency — the fix
  must preserve tone differentiation between personas (verify against the
  eval set from the companion personality prompt, Task 6).
- Do not add a second LLM call anywhere in the pilot path to "fix" this —
  the entire point of the merged path is one call; the bug is very likely a
  logic/routing issue, not something that needs new architecture.
- Keep changes compatible with `MOCK_MODE`.

## Acceptance criteria
- Filler/social questions resolve with `outcome=free` and near-zero latency
  for every shipped persona, verified by the regression test in Task 1.
- No persona shows legacy-tool-path latency (10s+) in a full re-run of the
  quality-review harness.
- `cowatcher_legacy_tool_path_total{persona_id=...}` stays at 0 across a
  sweep of all shipped personas.
- Root cause of the `calm_scout` regression is identified and documented
  (not just patched around), and either fixed or the persona is pulled from
  the shipped set until it is.
- Cache-miss behavior in the eval harness is confirmed intentional and
  documented, or fixed if it was an unintended regression.
