# Prediction Mode + Proactive Trivia

Pilot features for speculative guesses and sparse, opt-in production trivia.

## Prediction Mode

- Intent tag `[PREDICTION]` on the merged gate+agent path (single LLM call).
- Ack is persona-flavored and must not confirm/deny plot.
- Rows live in `predictions`; `reveal_ts` comes from `title_events` (`reveal_point` / `credits`) or end-of-title fallback.
- **Hard rule:** `PredictionStore.try_resolve` raises `PredictionTooEarlyError` before `reveal_ts` (server-side).
- Client: `/predictions/list`, `/predictions/pending` (polled from `/watch`); “Your guesses” log.

Env: `PREDICTION_MODE_ENABLED` (default true).

## Proactive Trivia

- **Opt-in**, off by default (`cowatcher.trivia_opt_in` in localStorage).
- Facts precomputed at ingest (`scene_trivia`) with a spoiler keyword filter; rejected rows are logged.
- `/trivia/tick` is a cheap playhead check — **no Gemini**. Persona flavoring is template-based.
- Pacing: `TRIVIA_MIN_GAP_SEC` (default 720), `TRIVIA_MAX_PER_SESSION` (default 4), ask cooldown.

Hard gate: `enabled: false` → `{trivia: null, reason: "opt_out"}`.
