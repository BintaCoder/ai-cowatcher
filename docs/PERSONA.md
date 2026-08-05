# Companion personality & gender voice

The co-watcher can sound like different couch friends without changing plot grounding.

## Personas (data)

JSON packs live under `ai_cowatcher/personas/`:

| `persona_id` | Display name | Vibe |
|---|---|---|
| `easygoing_friend` | Easygoing Friend | Warm, relaxed default |
| `witty_friend` | Witty Friend | Quippy, snappy |
| `calm_scout` | Calm Scout | Quiet observational |

Fields: `persona_id`, `display_name`, traits (`humor` / `formality` / `warmth` / `verbosity`), `style_notes`, `avoid`, `canned_social_reply`, optional `tts.rate` / `tts.pitch`.

Default: `DEFAULT_PERSONA_ID=easygoing_friend` (see `.env.example`).

## API

`POST /ask` and `POST /ask/stream` accept optional:

```json
{
  "title_id": "friends_ross",
  "current_ts": 45.0,
  "question": "Who is that?",
  "user_id": "web-viewer",
  "persona_id": "witty_friend",
  "companion_gender": "female"
}
```

- `persona_id` — tone pack used in `MERGED_SYSTEM_PROMPT` / conversation prompts and QA cache keys.
- `companion_gender` — `male` | `female` | `neutral`. Added as a one-line tone note (delivery voice of a friend). TTS gender matching is primarily client-side.

## QA cache isolation

Exact Redis keys and Qdrant semantic payloads/filters include `persona_id`. Same question + playhead + different persona **must not** cross-hit.

Warm demos seed every persona by default:

```bash
cowatcher-warm-qa-cache --title-id friends_ross
cowatcher-warm-qa-cache --persona-id witty_friend --title-id friends_ross
```

Cold first-pass benches often show 100% **miss** — that means a cold cache / new persona key space, not a broken path. Double-run the same seed to confirm hits. See `docs/BENCH_ASK.md`.

## SOCIAL / FILLER free path

Presence checks and fillers (`You there?`, `Hey, how's it going?`, `Thanks, that helped`, `Hmm okay`, `Wait—`) resolve on the **utterance gate** before Gemini for every persona. The companion’s `canned_social_reply` is returned as a static string (`model_name=gate:free`). They are not re-prompted through the LLM.

Invalid `persona_id` falls back to `DEFAULT_PERSONA_ID` on the same merged pilot path — never the multi-tool legacy agent.
## Watch UI

Open `/watch`:

1. **Companion** dropdown — picks `persona_id` (session `localStorage` key `cowatcher.persona_id`).
2. **Voice** dropdown — male / female / neutral (`cowatcher.companion_gender`).
3. Both are sent on `/ask/stream`. Browser `speechSynthesis` picks a matching voice when available and applies persona rate/pitch.

## Manual tone eval

~14 prompt rows with persona-specific tone notes:

`benchmarks/persona_eval.json`

Run the real or mock stack and score answers 1–5 for tone fit (facts/spoiler rules still apply). Example:

```bash
# Mock API, then curl each question with each persona_id
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "title_id":"friends_ross","current_ts":45,"question":"Who is that?",
  "user_id":"eval","persona_id":"witty_friend","companion_gender":"male"
}'
```

## Tests

```bash
pytest tests/test_persona.py tests/test_persona_latency_gate.py tests/test_utterance_gate.py tests/test_qa_cache.py -q
```
