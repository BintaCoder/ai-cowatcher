# Prompt: Conversation-Aware Audio Ducking (VAD-Driven, No Stream Restarts) — ai-cowatcher

## Context
You are working in the `ai-cowatcher` codebase (AI Co-watcher, pilot v0.3.0),
specifically the `/watch` client (browser UI: video playback, Web Speech API
mic capture, `SpeechSynthesis` TTS output, existing volume-duck logic per
§2 of `E2E_ARCHITECTURE.md`).

**Prior bug, now fixed by disabling auto-listening:** the previous
"conversation aware" implementation appears to have stopped/restarted the
mic `MediaStream` (and/or the `AudioContext`) around TTS playback boundaries.
This caused audio to break specifically at word 2-3 of every TTS answer —
consistent with the transient cost of renegotiating echo cancellation and
rebuilding the audio graph on every stop/start cycle. Auto-listening is
currently disabled to work around this.

## Objective
Reimplement conversation-aware behavior as **continuous gain-based ducking**
(similar to AirPods Pro 2 Conversation Awareness) instead of stream
start/stop cycling. The mic stream and audio graph must **never be torn down
or restarted** during a session — only gain values change. This should
restore the "always listening" UX without reintroducing the audio-break bug.

## Tasks

1. **Establish a persistent audio session**
   - On `/watch` load (or on first user interaction, per browser autoplay/mic
     permission requirements), acquire the mic `MediaStream` via
     `getUserMedia` **once**, with `echoCancellation: true`, and keep it
     alive for the duration of the watch session.
   - Create a single `AudioContext` at session start and keep it alive/warm
     for the whole session (resume it once on user interaction if suspended
     by autoplay policy — do not repeatedly create/close it).
   - **Explicit constraint:** no code path in this feature may call
     `getUserMedia` again, stop the mic track, or close/recreate the
     `AudioContext` after session start, for any reason (including TTS
     start/stop). If you find yourself needing to do this to make ducking
     work, that's a sign the design has regressed toward the old bug —
     stop and reconsider.

2. **Route TTS output through a GainNode**
   - Instead of sending TTS/answer audio directly to the audio destination,
     route it through a dedicated `GainNode`:
     `TTS source → GainNode (tts) → AudioDestination`.
   - If using the native `SpeechSynthesis` API (which doesn't expose a Web
     Audio node directly), evaluate whether to (a) switch the answer voice
     path to an `<audio>`/`AudioBufferSourceNode`-based TTS output that can
     be routed through Web Audio, or (b) duck via
     `SpeechSynthesisUtterance.volume` as a fallback if switching isn't
     feasible in this iteration. Prefer (a) if achievable without large
     scope — it gives smoother, click-free ramps.

3. **Route video/program audio through its own GainNode**
   - Apply the same pattern to existing video playback audio (currently
     ducked via whatever ad hoc mechanism is in place per §2): 
     `Video audio → GainNode (program) → AudioDestination`.
   - This unifies TTS ducking and video ducking under one consistent
     mechanism, both driven by the same VAD signal (Task 4), rather than two
     different implementations.

4. **Implement lightweight VAD on the mic stream**
   - Add a continuous voice-activity detector on the always-on mic stream,
     using an `AnalyserNode` (RMS/energy thresholding) as a first pass, or a
     small VAD library if energy thresholding proves too noisy against
     background program audio bleed.
   - This VAD is **separate from and in addition to** the existing wake-word
     detection — its only job is "is the user producing speech-like sound
     right now," not "did they say the wake word."
   - Tune the energy threshold against real test audio (quiet room, TV
     playing in background, user talking over it) before finalizing — do
     not ship a threshold that hasn't been validated against program-audio
     bleed, since that's a known false-positive risk called out in the
     architecture doc's "ambient mic" design (§2).

5. **Wire VAD signal to gain ramps (the actual ducking behavior)**
   - On VAD speech-start: ramp both `program` and `tts` GainNodes down
     smoothly (e.g. `gain.linearRampToValueAtTime(0.15, ctx.currentTime +
     0.15)` — duck to ~15%, not to zero; a full mute breaks the
     "conversation" feel).
   - On VAD speech-end, after a short silence hangover (e.g. 400-600ms, to
     avoid duck/undock flapping on brief pauses): ramp gain back up
     (`linearRampToValueAtTime(1.0, ctx.currentTime + 0.3)`).
   - Tune ramp durations and hangover window by ear during testing — these
     are UX-feel parameters, not fixed constants; get them close to the
     AirPods-style smoothness before considering this done.
   - Make ramp durations and hangover window configurable (constants or env
     config), not hardcoded magic numbers buried in the ducking function.

6. **Confirm no regression of the original bug**
   - Explicitly test: ask a question, let the answer play, and confirm no
     audio break/glitch at the start of TTS playback (the original symptom,
     word 2-3).
   - Explicitly test with video paused and with video playing, since the
     original bug reproduced in both cases.
   - Explicitly test with Cursor and other heavy local apps closed (control
     for the resource-contention variable already ruled out) to keep this
     test isolated to the audio-graph change itself.

7. **Re-enable auto-listening**
   - Once Tasks 1-6 are verified, remove whatever flag/workaround currently
     disables auto-listening, since the root cause (stream restart) is now
     addressed rather than avoided.

## Constraints
- No stopping/restarting the mic `MediaStream` or recreating the
  `AudioContext` after session start, under any circumstance, per Task 1.
- Do not remove or bypass the existing wake-word ("Hey") gating for
  triggering actual questions — VAD-driven ducking is a separate, purely
  audio-mixing concern and must not cause utterances to be sent to
  `/ask/stream` on its own.
- Keep `echoCancellation: true` on the mic constraints throughout, since the
  stream is now long-lived and can safely benefit from it without the
  restart-transient cost that caused the original bug.
- This is a client-side (`/watch`) change only — no backend/API changes
  required for this feature.

## Acceptance criteria
- Mic stream and `AudioContext` are created exactly once per session and
  never torn down/recreated as part of ducking behavior.
- TTS and video program audio both duck and un-duck smoothly (gain ramps,
  not hard mute/unmute) in response to detected user speech.
- The original audio-break bug (word 2-3 glitch) does not reproduce, tested
  with video both paused and playing.
- Auto-listening is re-enabled and stays stable across at least 20
  consecutive question/answer cycles in manual testing.
- Ducking feels smooth and conversational (subjective — validate by ear
  against the AirPods Pro 2 Conversation Awareness reference behavior
  described in this prompt) rather than abrupt.
