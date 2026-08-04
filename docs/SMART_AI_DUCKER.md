# SmartAIDucker — state-driven conversation ducking

Amplitude / RMS “VAD” on the mic was rolled back: room noise and speaker
bleed false-trigger ducks, and AI TTS feeding the mic creates a feedback
loop. Program volume is now driven only by **explicit speech-state hooks**.

## State matrix

| User speaking | AI speaking | Pipeline (ask) | Program gain |
|---------------|-------------|----------------|--------------|
| yes | * | * | **0.15** (duck, ~0.15s) |
| * | yes | * | **0.15** |
| * | * | yes | **0.15** |
| no | no | no | **1.0** after **1.5s** continuous silence (~0.6s recover) |

Cross-talk: if the user speaks while recovering, `cancelScheduledValues` runs
and gain locks back at the duck floor immediately.

## Files

| Path | Role |
|------|------|
| `ai_cowatcher/web/smart_ai_ducker.ts` | Canonical TypeScript class |
| `ai_cowatcher/web/smart_ai_ducker.js` | Browser / CommonJS runtime (served at `/watch/smart_ai_ducker.js`) |
| `ai_cowatcher/web/smart_ai_ducker.d.ts` | Types for editors |
| `ai_cowatcher/web/watch.html` | Wires PTT / STT / TTS / ask → hooks |

`conversation_ducking.js` (RMS helpers) is **deprecated** and no longer loaded by `/watch`.

## Hook API

```ts
const ducker = new SmartAIDucker(videoEl, {
  duckGain: 0.15,
  duckAttackSec: 0.15,
  recoverySec: 0.6,
  holdMs: 1500,
});
await ducker.start(); // video → GainNode only; do not openMic for amplitude

ducker.onUserSpeechStart();
ducker.onUserSpeechEnd();
ducker.onAISpeechStart();
ducker.onAISpeechEnd();
ducker.setPipelineActive(true); // optional: duck while /ask is in-flight
```

On `/watch`, Hold-to-talk and SpeechRecognition call the user hooks; TTS calls
the AI hooks; submit/ask sets pipeline active. The Duck checkbox maps to
`setEnabled`.

## Wiring a lightweight client VAD (`@ricky0123/vad-web`)

Install and serve the ONNX / worklet assets per the library docs, then:

```ts
import { MicVAD } from "@ricky0123/vad-web";
import { SmartAIDucker, MIC_CONSTRAINTS } from "./smart_ai_ducker";

const ducker = new SmartAIDucker(videoEl, { holdMs: 1500, duckGain: 0.15 });
await ducker.start();

// Reuse AEC+NS constraints so the mic does not hear the laptop speakers.
const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);

const vad = await MicVAD.new({
  stream,
  onSpeechStart: () => ducker.onUserSpeechStart(),
  onSpeechEnd: () => ducker.onUserSpeechEnd(),
});
vad.start();
```

### LiveKit Silero wrapper

Same idea: subscribe to their `VAD` / `SpeechEvent` callbacks and forward
`Speaking` → `onUserSpeechStart()` and `Silence` → `onUserSpeechEnd()`. Do
**not** also run an RMS analyser on the same stream for ducking.

### Important

- Prefer headphones when testing; even with `echoCancellation: true`, open
  speakers can still confuse STT.
- Do not call `getUserMedia` again on every duck/unduck — keep one stream for
  the VAD (or let SpeechRecognition own the mic when using Hold-to-talk only).
- Never barge-in cancel TTS solely because VAD fired; only change program gain.
