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
| no | no | no | **1.0** after **1.5s** (+ BT pad) continuous silence (~0.6s recover) |

Cross-talk: if the user speaks while recovering, `cancelScheduledValues` runs
and gain locks back at the duck floor immediately.

## Files

| Path | Role |
|------|------|
| `ai_cowatcher/web/smart_ai_ducker.ts` | Canonical TypeScript class |
| `ai_cowatcher/web/smart_ai_ducker.js` | Browser / CommonJS runtime (served at `/watch/smart_ai_ducker.js`) |
| `ai_cowatcher/web/smart_ai_ducker.d.ts` | Types for editors |
| `ai_cowatcher/web/watch.html` | Wires PTT / STT / TTS / ask → hooks + AirPods sink bind |

`conversation_ducking.js` (RMS helpers) is **deprecated** and no longer loaded by `/watch`.

## Hook API

```ts
const ducker = new SmartAIDucker(videoEl, {
  duckGain: 0.15,
  duckAttackSec: 0.15,
  recoverySec: 0.6,
  holdMs: 1500,
  sampleRate: 48000,
  bluetoothOffsetMs: 150,
});
await ducker.start(); // video → GainNode @ 48 kHz; do not openMic for amplitude

ducker.onUserSpeechStart();
ducker.onUserSpeechEnd();
ducker.onAISpeechStart();
ducker.onAISpeechEnd();
ducker.setPipelineActive(true); // optional: duck while /ask is in-flight
```

On `/watch`, Hold-to-talk and SpeechRecognition call the user hooks; TTS calls
the AI hooks; submit/ask sets pipeline active. The Duck checkbox maps to
`setEnabled`.

## AirPods Pro 2 / Bluetooth defenses

Laptop speakers often work while AirPods fail because:

1. **HFP profile switch** drops the mic path to 8/16 kHz and can destabilize an
   unbound `AudioContext`.
2. **Sink mismatch** — HTML `<video>` plays on AirPods while the Web Audio
   destination (or a Gemini WebRTC track) stays on the laptop speaker.

### Unified sample rate

`start()` creates `new AudioContext({ sampleRate: 48000 })` so the browser
resamples gracefully instead of letting an HFP rate switch collapse nodes.

### Dual `setSinkId` — bind video + context to the same AirPods

```ts
import {
  SmartAIDucker,
  findAirPodsOutput,
  MIC_CONSTRAINTS,
} from "./smart_ai_ducker";

const videoEl = document.querySelector("video")!;
const ducker = new SmartAIDucker(videoEl, {
  holdMs: 1500,
  bluetoothOffsetMs: 150, // pad recover for BT air delay
  sampleRate: 48000,
});
await ducker.start();

// Labels need a prior mic permission in Chromium.
await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);

const devices = await navigator.mediaDevices.enumerateDevices();
const airpods = findAirPodsOutput(devices);
if (airpods?.deviceId) {
  // Locks BOTH videoElement.setSinkId + audioCtx.setSinkId, and enables BT pad.
  await ducker.bindToAirPods(airpods.deviceId);
}
```

`/watch` calls the same flow via `maybeBindAirPodsOutput()` after the graph starts.

### Mic constraints (Gemini / VAD)

```js
{
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false,      // avoid AirPods mic pumping
    latency: { ideal: 0.010 }, // hint for BT SCO timing
  },
  video: false,
}
```

Exported as `MIC_CONSTRAINTS` / `SmartAIDucker.MIC_CONSTRAINTS`.

### Volume recovery padding

Base conversational hold is **1500 ms**. When Bluetooth padding is active
(`bindToAirPods` or `setBluetoothPadding(true)`), effective hold becomes:

`holdMs + bluetoothOffsetMs` (default **1500 + 150 = 1650 ms**)

Tune with `setBluetoothPadding(200)` if your headset needs more air time for
the last Gemini packet.

## Wiring a lightweight client VAD (`@ricky0123/vad-web`)

```ts
import { MicVAD } from "@ricky0123/vad-web";
import { SmartAIDucker, MIC_CONSTRAINTS } from "./smart_ai_ducker";

const ducker = new SmartAIDucker(videoEl, { holdMs: 1500, duckGain: 0.15 });
await ducker.start();

const stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
const vad = await MicVAD.new({
  stream,
  onSpeechStart: () => ducker.onUserSpeechStart(),
  onSpeechEnd: () => ducker.onUserSpeechEnd(),
});
vad.start();
```

### Important

- Prefer headphones when testing; even with `echoCancellation: true`, open
  speakers can still confuse STT.
- Do not call `getUserMedia` again on every duck/unduck — keep one stream for
  the VAD (or let SpeechRecognition own the mic when using Hold-to-talk only).
- Never barge-in cancel TTS solely because VAD fired; only change program gain.
- `AudioContext.setSinkId` needs a Chromium build that supports it; failures are
  logged and the video sink bind still applies when available.
