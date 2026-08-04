/**
 * SmartAIDucker — production state-driven program-audio ducker.
 *
 * Duck when user OR AI is speaking (or pipeline is active).
 * Recover only after both are silent for `holdMs` (+ optional Bluetooth pad).
 *
 * Bluetooth / AirPods Pro 2 defenses:
 * - Unified AudioContext sampleRate (48 kHz) to survive HFP profile switches
 * - Dual setSinkId binding (video + AudioContext) onto the same output device
 * - Mic constraints tuned for BT (AEC/NS on, AGC off, low ideal latency)
 * - Recovery hold padded for ~100–200 ms BT transport delay
 *
 * Runtime companion: `smart_ai_ducker.js` (loaded by /watch without a bundler).
 * See docs/SMART_AI_DUCKER.md for VAD + AirPods init snippets.
 */

/** AudioContext.setSinkId is Chromium 110+; not in all TS lib DOM typings yet. */
type AudioContextWithSink = AudioContext & {
  setSinkId?: (sinkId: string) => Promise<void>;
  readonly sinkId?: string;
};

type MediaElementWithSink = HTMLMediaElement & {
  setSinkId?: (sinkId: string) => Promise<void>;
  readonly sinkId?: string;
};

export type SmartAIDuckerOptions = {
  /** Floor while ducked (default 0.15). */
  duckGain?: number;
  fullGain?: number;
  /** Fast fade-out duration seconds (default 0.15). */
  duckAttackSec?: number;
  /** Gentle fade-in duration seconds (default 0.6). */
  recoverySec?: number;
  /** Absolute silence before recover (default 1500). */
  holdMs?: number;
  /**
   * Extra silence hold while Bluetooth padding is active (default 150 ms).
   * Compensates for AirPods / BT transport delay vs laptop speakers.
   */
  bluetoothOffsetMs?: number;
  /** Start with Bluetooth recovery padding enabled. */
  bluetoothPadding?: boolean;
  /**
   * Hardcoded AudioContext sample rate (default 48000).
   * Forces graceful resample instead of HFP rate-switch node crashes.
   */
  sampleRate?: number;
  audioContext?: AudioContext;
  onStateChange?: (state: SmartAIDuckerState) => void;
  onGainChange?: (gain: number) => void;
};

export type SmartAIDuckerState = {
  userSpeaking: boolean;
  aiSpeaking: boolean;
  pipelineActive: boolean;
  ducked: boolean;
  gain: number;
  recovering: boolean;
  sinkId: string | null;
  bluetoothPadding: boolean;
  effectiveHoldMs: number;
  sampleRate: number | null;
};

/**
 * Bluetooth-friendly getUserMedia constraints for Gemini / VAD / STT.
 * AGC off reduces AirPods mic pumping; low ideal latency hints at SCO timing.
 */
export const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false,
    // Chromium accepts ContstrainDouble; cast keeps older TS DOM libs happy.
    latency: { ideal: 0.01 } as unknown as ConstrainDouble,
  },
  video: false,
};

/** Preferred unified graph rate for laptop + AirPods (A2DP / WebAudio). */
export const UNIFIED_SAMPLE_RATE = 48000;

const DEFAULTS = {
  duckGain: 0.15,
  fullGain: 1.0,
  duckAttackSec: 0.15,
  recoverySec: 0.6,
  holdMs: 1500,
  bluetoothOffsetMs: 150,
  sampleRate: UNIFIED_SAMPLE_RATE,
} as const;

/** setTargetAtTime timeConstant ≈ duration/3 (~95% in ~3τ). */
export function timeConstantFor(durationSec: number): number {
  return Math.max(0.02, (Number(durationSec) || 0.15) / 3);
}

/**
 * Pick an AirPods / Bluetooth audiooutput deviceId from enumerateDevices().
 * Prefers labels matching AirPods, then any bluetooth headset, else null.
 */
export function findAirPodsOutput(
  devices: ReadonlyArray<MediaDeviceInfo>
): MediaDeviceInfo | null {
  const outputs = devices.filter((d) => d.kind === "audiooutput" && d.deviceId);
  if (!outputs.length) return null;

  const airpods = outputs.find((d) => /airpods/i.test(d.label || ""));
  if (airpods) return airpods;

  const bt = outputs.find((d) =>
    /bluetooth|headset|buds|galaxy buds|pixel buds|wh-\d|sony|bose/i.test(d.label || "")
  );
  if (bt) return bt;

  return null;
}

export class SmartAIDucker {
  static readonly MIC_CONSTRAINTS = MIC_CONSTRAINTS;
  static readonly UNIFIED_SAMPLE_RATE = UNIFIED_SAMPLE_RATE;

  readonly mediaElement: MediaElementWithSink;
  readonly opts: {
    duckGain: number;
    fullGain: number;
    duckAttackSec: number;
    recoverySec: number;
    holdMs: number;
    bluetoothOffsetMs: number;
    sampleRate: number;
    audioContext?: AudioContext;
  };

  ctx: AudioContextWithSink | null;
  programGain: GainNode | null;
  micStream: MediaStream | null;

  private _mediaSource: MediaElementAudioSourceNode | null = null;
  private _started = false;
  private _enabled = true;
  private _destroyed = false;
  private _corsPrepared = false;
  private _fallbackVolume = true;
  private _ownsMic = false;

  private _userSpeaking = false;
  private _aiSpeaking = false;
  private _pipelineActive = false;
  private _holdTimer: ReturnType<typeof setTimeout> | null = null;
  private _recovering = false;
  private _lastGain: number;
  private _generation = 0;

  /** When true, effective hold = holdMs + bluetoothOffsetMs. */
  private _bluetoothPadding: boolean;
  private _sinkId: string | null = null;

  private readonly _onStateChange: ((s: SmartAIDuckerState) => void) | null;
  private readonly _onGainChange: ((g: number) => void) | null;

  constructor(mediaElement: HTMLMediaElement, options: SmartAIDuckerOptions = {}) {
    if (!mediaElement || typeof mediaElement.play !== "function") {
      throw new TypeError("SmartAIDucker requires an HTMLMediaElement");
    }
    this.mediaElement = mediaElement as MediaElementWithSink;
    this.opts = {
      duckGain: clamp01(options.duckGain ?? DEFAULTS.duckGain),
      fullGain: clamp01(options.fullGain ?? DEFAULTS.fullGain),
      duckAttackSec: Math.max(0.01, options.duckAttackSec ?? DEFAULTS.duckAttackSec),
      recoverySec: Math.max(0.01, options.recoverySec ?? DEFAULTS.recoverySec),
      holdMs: Math.max(0, options.holdMs ?? DEFAULTS.holdMs),
      bluetoothOffsetMs: Math.max(0, options.bluetoothOffsetMs ?? DEFAULTS.bluetoothOffsetMs),
      sampleRate: Math.max(8000, options.sampleRate ?? DEFAULTS.sampleRate),
      audioContext: options.audioContext,
    };
    this.ctx = (options.audioContext as AudioContextWithSink | undefined) ?? null;
    this.programGain = null;
    this.micStream = null;
    this._lastGain = this.opts.fullGain;
    this._bluetoothPadding = options.bluetoothPadding === true;
    this._onStateChange =
      typeof options.onStateChange === "function" ? options.onStateChange : null;
    this._onGainChange =
      typeof options.onGainChange === "function" ? options.onGainChange : null;
  }

  get isDucked(): boolean {
    return this._shouldDuck();
  }

  get isUserSpeaking(): boolean {
    return this._userSpeaking;
  }

  get isAISpeaking(): boolean {
    return this._aiSpeaking;
  }

  get currentGain(): number {
    return this._lastGain;
  }

  get sinkId(): string | null {
    return this._sinkId;
  }

  /** holdMs + Bluetooth pad when padding is active. */
  getEffectiveHoldMs(): number {
    const pad = this._bluetoothPadding ? this.opts.bluetoothOffsetMs : 0;
    return this.opts.holdMs + pad;
  }

  getState(): SmartAIDuckerState {
    return {
      userSpeaking: this._userSpeaking,
      aiSpeaking: this._aiSpeaking,
      pipelineActive: this._pipelineActive,
      ducked: this._shouldDuck(),
      gain: this._lastGain,
      recovering: this._recovering,
      sinkId: this._sinkId,
      bluetoothPadding: this._bluetoothPadding,
      effectiveHoldMs: this.getEffectiveHoldMs(),
      sampleRate: this.ctx ? this.ctx.sampleRate : null,
    };
  }

  prepareMediaCors(): void {
    if (this._corsPrepared) return;
    try {
      if (!this.mediaElement.crossOrigin) {
        this.mediaElement.crossOrigin = "anonymous";
      }
    } catch {
      /* ignore */
    }
    this._corsPrepared = true;
  }

  /**
   * Wire video → GainNode at a unified sampleRate.
   * Optionally open mic with Bluetooth-friendly constraints (no amplitude VAD).
   */
  async start(opts?: { openMic?: boolean }): Promise<void> {
    this._assertAlive();
    this.prepareMediaCors();

    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) throw new Error("Web Audio API unavailable");

    if (!this.ctx) {
      // Hardcode sample rate so AirPods HFP (8/16 kHz) cannot collapse the graph.
      try {
        this.ctx = new AC({ sampleRate: this.opts.sampleRate }) as AudioContextWithSink;
      } catch {
        this.ctx = new AC() as AudioContextWithSink;
      }
    }

    if (this.ctx.state === "suspended") {
      try {
        await this.ctx.resume();
      } catch {
        /* gesture required */
      }
    }

    if (!this.programGain) {
      try {
        this._mediaSource = this.ctx.createMediaElementSource(this.mediaElement);
        this.programGain = this.ctx.createGain();
        this.programGain.gain.value = this.opts.fullGain;
        this._mediaSource.connect(this.programGain);
        this.programGain.connect(this.ctx.destination);
        this._fallbackVolume = false;
      } catch {
        this.programGain = null;
        this._fallbackVolume = true;
      }
    }

    if (this._sinkId) {
      await this._applySinkId(this._sinkId);
    }

    if (opts?.openMic) {
      await this.ensureMicStream();
    }

    this._started = true;
    this._reconcile();
  }

  /**
   * Lock HTMLMediaElement + AudioContext onto the same physical output
   * (e.g. AirPods Pro 2 deviceId from enumerateDevices).
   * Also enables Bluetooth recovery padding.
   */
  async bindToAirPods(deviceId: string): Promise<void> {
    this._assertAlive();
    const id = String(deviceId || "").trim();
    if (!id) {
      throw new TypeError("bindToAirPods requires a non-empty deviceId");
    }
    this._sinkId = id;
    this._bluetoothPadding = true;

    if (!this.ctx) {
      // Graph not started yet — sink applied in start().
      this._emitState();
      return;
    }
    await this._applySinkId(id);
    this._emitState();
  }

  /** Clear dual sink binding (back to default OS output). */
  async clearSinkBinding(): Promise<void> {
    this._assertAlive();
    this._sinkId = null;
    await this._applySinkId("");
    this._emitState();
  }

  /**
   * Toggle / set Bluetooth recovery padding.
   * Pass a number to also update bluetoothOffsetMs.
   */
  setBluetoothPadding(enabledOrOffsetMs: boolean | number): void {
    this._assertAlive();
    if (typeof enabledOrOffsetMs === "number") {
      this.opts.bluetoothOffsetMs = Math.max(0, enabledOrOffsetMs);
      this._bluetoothPadding = enabledOrOffsetMs > 0;
    } else {
      this._bluetoothPadding = Boolean(enabledOrOffsetMs);
    }
    // If a hold is already armed, reschedule with the new effective duration.
    if (this._recovering && !this._shouldDuck()) {
      this._scheduleRecover();
    }
    this._emitState();
  }

  async ensureMicStream(): Promise<MediaStream | null> {
    this._assertAlive();
    if (this.micStream) return this.micStream;
    if (!navigator.mediaDevices?.getUserMedia) return null;
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
      this._ownsMic = true;
      return this.micStream;
    } catch {
      // Retry without latency constraint if the UA rejects it.
      try {
        this.micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: false,
          },
          video: false,
        });
        this._ownsMic = true;
        return this.micStream;
      } catch {
        this.micStream = null;
        this._ownsMic = false;
        return null;
      }
    }
  }

  setMicMuted(muted: boolean): void {
    if (!this.micStream) return;
    try {
      this.micStream.getAudioTracks().forEach((t) => {
        t.enabled = !muted;
      });
    } catch {
      /* ignore */
    }
  }

  setEnabled(enabled: boolean): void {
    this._enabled = Boolean(enabled);
    if (!this._enabled) {
      this._clearHoldTimer();
      this._userSpeaking = false;
      this._aiSpeaking = false;
      this._pipelineActive = false;
      this._recovering = false;
      this._rampTo(this.opts.fullGain, this.opts.recoverySec);
      this._emitState();
    } else {
      this._reconcile();
    }
  }

  stop(): void {
    this._clearHoldTimer();
    this._recovering = false;
    this._rampTo(this.opts.fullGain, this.opts.recoverySec);
    this._emitState();
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._clearHoldTimer();
    this._generation += 1;

    try {
      this._mediaSource?.disconnect();
    } catch {
      /* ignore */
    }
    try {
      this.programGain?.disconnect();
    } catch {
      /* ignore */
    }

    if (this._ownsMic && this.micStream) {
      try {
        this.micStream.getTracks().forEach((t) => t.stop());
      } catch {
        /* ignore */
      }
    }
    this.micStream = null;
    this._ownsMic = false;
    this._mediaSource = null;
    this.programGain = null;
    this._started = false;
  }

  /** User VAD / PTT / SpeechRecognition → speech began. */
  onUserSpeechStart(): void {
    this._assertAlive();
    if (!this._enabled) return;
    this._userSpeaking = true;
    this._onActivityStart();
  }

  /** User VAD / PTT / SpeechRecognition → speech ended. */
  onUserSpeechEnd(): void {
    this._assertAlive();
    this._userSpeaking = false;
    this._onActivityEnd();
  }

  /** AI TTS / streaming audio → playback began. */
  onAISpeechStart(): void {
    this._assertAlive();
    if (!this._enabled) return;
    this._aiSpeaking = true;
    this._onActivityStart();
  }

  /** AI TTS / streaming audio → playback ended. */
  onAISpeechEnd(): void {
    this._assertAlive();
    this._aiSpeaking = false;
    this._onActivityEnd();
  }

  /**
   * Optional: duck while /ask is in-flight (thinking) before TTS starts.
   */
  setPipelineActive(active: boolean): void {
    this._assertAlive();
    const next = Boolean(active);
    if (next === this._pipelineActive) return;
    this._pipelineActive = next;
    if (next) this._onActivityStart();
    else this._onActivityEnd();
  }

  reset(forceImmediate?: boolean): void {
    this._assertAlive();
    this._userSpeaking = false;
    this._aiSpeaking = false;
    this._pipelineActive = false;
    this._clearHoldTimer();
    if (forceImmediate) {
      this._recovering = false;
      this._rampTo(this.opts.fullGain, this.opts.recoverySec);
      this._emitState();
    } else {
      this._scheduleRecover();
    }
  }

  // —— internals ——

  private async _applySinkId(deviceId: string): Promise<void> {
    const el = this.mediaElement;
    const ctx = this.ctx;

    const tasks: Promise<void>[] = [];

    if (typeof el.setSinkId === "function") {
      tasks.push(
        el.setSinkId(deviceId).catch((err) => {
          if (typeof console !== "undefined" && console.warn) {
            console.warn("[SmartAIDucker] videoElement.setSinkId failed", err);
          }
        })
      );
    }

    if (ctx && typeof ctx.setSinkId === "function") {
      tasks.push(
        ctx.setSinkId(deviceId).catch((err) => {
          if (typeof console !== "undefined" && console.warn) {
            console.warn("[SmartAIDucker] audioCtx.setSinkId failed", err);
          }
        })
      );
    }

    if (tasks.length) await Promise.all(tasks);
  }

  private _assertAlive(): void {
    if (this._destroyed) throw new Error("SmartAIDucker has been destroyed");
  }

  private _shouldDuck(): boolean {
    return this._userSpeaking || this._aiSpeaking || this._pipelineActive;
  }

  /** Cross-talk / activity start: cancel any scheduled recover, lock ducked. */
  private _onActivityStart(): void {
    this._clearHoldTimer();
    this._recovering = false;
    this._generation += 1;
    this._rampTo(this.opts.duckGain, this.opts.duckAttackSec);
    this._emitState();
  }

  /** Activity end: only recover when EVERYONE is silent, after hold (+ BT pad). */
  private _onActivityEnd(): void {
    this._emitState();
    if (this._shouldDuck()) {
      this._clearHoldTimer();
      this._recovering = false;
      this._rampTo(this.opts.duckGain, this.opts.duckAttackSec);
      return;
    }
    this._scheduleRecover();
  }

  private _reconcile(): void {
    if (!this._enabled) return;
    if (this._shouldDuck()) {
      this._clearHoldTimer();
      this._recovering = false;
      this._rampTo(this.opts.duckGain, this.opts.duckAttackSec);
    } else {
      this._scheduleRecover();
    }
    this._emitState();
  }

  private _scheduleRecover(): void {
    this._clearHoldTimer();
    if (!this._enabled || this._shouldDuck()) return;

    const gen = this._generation;
    const hold = this.getEffectiveHoldMs();
    this._recovering = true;
    this._emitState();

    this._holdTimer = setTimeout(() => {
      this._holdTimer = null;
      if (gen !== this._generation) return;
      if (this._destroyed || !this._enabled || this._shouldDuck()) {
        this._recovering = false;
        return;
      }
      this._recovering = false;
      this._rampTo(this.opts.fullGain, this.opts.recoverySec);
      this._emitState();
    }, hold);
  }

  private _clearHoldTimer(): void {
    if (this._holdTimer != null) {
      clearTimeout(this._holdTimer);
      this._holdTimer = null;
    }
  }

  /**
   * Smooth exponential-style gain via setTargetAtTime; cancel prior automation.
   */
  private _rampTo(target: number, durationSec: number): void {
    const end = Math.min(1, Math.max(0, target));
    this._lastGain = end;

    if (this.programGain && this.ctx) {
      const param = this.programGain.gain;
      const now = this.ctx.currentTime;
      try {
        param.cancelScheduledValues(now);
        param.setValueAtTime(param.value, now);
        const tau = timeConstantFor(durationSec);
        param.setTargetAtTime(end, now, tau);
        const settleAt = now + Math.max(durationSec || 0.15, tau * 4);
        param.setValueAtTime(end, settleAt);
      } catch {
        try {
          param.value = end;
        } catch {
          /* ignore */
        }
      }
    } else if (this._fallbackVolume) {
      this._fadeElementVolume(end, durationSec);
    }

    if (this._onGainChange) {
      try {
        this._onGainChange(end);
      } catch {
        /* ignore */
      }
    }
  }

  private _fadeElementVolume(end: number, durationSec: number): void {
    const el = this.mediaElement;
    const start = el.volume;
    if (Math.abs(start - end) < 0.01) {
      el.volume = end;
      return;
    }
    if (typeof requestAnimationFrame !== "function" || typeof performance === "undefined") {
      el.volume = end;
      return;
    }
    const ms = Math.max(40, (durationSec || 0.15) * 1000);
    const t0 = performance.now();
    const gen = this._generation;
    const step = (t: number) => {
      if (this._destroyed || gen !== this._generation) return;
      const u = Math.min(1, (t - t0) / ms);
      el.volume = start + (end - start) * (1 - (1 - u) * (1 - u));
      if (u < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  private _emitState(): void {
    if (!this._onStateChange) return;
    try {
      this._onStateChange(this.getState());
    } catch {
      /* ignore */
    }
  }
}

function clamp01(n: number): number {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0.15;
  return Math.max(0, Math.min(1, x));
}

export default SmartAIDucker;
