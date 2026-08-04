/**
 * SmartAIDucker — production state-driven program-audio ducker.
 *
 * Duck when user OR AI is speaking (or pipeline is active).
 * Recover only after both are silent for `holdMs` continuous milliseconds.
 *
 * Runtime companion: `smart_ai_ducker.js` (loaded by /watch without a bundler).
 * See docs/SMART_AI_DUCKER.md for VAD library wiring.
 */

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
};

/** Recommended getUserMedia constraints (AEC + NS) for client VAD / STT. */
export const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
  video: false,
};

const DEFAULTS = {
  duckGain: 0.15,
  fullGain: 1.0,
  duckAttackSec: 0.15,
  recoverySec: 0.6,
  holdMs: 1500,
} as const;

/** setTargetAtTime timeConstant ≈ duration/3 (~95% in ~3τ). */
export function timeConstantFor(durationSec: number): number {
  return Math.max(0.02, (Number(durationSec) || 0.15) / 3);
}

export class SmartAIDucker {
  static readonly MIC_CONSTRAINTS = MIC_CONSTRAINTS;

  readonly mediaElement: HTMLMediaElement;
  readonly opts: Required<
    Pick<
      SmartAIDuckerOptions,
      "duckGain" | "fullGain" | "duckAttackSec" | "recoverySec" | "holdMs"
    >
  > & {
    audioContext?: AudioContext;
  };

  ctx: AudioContext | null;
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

  private readonly _onStateChange: ((s: SmartAIDuckerState) => void) | null;
  private readonly _onGainChange: ((g: number) => void) | null;

  constructor(mediaElement: HTMLMediaElement, options: SmartAIDuckerOptions = {}) {
    if (!mediaElement || typeof mediaElement.play !== "function") {
      throw new TypeError("SmartAIDucker requires an HTMLMediaElement");
    }
    this.mediaElement = mediaElement;
    this.opts = {
      duckGain: clamp01(options.duckGain ?? DEFAULTS.duckGain),
      fullGain: clamp01(options.fullGain ?? DEFAULTS.fullGain),
      duckAttackSec: Math.max(0.01, options.duckAttackSec ?? DEFAULTS.duckAttackSec),
      recoverySec: Math.max(0.01, options.recoverySec ?? DEFAULTS.recoverySec),
      holdMs: Math.max(0, options.holdMs ?? DEFAULTS.holdMs),
      audioContext: options.audioContext,
    };
    this.ctx = options.audioContext ?? null;
    this.programGain = null;
    this.micStream = null;
    this._lastGain = this.opts.fullGain;
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

  getState(): SmartAIDuckerState {
    return {
      userSpeaking: this._userSpeaking,
      aiSpeaking: this._aiSpeaking,
      pipelineActive: this._pipelineActive,
      ducked: this._shouldDuck(),
      gain: this._lastGain,
      recovering: this._recovering,
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
   * Wire video → GainNode. Optionally open mic with echoCancellation for an
   * external VAD (does NOT start amplitude metering).
   */
  async start(opts?: { openMic?: boolean }): Promise<void> {
    this._assertAlive();
    this.prepareMediaCors();

    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) throw new Error("Web Audio API unavailable");

    if (!this.ctx) this.ctx = new AC();
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

    if (opts?.openMic) {
      await this.ensureMicStream();
    }

    this._started = true;
    this._reconcile();
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
      this.micStream = null;
      this._ownsMic = false;
      return null;
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

  /** Activity end: only recover when EVERYONE is silent, after holdMs. */
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
    }, this.opts.holdMs);
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
