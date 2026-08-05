/**
 * SmartAIDucker — state-driven conversation-aware program ducking.
 *
 * Duck when userSpeaking OR aiSpeaking (or optional pipeline hold).
 * Recover only after BOTH are silent for `holdMs` (+ optional Bluetooth pad).
 *
 * Bluetooth / AirPods Pro 2 defenses:
 * - Unified AudioContext sampleRate (48000)
 * - Dual setSinkId (videoElement + audioCtx) via bindToAirPods(deviceId)
 * - Mic constraints: AEC/NS on, AGC off, latency ideal 10ms
 * - Recovery hold padded for BT transport delay
 *
 * Browser: load via <script>, global `CowatcherAudio.SmartAIDucker`.
 * Node/tests: `require('./smart_ai_ducker.js')`.
 */
(function (root) {
  "use strict";

  /**
   * @typedef {object} SmartAIDuckerOptions
   * @property {number} [userDuckVolume=0.20] Floor while user talks.
   * @property {number} [geminiDuckVolume=0.05] Floor while Gemini TTS plays.
   * @property {number} [userCutSec=0.02] Instant cut duration on Hold-to-talk.
   * @property {number} [geminiDuckSec=0.2] Deep-duck ramp for Gemini.
   * @property {number} [duckGain=0.20] Deprecated alias of userDuckVolume.
   * @property {number} [fullGain=1]
   * @property {number} [duckAttackSec=0.15] Fast fade-out.
   * @property {number} [recoverySec=0.6] Gentle fade-in.
   * @property {number} [holdMs=1500] Absolute silence before recover.
   * @property {number} [bluetoothOffsetMs=150] Extra hold while BT padding on.
   * @property {boolean} [bluetoothPadding=false]
   * @property {number} [sampleRate=48000] Unified AudioContext rate.
   * @property {AudioContext} [audioContext]
   * @property {(state: SmartAIDuckerState) => void} [onStateChange]
   * @property {(gain: number) => void} [onGainChange]
   */

  /**
   * @typedef {object} SmartAIDuckerState
   * @property {boolean} userSpeaking
   * @property {boolean} aiSpeaking
   * @property {boolean} pipelineActive
   * @property {boolean} ducked
   * @property {number} gain
   * @property {boolean} recovering
   * @property {string|null} sinkId
   * @property {boolean} bluetoothPadding
   * @property {number} effectiveHoldMs
   * @property {number|null} sampleRate
   */

  const UNIFIED_SAMPLE_RATE = 48000;

  /** Movie volume while the user is talking (Hold to talk / VAD). */
  const USER_DUCK_VOLUME = 0.2;
  /** Movie volume while Gemini TTS is speaking (deep duck). */
  const GEMINI_DUCK_VOLUME = 0.05;
  /** Instant cut duration for Hold-to-talk (bypass exponential lag). */
  const USER_CUT_SEC = 0.02;
  /** Smooth deep-duck duration when Gemini starts. */
  const GEMINI_DUCK_SEC = 0.2;

  const DEFAULTS = Object.freeze({
    userDuckVolume: USER_DUCK_VOLUME,
    geminiDuckVolume: GEMINI_DUCK_VOLUME,
    userCutSec: USER_CUT_SEC,
    geminiDuckSec: GEMINI_DUCK_SEC,
    /** @deprecated alias of userDuckVolume */
    duckGain: USER_DUCK_VOLUME,
    fullGain: 1.0,
    /** @deprecated alias of userCutSec */
    duckAttackSec: USER_CUT_SEC,
    recoverySec: 0.6,
    holdMs: 1500,
    bluetoothOffsetMs: 150,
    sampleRate: UNIFIED_SAMPLE_RATE,
  });

  /** Bluetooth-friendly mic constraints for Gemini / VAD / STT. */
  const MIC_CONSTRAINTS = Object.freeze({
    audio: Object.freeze({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
      latency: Object.freeze({ ideal: 0.01 }),
    }),
    video: false,
  });

  /**
   * setTargetAtTime timeConstant ≈ duration/3 (~95% in ~3τ).
   * @param {number} durationSec
   */
  function timeConstantFor(durationSec) {
    return Math.max(0.02, (Number(durationSec) || 0.15) / 3);
  }

  /**
   * Pick AirPods / Bluetooth audiooutput from enumerateDevices().
   * @param {MediaDeviceInfo[]} devices
   * @returns {MediaDeviceInfo|null}
   */
  function findAirPodsOutput(devices) {
    if (!devices || !devices.length) return null;
    const outputs = [];
    for (let i = 0; i < devices.length; i += 1) {
      const d = devices[i];
      if (d && d.kind === "audiooutput" && d.deviceId) outputs.push(d);
    }
    if (!outputs.length) return null;

    for (let i = 0; i < outputs.length; i += 1) {
      if (/airpods/i.test(outputs[i].label || "")) return outputs[i];
    }
    for (let i = 0; i < outputs.length; i += 1) {
      if (
        /bluetooth|headset|buds|galaxy buds|pixel buds|wh-\d|sony|bose/i.test(
          outputs[i].label || ""
        )
      ) {
        return outputs[i];
      }
    }
    return null;
  }

  class SmartAIDucker {
    /**
     * @param {HTMLMediaElement} mediaElement
     * @param {SmartAIDuckerOptions} [options]
     */
    constructor(mediaElement, options) {
      if (!mediaElement || typeof mediaElement.play !== "function") {
        throw new TypeError("SmartAIDucker requires an HTMLMediaElement");
      }
      const o = options || {};
      this.mediaElement = mediaElement;
      const userDuck = clamp01(
        o.userDuckVolume != null
          ? o.userDuckVolume
          : o.duckGain != null
            ? o.duckGain
            : DEFAULTS.userDuckVolume
      );
      const userCut = Math.max(
        0.005,
        o.userCutSec != null
          ? o.userCutSec
          : o.duckAttackSec != null
            ? o.duckAttackSec
            : DEFAULTS.userCutSec
      );
      this.opts = {
        userDuckVolume: userDuck,
        geminiDuckVolume: clamp01(
          o.geminiDuckVolume != null ? o.geminiDuckVolume : DEFAULTS.geminiDuckVolume
        ),
        userCutSec: userCut,
        geminiDuckSec: Math.max(
          0.02,
          o.geminiDuckSec != null ? o.geminiDuckSec : DEFAULTS.geminiDuckSec
        ),
        duckGain: userDuck,
        duckAttackSec: userCut,
        fullGain: clamp01(o.fullGain != null ? o.fullGain : DEFAULTS.fullGain),
        recoverySec: Math.max(
          0.01,
          o.recoverySec != null ? o.recoverySec : DEFAULTS.recoverySec
        ),
        holdMs: Math.max(0, o.holdMs != null ? o.holdMs : DEFAULTS.holdMs),
        bluetoothOffsetMs: Math.max(
          0,
          o.bluetoothOffsetMs != null ? o.bluetoothOffsetMs : DEFAULTS.bluetoothOffsetMs
        ),
        sampleRate: Math.max(
          8000,
          o.sampleRate != null ? o.sampleRate : DEFAULTS.sampleRate
        ),
      };

      /** @type {AudioContext | null} */
      this.ctx = o.audioContext || null;
      /** @type {MediaElementAudioSourceNode | null} */
      this._mediaSource = null;
      /** @type {GainNode | null} */
      this.programGain = null;

      this._started = false;
      this._enabled = true;
      this._destroyed = false;
      this._corsPrepared = false;
      this._fallbackVolume = true;

      this._userSpeaking = false;
      this._aiSpeaking = false;
      this._pipelineActive = false;

      /** @type {ReturnType<typeof setTimeout> | null} */
      this._holdTimer = null;
      this._recovering = false;
      this._lastGain = this.opts.fullGain;
      this._generation = 0;

      this._bluetoothPadding = o.bluetoothPadding === true;
      /** @type {string|null} */
      this._sinkId = null;

      this._onStateChange =
        typeof o.onStateChange === "function" ? o.onStateChange : null;
      this._onGainChange =
        typeof o.onGainChange === "function" ? o.onGainChange : null;

      /** @type {MediaStream | null} */
      this.micStream = null;
      this._ownsMic = false;
    }

    // ── public getters ─────────────────────────────────────────────────────

    get isDucked() {
      return this._shouldDuck();
    }

    get isUserSpeaking() {
      return this._userSpeaking;
    }

    get isAISpeaking() {
      return this._aiSpeaking;
    }

    get currentGain() {
      return this._lastGain;
    }

    get sinkId() {
      return this._sinkId;
    }

    getEffectiveHoldMs() {
      const pad = this._bluetoothPadding ? this.opts.bluetoothOffsetMs : 0;
      return this.opts.holdMs + pad;
    }

    getState() {
      return {
        userSpeaking: this._userSpeaking,
        aiSpeaking: this._aiSpeaking,
        pipelineActive: this._pipelineActive,
        ducked: this._shouldDuck(),
        duckTier: this._duckTier(),
        gain: this._lastGain,
        recovering: this._recovering,
        sinkId: this._sinkId,
        bluetoothPadding: this._bluetoothPadding,
        effectiveHoldMs: this.getEffectiveHoldMs(),
        sampleRate: this.ctx ? this.ctx.sampleRate : null,
      };
    }

    // ── lifecycle ──────────────────────────────────────────────────────────

    /** Set crossOrigin before the media resource loads (CORS / MediaElementSource). */
    prepareMediaCors() {
      if (this._corsPrepared) return;
      try {
        if (!this.mediaElement.crossOrigin) {
          this.mediaElement.crossOrigin = "anonymous";
        }
      } catch (_) {
        /* ignore */
      }
      this._corsPrepared = true;
    }

    /**
     * Wire video → GainNode at unified sampleRate. Optionally open mic.
     * @param {{ openMic?: boolean }} [opts]
     */
    async start(opts) {
      this._assertAlive();
      this.prepareMediaCors();

      const AC = root.AudioContext || root.webkitAudioContext;
      if (!AC) throw new Error("Web Audio API unavailable");

      if (!this.ctx) {
        try {
          this.ctx = new AC({ sampleRate: this.opts.sampleRate });
        } catch (_) {
          this.ctx = new AC();
        }
      }
      if (this.ctx.state === "suspended") {
        try {
          await this.ctx.resume();
        } catch (_) {
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
        } catch (err) {
          this.programGain = null;
          this._fallbackVolume = true;
          if (typeof console !== "undefined" && console.warn) {
            console.warn(
              "[SmartAIDucker] MediaElementSource unavailable; volume fallback",
              err
            );
          }
        }
      }

      if (this._sinkId) {
        await this._applySinkId(this._sinkId);
      }

      if (opts && opts.openMic) {
        await this.ensureMicStream();
      }

      this._started = true;
      this._reconcile("start");
    }

    /**
     * Force video + AudioContext onto the same AirPods / BT output deviceId.
     * Enables Bluetooth recovery padding.
     * @param {string} deviceId
     */
    async bindToAirPods(deviceId) {
      this._assertAlive();
      const id = String(deviceId || "").trim();
      if (!id) throw new TypeError("bindToAirPods requires a non-empty deviceId");
      this._sinkId = id;
      this._bluetoothPadding = true;
      if (!this.ctx) {
        this._emitState();
        return;
      }
      await this._applySinkId(id);
      this._emitState();
    }

    /** Clear dual sink binding (default OS output). */
    async clearSinkBinding() {
      this._assertAlive();
      this._sinkId = null;
      await this._applySinkId("");
      this._emitState();
    }

    /**
     * Toggle Bluetooth recovery padding, or pass a number to set offset ms.
     * @param {boolean|number} enabledOrOffsetMs
     */
    setBluetoothPadding(enabledOrOffsetMs) {
      this._assertAlive();
      if (typeof enabledOrOffsetMs === "number") {
        this.opts.bluetoothOffsetMs = Math.max(0, enabledOrOffsetMs);
        this._bluetoothPadding = enabledOrOffsetMs > 0;
      } else {
        this._bluetoothPadding = Boolean(enabledOrOffsetMs);
      }
      if (this._recovering && !this._shouldDuck()) {
        this._scheduleRecover();
      }
      this._emitState();
    }

    /**
     * Acquire (once) a mic stream with Bluetooth-friendly constraints.
     * @returns {Promise<MediaStream | null>}
     */
    async ensureMicStream() {
      this._assertAlive();
      if (this.micStream) return this.micStream;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        return null;
      }
      try {
        this.micStream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
        this._ownsMic = true;
        return this.micStream;
      } catch (_) {
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
        } catch (_2) {
          this.micStream = null;
          this._ownsMic = false;
          return null;
        }
      }
    }

    /**
     * Mute/unmute owned mic tracks without stopping them.
     * @param {boolean} muted
     */
    setMicMuted(muted) {
      if (!this.micStream) return;
      try {
        this.micStream.getAudioTracks().forEach((t) => {
          t.enabled = !muted;
        });
      } catch (_) {
        /* ignore */
      }
    }

    /** @param {boolean} enabled */
    setEnabled(enabled) {
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
        this._reconcile("enabled");
      }
    }

    stop() {
      this._clearHoldTimer();
      this._recovering = false;
      this._rampTo(this.opts.fullGain, this.opts.recoverySec);
      this._emitState();
    }

    /** Full teardown (pagehide). */
    destroy() {
      if (this._destroyed) return;
      this._destroyed = true;
      this._clearHoldTimer();
      this._generation += 1;

      try {
        if (this._mediaSource) this._mediaSource.disconnect();
      } catch (_) {
        /* ignore */
      }
      try {
        if (this.programGain) this.programGain.disconnect();
      } catch (_) {
        /* ignore */
      }

      if (this._ownsMic && this.micStream) {
        try {
          this.micStream.getTracks().forEach((t) => t.stop());
        } catch (_) {
          /* ignore */
        }
      }
      this.micStream = null;
      this._ownsMic = false;
      this._mediaSource = null;
      this.programGain = null;
      this._started = false;
    }

    // ── integration hooks (AI SDK / VAD / UI) ──────────────────────────────

    /**
     * Hold-to-talk / user VAD — instant cut to USER_DUCK_VOLUME (bypasses setTargetAtTime).
     */
    onUserSpeechStart() {
      this._assertAlive();
      if (!this._enabled) return;
      this._userSpeaking = true;
      this._applyPriorityDuck("user");
    }

    onUserSpeechEnd() {
      this._assertAlive();
      this._userSpeaking = false;
      this._onActivityEnd("user");
    }

    /** Gemini / AI TTS started — deep-duck toward GEMINI_DUCK_VOLUME. */
    onAISpeechStart() {
      this._assertAlive();
      if (!this._enabled) return;
      this._aiSpeaking = true;
      this._applyPriorityDuck("ai");
    }

    onAISpeechEnd() {
      this._assertAlive();
      this._aiSpeaking = false;
      this._onActivityEnd("ai");
    }

    /** Alias for Gemini SDK naming. */
    onGeminiSpeechStart() {
      return this.onAISpeechStart();
    }

    /** Alias for Gemini SDK naming. */
    onGeminiSpeechEnd() {
      return this.onAISpeechEnd();
    }

    /**
     * @param {boolean} active
     */
    setPipelineActive(active) {
      this._assertAlive();
      const next = Boolean(active);
      if (next === this._pipelineActive) return;
      this._pipelineActive = next;
      if (next) this._applyPriorityDuck("pipeline");
      else this._onActivityEnd("pipeline");
    }

    /** @param {boolean} [forceImmediate] */
    reset(forceImmediate) {
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

    // ── internals ──────────────────────────────────────────────────────────

    /**
     * @param {string} deviceId
     */
    async _applySinkId(deviceId) {
      const el = this.mediaElement;
      const ctx = this.ctx;
      const tasks = [];

      if (el && typeof el.setSinkId === "function") {
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

    _assertAlive() {
      if (this._destroyed) {
        throw new Error("SmartAIDucker has been destroyed");
      }
    }

    _shouldDuck() {
      return this._userSpeaking || this._aiSpeaking || this._pipelineActive;
    }

    /**
     * @returns {"user"|"gemini"|"pipeline"|"none"}
     */
    _duckTier() {
      if (this._userSpeaking) return "user";
      if (this._aiSpeaking) return "gemini";
      if (this._pipelineActive) return "pipeline";
      return "none";
    }

    /**
     * Priority: user (20% instant) > Gemini (5% / 0.2s) > pipeline (20% instant).
     * @returns {number}
     */
    _duckTarget() {
      if (this._userSpeaking) return this.opts.userDuckVolume;
      if (this._aiSpeaking) return this.opts.geminiDuckVolume;
      if (this._pipelineActive) return this.opts.userDuckVolume;
      return this.opts.fullGain;
    }

    /**
     * Apply multi-tier duck from current speech flags.
     * @param {string} reason
     */
    _applyPriorityDuck(reason) {
      this._clearHoldTimer();
      this._recovering = false;
      this._generation += 1;
      const target = this._duckTarget();
      if (this._userSpeaking || reason === "user") {
        // Instant cut — Hold to talk / barge-in over Gemini (no exponential lag).
        this._cutTo(target, this.opts.userCutSec);
      } else if (this._aiSpeaking) {
        this._linearTo(target, this.opts.geminiDuckSec);
      } else {
        this._cutTo(target, this.opts.userCutSec);
      }
      this._emitState();
    }

    /**
     * Activity end: only recover when EVERYONE is silent, after hold (+ BT pad).
     * @param {string} _who
     */
    _onActivityEnd(_who) {
      this._emitState();
      if (this._shouldDuck()) {
        // Someone still active — re-apply priority (e.g. user ended, Gemini continues → deep duck).
        this._applyPriorityDuck("end");
        return;
      }
      this._scheduleRecover();
    }

    _reconcile(_why) {
      if (!this._enabled) return;
      if (this._shouldDuck()) {
        this._applyPriorityDuck("reconcile");
      } else {
        this._scheduleRecover();
      }
      this._emitState();
    }

    _scheduleRecover() {
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

    _clearHoldTimer() {
      if (this._holdTimer != null) {
        clearTimeout(this._holdTimer);
        this._holdTimer = null;
      }
    }

    /**
     * Instant / near-instant cut for Hold-to-talk (cancels any recover curve).
     * @param {number} target
     * @param {number} durationSec
     */
    _cutTo(target, durationSec) {
      const end = Math.min(1, Math.max(0, target));
      this._lastGain = end;
      const dur = Math.max(0.005, durationSec != null ? durationSec : USER_CUT_SEC);

      if (this.programGain && this.ctx) {
        const param = this.programGain.gain;
        const now = this.ctx.currentTime;
        try {
          param.cancelScheduledValues(now);
          param.setValueAtTime(param.value, now);
          // Aggressive linear snap (~20ms) — avoids setTargetAtTime Bluetooth lag.
          param.linearRampToValueAtTime(end, now + dur);
        } catch (_) {
          try {
            param.cancelScheduledValues(now);
            param.setValueAtTime(end, now);
          } catch (_2) {
            try {
              param.value = end;
            } catch (_3) {
              /* ignore */
            }
          }
        }
      } else if (this._fallbackVolume) {
        this._fadeElementVolume(end, dur);
      }

      if (this._onGainChange) {
        try {
          this._onGainChange(end);
        } catch (_) {
          /* ignore */
        }
      }
    }

    /**
     * Linear deep-duck for Gemini TTS clarity.
     * @param {number} target
     * @param {number} durationSec
     */
    _linearTo(target, durationSec) {
      const end = Math.min(1, Math.max(0, target));
      this._lastGain = end;
      const dur = Math.max(0.02, durationSec != null ? durationSec : GEMINI_DUCK_SEC);

      if (this.programGain && this.ctx) {
        const param = this.programGain.gain;
        const now = this.ctx.currentTime;
        try {
          param.cancelScheduledValues(now);
          param.setValueAtTime(param.value, now);
          param.linearRampToValueAtTime(end, now + dur);
        } catch (_) {
          try {
            param.value = end;
          } catch (_2) {
            /* ignore */
          }
        }
      } else if (this._fallbackVolume) {
        this._fadeElementVolume(end, dur);
      }

      if (this._onGainChange) {
        try {
          this._onGainChange(end);
        } catch (_) {
          /* ignore */
        }
      }
    }

    /**
     * Gentle recover via setTargetAtTime (AirPods-friendly).
     * @param {number} target
     * @param {number} durationSec
     */
    _rampTo(target, durationSec) {
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
          const settleAt = now + Math.max(durationSec || 0.6, tau * 4);
          param.setValueAtTime(end, settleAt);
        } catch (_) {
          try {
            param.value = end;
          } catch (_2) {
            /* ignore */
          }
        }
      } else if (this._fallbackVolume) {
        this._fadeElementVolume(end, durationSec);
      }

      if (this._onGainChange) {
        try {
          this._onGainChange(end);
        } catch (_) {
          /* ignore */
        }
      }
    }

    /**
     * @param {number} end
     * @param {number} durationSec
     */
    _fadeElementVolume(end, durationSec) {
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
      const step = (t) => {
        if (this._destroyed || gen !== this._generation) return;
        const u = Math.min(1, (t - t0) / ms);
        el.volume = start + (end - start) * (1 - (1 - u) * (1 - u));
        if (u < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    _emitState() {
      if (!this._onStateChange) return;
      try {
        this._onStateChange(this.getState());
      } catch (_) {
        /* ignore */
      }
    }
  }

  function clamp01(n) {
    const x = Number(n);
    if (!Number.isFinite(x)) return 0.15;
    return Math.max(0, Math.min(1, x));
  }

  const api = {
    SmartAIDucker,
    DEFAULTS,
    MIC_CONSTRAINTS,
    UNIFIED_SAMPLE_RATE,
    USER_DUCK_VOLUME,
    GEMINI_DUCK_VOLUME,
    USER_CUT_SEC,
    GEMINI_DUCK_SEC,
    timeConstantFor,
    findAirPodsOutput,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.CowatcherAudio = Object.assign({}, root.CowatcherAudio || {}, api);
})(typeof globalThis !== "undefined" ? globalThis : this);
