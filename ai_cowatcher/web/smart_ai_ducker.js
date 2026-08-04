/**
 * SmartAIDucker — state-driven conversation-aware program ducking.
 *
 * Duck when userSpeaking OR aiSpeaking (or optional pipeline hold).
 * Recover only after BOTH are silent for `holdMs` (default 1.5s).
 *
 * No microphone amplitude metering — user speech comes from an external VAD
 * (e.g. @ricky0123/vad-web) or UI events (Hold to talk / STT). AI speech
 * comes from TTS / streaming SDK hooks.
 *
 * Graph: <video|audio> → MediaElementSource → GainNode → destination
 *
 * Browser: load via <script>, global `CowatcherAudio.SmartAIDucker`.
 * Node/tests: `require('./smart_ai_ducker.js')`.
 */
(function (root) {
  "use strict";

  /**
   * @typedef {object} SmartAIDuckerOptions
   * @property {number} [duckGain=0.15] Floor while ducked (15%).
   * @property {number} [fullGain=1]
   * @property {number} [duckAttackSec=0.15] Fast fade-out.
   * @property {number} [recoverySec=0.6] Gentle fade-in.
   * @property {number} [holdMs=1500] Absolute silence before recover.
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
   */

  const DEFAULTS = Object.freeze({
    duckGain: 0.15,
    fullGain: 1.0,
    duckAttackSec: 0.15,
    recoverySec: 0.6,
    holdMs: 1500,
  });

  /** WebRTC-friendly mic constraints for an external VAD to reuse. */
  const MIC_CONSTRAINTS = Object.freeze({
    audio: Object.freeze({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
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

  class SmartAIDucker {
    /**
     * @param {HTMLMediaElement} mediaElement
     * @param {SmartAIDuckerOptions} [options]
     */
    constructor(mediaElement, options) {
      if (!mediaElement || typeof mediaElement.play !== "function") {
        throw new TypeError("SmartAIDucker requires an HTMLMediaElement");
      }
      this.mediaElement = mediaElement;
      this.opts = Object.assign({}, DEFAULTS, options || {});

      /** @type {AudioContext | null} */
      this.ctx = (options && options.audioContext) || null;
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

      this._onStateChange =
        options && typeof options.onStateChange === "function"
          ? options.onStateChange
          : null;
      this._onGainChange =
        options && typeof options.onGainChange === "function"
          ? options.onGainChange
          : null;

      /** Optional mic stream for external VAD (owned if we created it). */
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

    getState() {
      return {
        userSpeaking: this._userSpeaking,
        aiSpeaking: this._aiSpeaking,
        pipelineActive: this._pipelineActive,
        ducked: this._shouldDuck(),
        gain: this._lastGain,
        recovering: this._recovering,
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
     * Wire video → GainNode. Optionally open mic with echoCancellation for
     * an external VAD (does NOT start amplitude metering).
     * @param {{ openMic?: boolean }} [opts]
     */
    async start(opts) {
      this._assertAlive();
      this.prepareMediaCors();

      const AC = root.AudioContext || root.webkitAudioContext;
      if (!AC) throw new Error("Web Audio API unavailable");

      if (!this.ctx) this.ctx = new AC();
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

      if (opts && opts.openMic) {
        await this.ensureMicStream();
      }

      this._started = true;
      this._reconcile("start");
    }

    /**
     * Acquire (once) a mic stream with WebRTC echoCancellation for external VAD.
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
        this.micStream = null;
        this._ownsMic = false;
        return null;
      }
    }

    /**
     * Mute/unmute owned mic tracks without stopping them (frees hardware for STT).
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

    /**
     * Pause recovery / ducking without destroying the graph.
     * Does not clear speech flags — call speech-end hooks or reset() first if needed.
     */
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

      if (this.ctx) {
        try {
          // Do not close shared contexts passed in; only close if we created it
          // and nothing else needs it. Leave open to avoid Safari quirks — GC on nav.
        } catch (_) {
          /* ignore */
        }
      }
    }

    // ── integration hooks (AI SDK / VAD / UI) ──────────────────────────────

    /** User started speaking (Silero / @ricky0123/vad-web / Hold-to-talk). */
    onUserSpeechStart() {
      this._assertAlive();
      if (!this._enabled) return;
      this._userSpeaking = true;
      this._onActivityStart("user");
    }

    /** User stopped speaking (VAD end / release Hold-to-talk). */
    onUserSpeechEnd() {
      this._assertAlive();
      this._userSpeaking = false;
      this._onActivityEnd("user");
    }

    /** AI / TTS / streamed voice started. */
    onAISpeechStart() {
      this._assertAlive();
      if (!this._enabled) return;
      this._aiSpeaking = true;
      this._onActivityStart("ai");
    }

    /** AI / TTS finished. */
    onAISpeechEnd() {
      this._assertAlive();
      this._aiSpeaking = false;
      this._onActivityEnd("ai");
    }

    /**
     * Optional: duck while /ask is in-flight (thinking) before TTS starts.
     * @param {boolean} active
     */
    setPipelineActive(active) {
      this._assertAlive();
      const next = Boolean(active);
      if (next === this._pipelineActive) return;
      this._pipelineActive = next;
      if (next) this._onActivityStart("pipeline");
      else this._onActivityEnd("pipeline");
    }

    /** Clear all speech flags and unduck after hold (or immediately if force). */
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

    _assertAlive() {
      if (this._destroyed) {
        throw new Error("SmartAIDucker has been destroyed");
      }
    }

    _shouldDuck() {
      return this._userSpeaking || this._aiSpeaking || this._pipelineActive;
    }

    /**
     * Cross-talk / activity start: cancel any scheduled recover, lock ducked.
     * @param {string} _who
     */
    _onActivityStart(_who) {
      this._clearHoldTimer();
      this._recovering = false;
      this._generation += 1;
      this._rampTo(this.opts.duckGain, this.opts.duckAttackSec);
      this._emitState();
    }

    /**
     * Activity end: only recover when EVERYONE is silent, after holdMs.
     * @param {string} _who
     */
    _onActivityEnd(_who) {
      this._emitState();
      if (this._shouldDuck()) {
        // Still someone speaking — stay ducked; cancel recover if any.
        this._clearHoldTimer();
        this._recovering = false;
        this._rampTo(this.opts.duckGain, this.opts.duckAttackSec);
        return;
      }
      this._scheduleRecover();
    }

    _reconcile(_why) {
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

    _scheduleRecover() {
      this._clearHoldTimer();
      if (!this._enabled || this._shouldDuck()) return;

      const gen = this._generation;
      this._recovering = true;
      this._emitState();

      this._holdTimer = setTimeout(() => {
        this._holdTimer = null;
        // Stale timer (cross-talk reset bumped generation) — abort.
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

    _clearHoldTimer() {
      if (this._holdTimer != null) {
        clearTimeout(this._holdTimer);
        this._holdTimer = null;
      }
    }

    /**
     * Smooth exponential-style gain via setTargetAtTime; cancel prior automation.
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
          // Cross-talk reset: kill any in-flight recover curve immediately.
          param.cancelScheduledValues(now);
          param.setValueAtTime(param.value, now);
          const tau = timeConstantFor(durationSec);
          param.setTargetAtTime(end, now, tau);
          // Settle exactly after ~4τ so we don't drift forever toward target.
          const settleAt = now + Math.max(durationSec || 0.15, tau * 4);
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

  const api = {
    SmartAIDucker,
    DEFAULTS,
    MIC_CONSTRAINTS,
    timeConstantFor,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.CowatcherAudio = Object.assign({}, root.CowatcherAudio || {}, api);
})(typeof globalThis !== "undefined" ? globalThis : this);
