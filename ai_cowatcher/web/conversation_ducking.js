/**
 * DEPRECATED — RMS / amplitude helpers from the first conversation-ducking pass.
 * /watch now uses SmartAIDucker (smart_ai_ducker.js): state hooks only, no mic metering.
 * Kept for unit-test compatibility and older bookmarks of /watch/conversation_ducking.js.
 *
 * Pure helpers for conversation-aware (VAD + gain) ducking on /watch.
 * Loaded before the inline watch page script; also unit-testable in isolation.
 *
 * Design: mic MediaStream + AudioContext are owned by the page session and never
 * restarted for ducking. Only gain targets change.
 */
(function (root) {
  "use strict";

  /** Configurable UX tunables (not magic numbers buried in ducking functions). */
  const DUCK_CONFIG = {
    /** Program + TTS gain while user is speaking / ask / TTS is active (~15%). */
    DUCK_GAIN: 0.15,
    /** Milder program gain while ambient mic is open (no wake word yet). */
    AMBIENT_GAIN: 0.48,
    /** Silence hangover before unducking (ms) — avoids flap on brief pauses. */
    SPEECH_HANGOVER_MS: 500,
    /** Linear ramp down duration (seconds) on speech-start. */
    RAMP_DOWN_S: 0.15,
    /** Linear ramp up duration (seconds) after hangover. */
    RAMP_UP_S: 0.3,
    /** Floor RMS threshold for speech-like energy (0–1 scale from float samples). */
    VAD_RMS_THRESHOLD: 0.04,
    /** Never duck from threshold lower than this. */
    VAD_MIN_THRESHOLD: 0.028,
    /** Adaptive noise floor + margin for threshold. */
    VAD_THRESHOLD_MARGIN: 0.016,
    VAD_NOISE_ALPHA: 0.03,
    /** How often to sample the analyser. */
    VAD_POLL_MS: 40,
    /** Legacy HTMLMediaElement.volume floor when Web Audio graph is unavailable. */
    VOLUME_FALLBACK_MIN: 0.08,
  };

  /**
   * RMS energy from float time-domain samples (-1..1).
   * @param {ArrayLike<number>} samples
   * @returns {number}
   */
  function computeRms(samples) {
    if (!samples || !samples.length) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i += 1) {
      const v = samples[i];
      sum += v * v;
    }
    return Math.sqrt(sum / samples.length);
  }

  /**
   * Adaptive speech detection threshold from noise floor + config floor.
   * @param {number} noiseFloorRms
   * @param {object} [cfg]
   * @returns {number}
   */
  function vadThreshold(noiseFloorRms, cfg) {
    const c = cfg || DUCK_CONFIG;
    const floor = typeof noiseFloorRms === "number" && !Number.isNaN(noiseFloorRms)
      ? noiseFloorRms
      : 0;
    return Math.max(
      c.VAD_MIN_THRESHOLD,
      Math.max(c.VAD_RMS_THRESHOLD, floor + c.VAD_THRESHOLD_MARGIN)
    );
  }

  /**
   * Update exponential noise-floor estimate (only when not already in speech).
   * @param {number} prevFloor
   * @param {number} rms
   * @param {boolean} speechActive
   * @param {object} [cfg]
   * @returns {number}
   */
  function updateNoiseFloor(prevFloor, rms, speechActive, cfg) {
    const c = cfg || DUCK_CONFIG;
    if (speechActive) return prevFloor;
    const a = c.VAD_NOISE_ALPHA;
    const prev = typeof prevFloor === "number" && !Number.isNaN(prevFloor) ? prevFloor : 0.01;
    const sample = typeof rms === "number" && !Number.isNaN(rms) ? rms : 0;
    return prev * (1 - a) + sample * a;
  }

  /**
   * Target gains for program (video) and TTS from duck reasons + VAD.
   * Does not cancel TTS — only lowers gains (no barge-in synthesis cancel).
   *
   * @param {{
   *   enabled: boolean,
   *   reasons: Iterable<string>|Set<string>|string[],
   *   vadSpeechActive: boolean,
   *   cfg?: object
   * }} opts
   * @returns {{ program: number, tts: number, rampSec: number }}
   */
  function targetGains(opts) {
    const c = (opts && opts.cfg) || DUCK_CONFIG;
    if (!opts || !opts.enabled) {
      return { program: 1, tts: 1, rampSec: c.RAMP_UP_S };
    }
    const reasonSet = opts.reasons instanceof Set
      ? opts.reasons
      : new Set(opts.reasons || []);
    const hard =
      reasonSet.has("ask")
      || reasonSet.has("tts")
      || reasonSet.has("speech")
      || reasonSet.has("ptt")
      || Boolean(opts.vadSpeechActive);
    let program = 1;
    let tts = 1;
    if (hard) {
      program = Math.min(program, c.DUCK_GAIN);
      tts = Math.min(tts, c.DUCK_GAIN);
    } else if (reasonSet.has("ambient")) {
      program = Math.min(program, c.AMBIENT_GAIN);
    }
    const ducking = program < 0.99 || tts < 0.99;
    return {
      program,
      tts,
      rampSec: ducking ? c.RAMP_DOWN_S : c.RAMP_UP_S,
    };
  }

  /**
   * Whether silence hangover has elapsed after last speech energy.
   * @param {boolean} speechActive
   * @param {number} lastSpeechMs
   * @param {number} nowMs
   * @param {object} [cfg]
   * @returns {boolean} true if should end speech (unduck after hangover)
   */
  function hangoverElapsed(speechActive, lastSpeechMs, nowMs, cfg) {
    if (!speechActive) return false;
    const c = cfg || DUCK_CONFIG;
    return (nowMs - lastSpeechMs) >= c.SPEECH_HANGOVER_MS;
  }

  /**
   * Mic constraints for the once-per-session getUserMedia call.
   * echoCancellation stays on for the long-lived stream (no restart cost).
   */
  const MIC_CONSTRAINTS = {
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  };

  const api = {
    DUCK_CONFIG,
    MIC_CONSTRAINTS,
    computeRms,
    vadThreshold,
    updateNoiseFloor,
    targetGains,
    hangoverElapsed,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.CowatcherDucking = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
