export type SmartAIDuckerOptions = {
  userDuckVolume?: number;
  geminiDuckVolume?: number;
  userCutSec?: number;
  geminiDuckSec?: number;
  /** @deprecated Prefer userDuckVolume. */
  duckGain?: number;
  fullGain?: number;
  /** @deprecated Prefer userCutSec. */
  duckAttackSec?: number;
  recoverySec?: number;
  holdMs?: number;
  bluetoothOffsetMs?: number;
  bluetoothPadding?: boolean;
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
  duckTier: "user" | "gemini" | "pipeline" | "none";
  gain: number;
  recovering: boolean;
  sinkId: string | null;
  bluetoothPadding: boolean;
  effectiveHoldMs: number;
  sampleRate: number | null;
};

export declare const MIC_CONSTRAINTS: MediaStreamConstraints;
export declare const UNIFIED_SAMPLE_RATE: number;
export declare const USER_DUCK_VOLUME: number;
export declare const GEMINI_DUCK_VOLUME: number;
export declare const USER_CUT_SEC: number;
export declare const GEMINI_DUCK_SEC: number;
export declare function timeConstantFor(durationSec: number): number;
export declare function findAirPodsOutput(
  devices: ReadonlyArray<MediaDeviceInfo>
): MediaDeviceInfo | null;

export declare class SmartAIDucker {
  static readonly MIC_CONSTRAINTS: MediaStreamConstraints;
  static readonly UNIFIED_SAMPLE_RATE: number;
  readonly mediaElement: HTMLMediaElement;
  ctx: AudioContext | null;
  programGain: GainNode | null;
  micStream: MediaStream | null;

  constructor(mediaElement: HTMLMediaElement, options?: SmartAIDuckerOptions);

  readonly isDucked: boolean;
  readonly isUserSpeaking: boolean;
  readonly isAISpeaking: boolean;
  readonly currentGain: number;
  readonly sinkId: string | null;

  getEffectiveHoldMs(): number;
  getState(): SmartAIDuckerState;
  prepareMediaCors(): void;
  start(opts?: { openMic?: boolean }): Promise<void>;
  bindToAirPods(deviceId: string): Promise<void>;
  clearSinkBinding(): Promise<void>;
  setBluetoothPadding(enabledOrOffsetMs: boolean | number): void;
  ensureMicStream(): Promise<MediaStream | null>;
  setMicMuted(muted: boolean): void;
  setEnabled(enabled: boolean): void;
  stop(): void;
  destroy(): void;
  onUserSpeechStart(): void;
  onUserSpeechEnd(): void;
  onAISpeechStart(): void;
  onAISpeechEnd(): void;
  onGeminiSpeechStart(): void;
  onGeminiSpeechEnd(): void;
  setPipelineActive(active: boolean): void;
  reset(forceImmediate?: boolean): void;
}

export default SmartAIDucker;

declare global {
  interface Window {
    CowatcherAudio?: {
      SmartAIDucker?: typeof SmartAIDucker;
      MIC_CONSTRAINTS?: MediaStreamConstraints;
      UNIFIED_SAMPLE_RATE?: number;
      USER_DUCK_VOLUME?: number;
      GEMINI_DUCK_VOLUME?: number;
      USER_CUT_SEC?: number;
      GEMINI_DUCK_SEC?: number;
      DEFAULTS?: Record<string, number>;
      timeConstantFor?: typeof timeConstantFor;
      findAirPodsOutput?: typeof findAirPodsOutput;
    };
  }
}
