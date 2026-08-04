export type SmartAIDuckerOptions = {
  duckGain?: number;
  fullGain?: number;
  duckAttackSec?: number;
  recoverySec?: number;
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

export declare const MIC_CONSTRAINTS: MediaStreamConstraints;
export declare function timeConstantFor(durationSec: number): number;

export declare class SmartAIDucker {
  static readonly MIC_CONSTRAINTS: MediaStreamConstraints;
  readonly mediaElement: HTMLMediaElement;
  ctx: AudioContext | null;
  programGain: GainNode | null;
  micStream: MediaStream | null;

  constructor(mediaElement: HTMLMediaElement, options?: SmartAIDuckerOptions);

  readonly isDucked: boolean;
  readonly isUserSpeaking: boolean;
  readonly isAISpeaking: boolean;
  readonly currentGain: number;

  getState(): SmartAIDuckerState;
  prepareMediaCors(): void;
  start(opts?: { openMic?: boolean }): Promise<void>;
  ensureMicStream(): Promise<MediaStream | null>;
  setMicMuted(muted: boolean): void;
  setEnabled(enabled: boolean): void;
  stop(): void;
  destroy(): void;
  onUserSpeechStart(): void;
  onUserSpeechEnd(): void;
  onAISpeechStart(): void;
  onAISpeechEnd(): void;
  setPipelineActive(active: boolean): void;
  reset(forceImmediate?: boolean): void;
}

export default SmartAIDucker;

declare global {
  interface Window {
    CowatcherAudio?: {
      SmartAIDucker?: typeof SmartAIDucker;
      MIC_CONSTRAINTS?: MediaStreamConstraints;
      DEFAULTS?: Record<string, number>;
      timeConstantFor?: typeof timeConstantFor;
    };
  }
}
